"""
SensSmartTech 데이터셋 전처리 스크립트

이 스크립트는 SensSmartTech 데이터셋에서 ECG와 PPG 데이터를 
추출하고 전처리하여 레코드별로 저장합니다.

주요 기능:
1. ECG 데이터: Lead I > V4 > V3 우선순위로 채널 선택
2. PPG 데이터: 녹색광 > 적색광 > 적외선, 손목 > 경동맥 우선순위
3. 시간 칼럼 통일 (time), 데이터 칼럼 통일 (ecg, ppg)
4. 레코드별 폴더에 CSV 파일 생성
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장
import wfdb


class SensSmartPreprocessor:
    """SensSmartTech 데이터셋 전처리 클래스"""
    
    # ECG 채널 우선순위 매핑 (Lead I > V4 > V3)
    ECG_PRIORITY = {
        'Lead_I': 0,      # 사지 유도 Lead I
        'I': 0,           # 사지 유도 I 
        'V4': 1,          # 흉부 유도 V4
        'V3': 2,          # 흉부 유도 V3
        'II': 3,          # 사지 유도 II (백업)
        'III': 4,         # 사지 유도 III (백업)
    }
    
    # PPG 채널 우선순위 매핑
    # 녹색광(Green) > 적색광(Red/660nm) > 적외선(IR/880nm)
    # 손목(brachial) > 경동맥(carotid)
    PPG_PRIORITY = {
        # 녹색광 (가장 우선 - 하지만 이 데이터셋에는 없음)
        'brachial_green': 0,
        'brachial_grn': 1,
        'carotid_green': 2,
        'carotid_grn': 3,
        
        # 적색광 (660nm)
        'brachial_red': 10,
        'brachial_660nm': 11,
        'brachial_660': 12,
        'carotid_red': 13,
        'carotid_660nm': 14,
        'carotid_660': 15,
        
        # 적외선 (880nm)
        'brachial_ir': 20,
        'brachial_880nm': 21,
        'brachial_880': 22,
        'brachial_infrared': 23,
        'carotid_ir': 24,
        'carotid_880nm': 25,
        'carotid_880': 26,
        'carotid_infrared': 27,
    }
    
    def __init__(self, input_dir: str, output_dir: str,
                 apply_denoising: bool = True,
                 wavelet: str = 'db4',
                 level: int = 5,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None):
        """
        Args:
            input_dir: WFDB 파일이 있는 디렉토리
            output_dir: 출력 디렉토리
            apply_denoising: 웨이블릿 디노이징 적용 여부
            wavelet: 웨이블릿 종류 (db4, sym4, coif4 등)
            level: 분해 레벨 (기본값: 5)
            save_plots: 시각화 저장 여부
            plots_dir: 시각화 저장 디렉토리 (None이면 output_dir/../visualization)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.apply_denoising = apply_denoising
        self.wavelet = wavelet
        self.level = level
        self.save_plots = save_plots
        
        # 시각화 저장 디렉토리
        if self.save_plots:
            if plots_dir is not None:
                self.plots_dir = Path(plots_dir)
            else:
                self.plots_dir = self.output_dir.parent / "visualization"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def wavelet_denoise(self, signal_data: np.ndarray, 
                       wavelet: str = 'db4', 
                       level: int = 5,
                       threshold_mode: str = 'soft',
                       threshold_scale: float = 1.0) -> np.ndarray:
        """
        웨이블릿 변환을 이용한 신호 디노이징 (정규화 기반)
        
        Args:
            signal_data: 입력 신호 (1D numpy array)
            wavelet: 웨이블릿 종류 (db4: Daubechies 4)
            level: 분해 레벨 (높을수록 더 많은 주파수 대역 분석)
            threshold_mode: 임계값 처리 방식 ('soft' 또는 'hard')
            threshold_scale: threshold 스케일 조정 (기본 1.0, 작을수록 더 많이 제거)
            
        Returns:
            디노이징된 신호
        """
        # 신호 정규화 (평균=0, 표준편차=1)
        signal_mean = np.mean(signal_data)
        signal_std = np.std(signal_data)
        
        if signal_std < 1e-10:  # 거의 상수 신호
            return signal_data
        
        normalized_signal = (signal_data - signal_mean) / signal_std
        
        # 웨이블릿 분해
        coeffs = pywt.wavedec(normalized_signal, wavelet, level=level)
        
        # 노이즈 레벨 추정 (MAD: Median Absolute Deviation)
        # 가장 고주파 디테일 계수로부터 추정
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold 계산 (스케일 조정)
        threshold = sigma * np.sqrt(2 * np.log(len(normalized_signal))) * threshold_scale
        
        # 각 레벨의 디테일 계수에 임계값 적용
        # 첫 번째 계수(근사 계수)는 유지
        coeffs_thresholded = [coeffs[0]]
        for coeff in coeffs[1:]:
            coeffs_thresholded.append(
                pywt.threshold(coeff, threshold, mode=threshold_mode)
            )
        
        # 웨이블릿 재구성
        denoised_normalized = pywt.waverec(coeffs_thresholded, wavelet)
        
        # 길이 조정 (재구성 시 길이가 약간 달라질 수 있음)
        if len(denoised_normalized) > len(normalized_signal):
            denoised_normalized = denoised_normalized[:len(normalized_signal)]
        elif len(denoised_normalized) < len(normalized_signal):
            denoised_normalized = np.pad(denoised_normalized, 
                                         (0, len(normalized_signal) - len(denoised_normalized)), 
                                         mode='edge')
        
        # 역정규화: 원래 스케일로 복원
        denoised_signal = denoised_normalized * signal_std + signal_mean
        
        return denoised_signal
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, 
                             lowcut: float, highcut: float, 
                             fs: float, order: int = 4) -> np.ndarray:
        """
        Butterworth 밴드패스 필터 적용 (SOS 방식 - 수치적으로 안정적)
        
        Args:
            signal_data: 입력 신호
            lowcut: 저주파 컷오프 (Hz)
            highcut: 고주파 컷오프 (Hz)
            fs: 샘플링 주파수 (Hz)
            order: 필터 차수
            
        Returns:
            필터링된 신호
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # SOS (Second-Order Sections) 방식 사용 - 수치적으로 더 안정적
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered_signal = signal.sosfiltfilt(sos, signal_data)
        
        return filtered_signal
    
    def visualize_signal_comparison(self, record_name: str,
                                    signal_type: str,
                                    time: np.ndarray,
                                    original: np.ndarray,
                                    denoised: np.ndarray,
                                    window_seconds: float = 10.0) -> None:
        """
        전처리 전후 신호 비교 시각화 (시작/중간/끝 10초)
        
        Args:
            record_name: 레코드 ID
            signal_type: 'ECG' 또는 'PPG'
            time: 시간 배열 (초)
            original: 원본 신호
            denoised: 디노이징된 신호
            window_seconds: 표시할 시간 윈도우 (초)
        """
        if not self.save_plots:
            return
        
        total_duration = time[-1]
        mid_point = total_duration / 2
        
        # 시작, 중간, 끝 10초 인덱스 찾기
        start_mask = time <= window_seconds
        mid_mask = (time >= (mid_point - window_seconds/2)) & (time <= (mid_point + window_seconds/2))
        end_mask = time >= (total_duration - window_seconds)
        
        # 서브플롯 생성 (3열 2행)
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f'{record_name} - {signal_type} Signal: Before vs After Denoising', 
                     fontsize=16, fontweight='bold')
        
        # 좌측: 시작 10초
        axes[0, 0].plot(time[start_mask], original[start_mask], 'b-', linewidth=0.8, alpha=0.7)
        axes[0, 0].set_title(f'Original Signal - First {window_seconds}s', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel(f'{signal_type} Signal')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, window_seconds)
        
        axes[1, 0].plot(time[start_mask], denoised[start_mask], 'r-', linewidth=1.0)
        axes[1, 0].set_title(f'Denoised Signal - First {window_seconds}s', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel(f'{signal_type} Signal')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, window_seconds)
        
        # 중앙: 중간 10초
        mid_time_start = mid_point - window_seconds/2
        mid_time_end = mid_point + window_seconds/2
        
        axes[0, 1].plot(time[mid_mask], original[mid_mask], 'b-', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_title(f'Original Signal - Middle {window_seconds}s', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel(f'{signal_type} Signal')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(mid_time_start, mid_time_end)
        
        axes[1, 1].plot(time[mid_mask], denoised[mid_mask], 'r-', linewidth=1.0)
        axes[1, 1].set_title(f'Denoised Signal - Middle {window_seconds}s', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel(f'{signal_type} Signal')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(mid_time_start, mid_time_end)
        
        # 우측: 마지막 10초
        end_time_start = total_duration - window_seconds
        
        axes[0, 2].plot(time[end_mask], original[end_mask], 'b-', linewidth=0.8, alpha=0.7)
        axes[0, 2].set_title(f'Original Signal - Last {window_seconds}s', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel(f'{signal_type} Signal')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlim(end_time_start, total_duration)
        
        axes[1, 2].plot(time[end_mask], denoised[end_mask], 'r-', linewidth=1.0)
        axes[1, 2].set_title(f'Denoised Signal - Last {window_seconds}s', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel(f'{signal_type} Signal')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlim(end_time_start, total_duration)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        plot_filename = self.plots_dir / f"{record_name}_{signal_type}_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 {record_name}: {signal_type} 시각화 저장 완료 → {plot_filename.name}")
    
    def select_best_ecg_channel(self, record, sig_names: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """
        우선순위에 따라 최적의 ECG 채널 선택
        
        Args:
            record: WFDB 레코드
            sig_names: 채널 이름 리스트
            
        Returns:
            (channel_index, channel_name) 또는 (None, None)
        """
        best_priority = float('inf')
        best_channel = None
        best_name = None
        
        for idx, name in enumerate(sig_names):
            # 채널 이름에서 ECG 관련 키워드 찾기
            name_upper = name.upper()
            
            for ecg_key, priority in self.ECG_PRIORITY.items():
                if ecg_key.upper() in name_upper:
                    if priority < best_priority:
                        best_priority = priority
                        best_channel = idx
                        best_name = name
                        break
        
        return best_channel, best_name
    
    def select_best_ppg_channel(self, record, sig_names: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """
        우선순위에 따라 최적의 PPG 채널 선택
        
        녹색광(Green) > 적색광(Red/660nm) > 적외선(IR/880nm)
        손목(brachial) > 경동맥(carotid)
        
        Args:
            record: WFDB 레코드
            sig_names: 채널 이름 리스트
            
        Returns:
            (channel_index, channel_name) 또는 (None, None)
        """
        best_priority = float('inf')
        best_channel = None
        best_name = None
        
        for idx, name in enumerate(sig_names):
            name_lower = name.lower()
            
            # PPG 관련 키워드 확인 (더 넓은 범위)
            # 'ppg', 'pleth', 'photopleth' 등 모두 인식
            is_ppg = any(keyword in name_lower for keyword in 
                        ['ppg', 'pleth', 'photo', 'pulse'])
            
            # PPG 파일이면 모든 채널이 PPG일 가능성이 높음
            # 또는 carotid/brachial이 포함된 경우도 PPG
            if not is_ppg:
                if 'brachial' in name_lower or 'carotid' in name_lower:
                    is_ppg = True
            
            if not is_ppg:
                continue
            
            # 부위와 파장 확인
            location = None
            wavelength = None
            
            if 'brachial' in name_lower or 'wrist' in name_lower or 'arm' in name_lower:
                location = 'brachial'
            elif 'carotid' in name_lower or 'neck' in name_lower:
                location = 'carotid'
            
            # 파장 인식 (nm 포함)
            if 'green' in name_lower or 'grn' in name_lower or '525' in name_lower or '565' in name_lower:
                wavelength = 'green'
            elif 'red' in name_lower or '660' in name_lower:
                wavelength = 'red'
            elif 'ir' in name_lower or 'infrared' in name_lower or '880' in name_lower:
                wavelength = 'ir'
            
            # nm 단위 파장 직접 인식
            if '660nm' in name_lower or '660' in name_lower:
                wavelength = '660nm'
            elif '880nm' in name_lower or '880' in name_lower:
                wavelength = '880nm'
            
            # 우선순위 결정
            if location and wavelength:
                ppg_key = f"{location}_{wavelength}"
                if ppg_key in self.PPG_PRIORITY:
                    priority = self.PPG_PRIORITY[ppg_key]
                    if priority < best_priority:
                        best_priority = priority
                        best_channel = idx
                        best_name = name
            elif is_ppg:
                # 부위나 파장을 명확히 알 수 없는 경우 첫 번째 PPG 채널 사용
                if best_channel is None:
                    best_channel = idx
                    best_name = name
        
        return best_channel, best_name
    
    def preprocess_record_combined(self, record_name: str, 
                                   ecg_path: Path, 
                                   ppg_path: Path) -> None:
        """
        ECG와 PPG를 함께 처리하여 하나의 레코드 폴더에 저장
        
        Args:
            record_name: 레코드명 (예: 1_10-09-54)
            ecg_path: ECG .hea 파일 경로
            ppg_path: PPG .hea 파일 경로
        """
        print(f"\n[{record_name}] 전처리 시작...")
        
        # 레코드별 출력 폴더 생성
        record_output_dir = self.output_dir / record_name
        record_output_dir.mkdir(parents=True, exist_ok=True)
        
        ecg_data = None
        ppg_data = None
        time_ecg = None
        time_ppg = None
        
        # ECG 처리
        if ecg_path.exists():
            try:
                record = wfdb.rdrecord(str(ecg_path.parent / ecg_path.stem))
                fs = record.fs
                sig_names = record.sig_name
                signals = record.p_signal
                
                print(f"  📊 ECG 파일 구조:")
                print(f"     - 샘플링 주파수: {fs} Hz")
                print(f"     - 총 샘플 수: {len(signals)}")
                print(f"     - 채널 수: {len(sig_names)}")
                print(f"     - 채널 이름: {sig_names}")
                
                # ECG 채널 선택
                ecg_channel, ecg_name = self.select_best_ecg_channel(record, sig_names)
                
                if ecg_channel is not None:
                    print(f"  ✓ ECG 채널 선택: {ecg_name} (채널 {ecg_channel})")
                    ecg_signal = signals[:, ecg_channel]
                    time_ecg = np.arange(len(signals)) / fs
                    
                    if self.apply_denoising:
                        # ECG: 0.5-40 Hz 밴드패스 필터
                        filtered_ecg = self.apply_bandpass_filter(
                            ecg_signal, lowcut=0.5, highcut=40, fs=fs
                        )
                        denoised_ecg = self.wavelet_denoise(
                            filtered_ecg, wavelet=self.wavelet, level=self.level
                        )
                        
                        # 시각화
                        self.visualize_signal_comparison(
                            record_name, 'ECG', time_ecg, ecg_signal, denoised_ecg, 10.0
                        )
                        
                        ecg_data = denoised_ecg
                    else:
                        ecg_data = ecg_signal
                    
                    # ECG 저장
                    ecg_df = pd.DataFrame({
                        'time': time_ecg,
                        'ecg': ecg_data
                    })
                    ecg_output_path = record_output_dir / "ecg.csv"
                    ecg_df.to_csv(ecg_output_path, index=False)
                    print(f"  💾 ECG 저장: {ecg_output_path}")
                else:
                    print(f"  ⚠️  ECG 채널을 찾을 수 없습니다.")
            
            except Exception as e:
                print(f"  ❌ ECG 처리 중 오류: {str(e)}")
        else:
            print(f"  ⚠️  ECG 파일이 존재하지 않습니다.")
        
        # PPG 처리
        if ppg_path.exists():
            try:
                record = wfdb.rdrecord(str(ppg_path.parent / ppg_path.stem))
                fs = record.fs
                sig_names = record.sig_name
                signals = record.p_signal
                
                print(f"  📊 PPG 파일 구조:")
                print(f"     - 샘플링 주파수: {fs} Hz")
                print(f"     - 총 샘플 수: {len(signals)}")
                print(f"     - 채널 수: {len(sig_names)}")
                print(f"     - 채널 이름: {sig_names}")
                
                # PPG 채널 선택
                ppg_channel, ppg_name = self.select_best_ppg_channel(record, sig_names)
                
                if ppg_channel is not None:
                    print(f"  ✓ PPG 채널 선택: {ppg_name} (채널 {ppg_channel})")
                    ppg_signal = signals[:, ppg_channel]
                    time_ppg = np.arange(len(signals)) / fs
                    
                    if self.apply_denoising:
                        # PPG: 0.1-8 Hz 밴드패스 필터 (저주파 성분 보존)
                        filtered_ppg = self.apply_bandpass_filter(
                            ppg_signal, lowcut=0.1, highcut=8, fs=fs
                        )
                        # PPG는 더 약한 threshold 사용 (threshold_scale=0.5)
                        denoised_ppg = self.wavelet_denoise(
                            filtered_ppg, wavelet=self.wavelet, level=self.level,
                            threshold_scale=0.5
                        )
                        
                        # 시각화
                        self.visualize_signal_comparison(
                            record_name, 'PPG', time_ppg, ppg_signal, denoised_ppg, 10.0
                        )
                        
                        ppg_data = denoised_ppg
                    else:
                        ppg_data = ppg_signal
                    
                    # PPG 저장
                    ppg_df = pd.DataFrame({
                        'time': time_ppg,
                        'ppg': ppg_data
                    })
                    ppg_output_path = record_output_dir / "ppg.csv"
                    ppg_df.to_csv(ppg_output_path, index=False)
                    print(f"  💾 PPG 저장: {ppg_output_path}")
                else:
                    print(f"  ⚠️  PPG 채널을 찾을 수 없습니다.")
            
            except Exception as e:
                print(f"  ❌ PPG 처리 중 오류: {str(e)}")
        else:
            print(f"  ⚠️  PPG 파일이 존재하지 않습니다.")
        
        if ecg_data is not None or ppg_data is not None:
            print(f"  ✓ {record_name} 처리 완료")
        else:
            print(f"  ❌ {record_name}: ECG 및 PPG 데이터를 모두 찾을 수 없습니다.")
    
    
    def process_all_records(self) -> None:
        """모든 WFDB 레코드 처리"""
        # .hea 파일 찾기
        hea_files = sorted(self.input_dir.glob("*.hea"))
        
        if not hea_files:
            print(f"❌ {self.input_dir}에서 .hea 파일을 찾을 수 없습니다.")
            return
        
        # 레코드명 추출 (중복 제거)
        record_names = set()
        for hea_file in hea_files:
            record_name = hea_file.stem
            # _ecg, _ppg 제거하여 실제 레코드명만 추출
            if record_name.endswith('_ecg') or record_name.endswith('_ppg'):
                record_name = record_name.rsplit('_', 1)[0]
            record_names.add(record_name)
        
        record_names = sorted(record_names)
        
        print(f"\n{'='*70}")
        print(f"SensSmartTech 데이터셋 전처리 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 레코드 수: {len(record_names)}")
        print(f"{'='*70}\n")
        
        success_count = 0
        
        for record_name in record_names:
            # ECG와 PPG 파일 경로 확인
            ecg_path = self.input_dir / f"{record_name}_ecg.hea"
            ppg_path = self.input_dir / f"{record_name}_ppg.hea"
            
            # 적어도 하나는 존재해야 함
            if ecg_path.exists() or ppg_path.exists():
                self.preprocess_record_combined(record_name, ecg_path, ppg_path)
                success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*70}")
        print(f"전처리 완료!")
        print(f"{'='*70}")
        print(f"처리한 레코드: {success_count}/{len(record_names)}")
        print(f"출력 위치: {self.output_dir}")
        if self.save_plots:
            print(f"시각화 위치: {self.plots_dir}")
        print(f"{'='*70}\n")

def main():
    """메인 실행 함수"""
    # 경로 설정
    # 실제 사용 시 아래 경로를 수정하세요
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\senssmarttech-database\senssmarttech-database-of-cardiovascular-signals-synchronously-recorded-by-an-electrocardiograph-phonocardiograph-photoplethysmograph-and-accelerometer-1.0.0\WFDB"  # WFDB 파일 경로
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\senssmarttech"  # 출력 경로
    
    # 웨이블릿 디노이징 설정
    APPLY_DENOISING = True      # True: 디노이징 적용, False: 원본만 저장
    WAVELET_TYPE = 'db4'        # 웨이블릿 종류 (db4, sym4, coif4 등)
    DECOMPOSITION_LEVEL = 5     # 분해 레벨 (3-6 권장)
    
    # 시각화 설정
    SAVE_PLOTS = True           # True: 전처리 전후 비교 그래프 저장
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\plots\senssmarttech"  # None이면 자동 설정 (OUTPUT_DIR/../visualization)
                                # 직접 지정: "/mnt/user-data/outputs/my_plots"
    
    # 전처리 실행
    preprocessor = SensSmartPreprocessor(
        INPUT_DIR, 
        OUTPUT_DIR,
        apply_denoising=APPLY_DENOISING,
        wavelet=WAVELET_TYPE,
        level=DECOMPOSITION_LEVEL,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR
    )
    preprocessor.process_all_records()


if __name__ == "__main__":
    main()
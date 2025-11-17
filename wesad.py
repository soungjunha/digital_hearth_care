"""
WESAD 데이터셋 전처리 스크립트

이 스크립트는 WESAD 데이터셋에서 ECG(chest)와 PPG(wrist) 데이터를 
추출하고 전처리하여 피험자별로 저장합니다.

주요 기능:
1. ECG 데이터 (RespiBAN chest): 700Hz 샘플링
2. PPG 데이터 (Empatica E4 wrist BVP): 64Hz 샘플링
3. 시간 칼럼 통일 (time), 데이터 칼럼 통일 (ecg, ppg)
4. 피험자별 CSV 파일 생성
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장


class WESADPreprocessor:
    """WESAD 데이터셋 전처리 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 apply_denoising: bool = True,
                 wavelet: str = 'db4',
                 level: int = 5,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None):
        """
        Args:
            input_dir: 입력 데이터셋 디렉토리 (WESAD 폴더)
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
                       threshold_mode: str = 'soft') -> np.ndarray:
        """
        웨이블릿 변환을 이용한 신호 디노이징
        
        Args:
            signal_data: 입력 신호 (1D numpy array)
            wavelet: 웨이블릿 종류 (db4: Daubechies 4)
            level: 분해 레벨 (높을수록 더 많은 주파수 대역 분석)
            threshold_mode: 임계값 처리 방식 ('soft' 또는 'hard')
            
        Returns:
            디노이징된 신호
        """
        # 원본 신호의 평균값 저장 (베이스라인 복원용)
        original_mean = np.mean(signal_data)
        
        # 웨이블릿 분해
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # 노이즈 레벨 추정 (MAD: Median Absolute Deviation)
        # 가장 고주파 디테일 계수로부터 추정
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold 계산
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # 각 레벨의 디테일 계수에 임계값 적용
        # 첫 번째 계수(근사 계수)는 유지
        coeffs_thresholded = [coeffs[0]]
        for coeff in coeffs[1:]:
            coeffs_thresholded.append(
                pywt.threshold(coeff, threshold, mode=threshold_mode)
            )
        
        # 웨이블릿 재구성
        denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
        
        # 길이 조정 (재구성 시 길이가 약간 달라질 수 있음)
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]
        elif len(denoised_signal) < len(signal_data):
            denoised_signal = np.pad(denoised_signal, 
                                    (0, len(signal_data) - len(denoised_signal)), 
                                    mode='edge')
        
        # 베이스라인 복원: 원본 평균값을 다시 더함
        denoised_mean = np.mean(denoised_signal)
        denoised_signal = denoised_signal - denoised_mean + original_mean
        
        return denoised_signal
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, 
                             lowcut: float, highcut: float, 
                             fs: float, order: int = 4) -> np.ndarray:
        """
        Butterworth 밴드패스 필터 적용
        
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
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def visualize_signal_comparison(self, subject_id: str,
                                    signal_type: str,
                                    time: np.ndarray,
                                    original: np.ndarray,
                                    denoised: np.ndarray,
                                    window_seconds: float = 10.0) -> None:
        """
        전처리 전후 신호 비교 시각화 (시작/중간/끝 10초)
        
        Args:
            subject_id: 피험자 ID
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
        fig.suptitle(f'{subject_id} - {signal_type} Signal: Before vs After Denoising', 
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
        subject_plot_dir = self.plots_dir / subject_id
        subject_plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = subject_plot_dir / f"{signal_type.lower()}_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 {subject_id}: {signal_type} 시각화 저장 완료 → {plot_filename.name}")
    
    def convert_ecg_to_mv(self, raw_ecg: np.ndarray) -> np.ndarray:
        """
        RespiBAN ECG raw 값을 mV로 변환
        
        Formula: ((signal/chan_bit-0.5)*vcc)
        where vcc=3, chan_bit=2^16
        
        Args:
            raw_ecg: raw ECG 값
            
        Returns:
            mV 단위 ECG
        """
        vcc = 3.0
        chan_bit = 2**16
        ecg_mv = ((raw_ecg / chan_bit - 0.5) * vcc)
        return ecg_mv
        
    def preprocess_ecg(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        RespiBAN chest ECG 데이터 전처리
        
        Args:
            subject_id: 피험자 ID (예: S2, S3, ...)
            
        Returns:
            전처리된 ECG 데이터프레임 (칼럼: time, ecg)
            데이터가 없으면 None 반환
        """
        pkl_path = self.input_dir / subject_id / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            print(f"⚠️  {subject_id}: PKL 파일이 존재하지 않습니다.")
            return None
        
        try:
            # PKL 파일 로드
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # chest 데이터에서 ECG 추출
            if 'signal' not in data or 'chest' not in data['signal']:
                print(f"⚠️  {subject_id}: chest 신호가 존재하지 않습니다.")
                return None
            
            chest_data = data['signal']['chest']
            
            # ECG는 chest 데이터의 특정 칼럼 (README에 따르면 ECG 칼럼 존재)
            # chest는 dictionary 형태: {'ACC': array, 'ECG': array, 'EDA': array, ...}
            if 'ECG' not in chest_data:
                print(f"⚠️  {subject_id}: ECG 칼럼이 존재하지 않습니다.")
                return None
            
            ecg_raw = chest_data['ECG'].flatten()
            
            # Raw 값을 mV로 변환
            ecg_mv = self.convert_ecg_to_mv(ecg_raw)
            
            # 시간 배열 생성 (700Hz 샘플링)
            fs_ecg = 700  # Hz
            time_ecg = np.arange(len(ecg_mv)) / fs_ecg
            
            # 데이터프레임 생성
            df_processed = pd.DataFrame({
                'time': time_ecg,
                'ecg': ecg_mv
            })
            
            # 시간을 0초부터 시작하도록 정규화
            df_processed['time'] = df_processed['time'] - df_processed['time'].iloc[0]
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-40Hz) - ECG 생리학적 범위
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ecg'].values, 
                    lowcut=0.5, 
                    highcut=40, 
                    fs=fs_ecg
                )
                
                # 2. Wavelet denoising
                denoised_signal = self.wavelet_denoise(
                    filtered_signal,
                    wavelet=self.wavelet,
                    level=self.level
                )
                
                # 시각화용 원본 저장
                original_signal = df_processed['ecg'].values.copy()
                
                # 디노이징된 신호를 ecg로 저장 (원본은 저장하지 않음)
                df_processed['ecg'] = denoised_signal
                
                print(f"  → {subject_id}: 웨이블릿 디노이징 적용 완료")
                
                # 3. 시각화 (양끝 10초)
                if self.save_plots:
                    self.visualize_signal_comparison(
                        subject_id=subject_id,
                        signal_type='ECG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임: time, ecg (디노이징된 값)
            df_processed = df_processed[['time', 'ecg']]
            
            print(f"✓ {subject_id}: ECG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {subject_id}: ECG 전처리 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_ppg(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Empatica E4 wrist BVP(PPG) 데이터 전처리
        
        Args:
            subject_id: 피험자 ID (예: S2, S3, ...)
            
        Returns:
            전처리된 PPG 데이터프레임 (칼럼: time, ppg)
            데이터가 없으면 None 반환
        """
        pkl_path = self.input_dir / subject_id / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            print(f"⚠️  {subject_id}: PKL 파일이 존재하지 않습니다.")
            return None
        
        try:
            # PKL 파일 로드
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # wrist 데이터에서 BVP(PPG) 추출
            if 'signal' not in data or 'wrist' not in data['signal']:
                print(f"⚠️  {subject_id}: wrist 신호가 존재하지 않습니다.")
                return None
            
            wrist_data = data['signal']['wrist']
            
            # BVP는 wrist 데이터의 photoplethysmograph (PPG)
            if 'BVP' not in wrist_data:
                print(f"⚠️  {subject_id}: BVP(PPG) 칼럼이 존재하지 않습니다.")
                return None
            
            ppg_data = wrist_data['BVP'].flatten()
            
            # 시간 배열 생성 (64Hz 샘플링)
            fs_ppg = 64  # Hz
            time_ppg = np.arange(len(ppg_data)) / fs_ppg
            
            # 데이터프레임 생성
            df_processed = pd.DataFrame({
                'time': time_ppg,
                'ppg': ppg_data
            })
            
            # 시간을 0초부터 시작하도록 정규화
            df_processed['time'] = df_processed['time'] - df_processed['time'].iloc[0]
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-8Hz) - PPG 생리학적 범위
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ppg'].values, 
                    lowcut=0.5, 
                    highcut=8, 
                    fs=fs_ppg
                )
                
                # 2. Wavelet denoising
                denoised_signal = self.wavelet_denoise(
                    filtered_signal,
                    wavelet=self.wavelet,
                    level=self.level
                )
                
                # 시각화용 원본 저장
                original_signal = df_processed['ppg'].values.copy()
                
                # 디노이징된 신호를 ppg로 저장 (원본은 저장하지 않음)
                df_processed['ppg'] = denoised_signal
                
                print(f"  → {subject_id}: 웨이블릿 디노이징 적용 완료")
                
                # 3. 시각화 (양끝 10초)
                if self.save_plots:
                    self.visualize_signal_comparison(
                        subject_id=subject_id,
                        signal_type='PPG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임: time, ppg (디노이징된 값)
            df_processed = df_processed[['time', 'ppg']]
            
            print(f"✓ {subject_id}: PPG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {subject_id}: PPG 전처리 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_processed_data(self, subject_id: str, 
                          ecg_df: Optional[pd.DataFrame], 
                          ppg_df: Optional[pd.DataFrame]) -> None:
        """
        전처리된 데이터를 파일로 저장
        
        Args:
            subject_id: 피험자 ID
            ecg_df: ECG 데이터프레임
            ppg_df: PPG 데이터프레임
        """
        subject_output_dir = self.output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ECG 데이터 저장
        if ecg_df is not None:
            ecg_output_path = subject_output_dir / "ecg.csv"
            ecg_df.to_csv(ecg_output_path, index=False)
            print(f"  💾 ECG 저장: {ecg_output_path}")
        
        # PPG 데이터 저장
        if ppg_df is not None:
            ppg_output_path = subject_output_dir / "ppg.csv"
            ppg_df.to_csv(ppg_output_path, index=False)
            print(f"  💾 PPG 저장: {ppg_output_path}")
    
    def process_all_subjects(self) -> None:
        """모든 피험자의 데이터 전처리"""
        # 피험자 디렉토리 목록 가져오기 (S2, S3, ..., S17, S1과 S12는 제외)
        subject_dirs = sorted([d for d in self.input_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('S') and 
                              d.name not in ['S1', 'S12']])
        
        print(f"\n{'='*60}")
        print(f"WESAD 데이터셋 전처리 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 피험자 수: {len(subject_dirs)}")
        print(f"{'='*60}\n")
        
        success_count = 0
        ecg_success = 0
        ppg_success = 0
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            print(f"\n[{subject_id}] 전처리 시작...")
            
            # ECG 전처리
            ecg_df = self.preprocess_ecg(subject_id)
            if ecg_df is not None:
                ecg_success += 1
            
            # PPG 전처리
            ppg_df = self.preprocess_ppg(subject_id)
            if ppg_df is not None:
                ppg_success += 1
            
            # 저장
            if ecg_df is not None or ppg_df is not None:
                self.save_processed_data(subject_id, ecg_df, ppg_df)
                success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"전처리 완료!")
        print(f"{'='*60}")
        print(f"성공한 피험자: {success_count}/{len(subject_dirs)}")
        print(f"ECG 데이터 처리: {ecg_success}명")
        print(f"PPG 데이터 처리: {ppg_success}명")
        print(f"출력 위치: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    # 실제 사용 시 아래 경로를 수정하세요
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\WESAD\WESAD"  # WESAD 데이터셋 경로
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\wesad"     # 출력 경로
    
    # 웨이블릿 디노이징 설정
    APPLY_DENOISING = True      # True: 디노이징 적용, False: 원본만 저장
    WAVELET_TYPE = 'db4'        # 웨이블릿 종류 (db4, sym4, coif4 등)
    DECOMPOSITION_LEVEL = 5     # 분해 레벨 (3-6 권장)
    
    # 시각화 설정
    SAVE_PLOTS = True           # True: 전처리 전후 비교 그래프 저장
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\plots\wesad"  # None이면 자동 설정 (OUTPUT_DIR/../visualization)
                                # 직접 지정: "/mnt/user-data/outputs/my_plots"
    
    # 전처리 실행
    preprocessor = WESADPreprocessor(
        INPUT_DIR, 
        OUTPUT_DIR,
        apply_denoising=APPLY_DENOISING,
        wavelet=WAVELET_TYPE,
        level=DECOMPOSITION_LEVEL,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR
    )
    preprocessor.process_all_subjects()


if __name__ == "__main__":
    main()
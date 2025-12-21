"""
GalaxyPPG 데이터셋 전처리 스크립트

이 스크립트는 GalaxyPPG 데이터셋에서 ECG(Polar H10)와 PPG(Galaxy Watch 5) 데이터를 
추출하고 전처리하여 피험자별로 저장합니다.

주요 기능:
1. ECG 데이터 (Polar H10): 130Hz 샘플링
2. PPG 데이터 (Galaxy Watch): 25Hz 샘플링
3. 시간 칼럼 통일 (time), 데이터 칼럼 통일 (ecg, ppg)
4. 피험자별 CSV 파일 생성
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장


class GalaxyPPGPreprocessor:
    """GalaxyPPG 데이터셋 전처리 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 apply_denoising: bool = True,
                 wavelet: str = 'db4',
                 level: int = 5,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None):
        """
        Args:
            input_dir: 입력 데이터셋 디렉토리 (GalaxyPPG/Dataset)
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
    
    def visualize_signal_comparison(self, participant_id: str,
                                    signal_type: str,
                                    time: np.ndarray,
                                    original: np.ndarray,
                                    denoised: np.ndarray,
                                    window_seconds: float = 10.0) -> None:
        """
        전처리 전후 신호 비교 시각화 (시작/중간/끝 10초)
        
        Args:
            participant_id: 피험자 ID
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
        fig.suptitle(f'{participant_id} - {signal_type} Signal: Before vs After Denoising', 
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
        participant_plot_dir = self.plots_dir / participant_id
        participant_plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = participant_plot_dir / f"{signal_type.lower()}_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 {participant_id}: {signal_type} 시각화 저장 완료 → {plot_filename.name}")
        
    def preprocess_ecg(self, participant_id: str) -> Optional[pd.DataFrame]:
        """
        Polar H10의 ECG 데이터 전처리
        
        Args:
            participant_id: 피험자 ID (예: P01, P02, ...)
            
        Returns:
            전처리된 ECG 데이터프레임 (칼럼: time, ecg, ecg_denoised)
            데이터가 없으면 None 반환
        """
        ecg_path = self.input_dir / participant_id / "PolarH10" / "ECG.csv"
        
        if not ecg_path.exists():
            print(f"⚠️  {participant_id}: ECG 파일이 존재하지 않습니다.")
            return None
        
        try:
            # ECG 데이터 로드
            df = pd.read_csv(ecg_path)
            
            # 칼럼명 확인 및 변경
            # phoneTimestamp (ms) -> time (초 단위로 변환)
            # ecg (μV) -> ecg
            df_processed = pd.DataFrame({
                'time': df['phoneTimestamp'] / 1000.0,  # ms -> seconds
                'ecg': df['ecg']  # μV 단위 유지
            })
            
            # 시간 순서대로 정렬
            df_processed = df_processed.sort_values('time').reset_index(drop=True)
            
            # 시간을 0초부터 시작하도록 정규화
            df_processed['time'] = df_processed['time'] - df_processed['time'].iloc[0]
            
            # 중복 제거 (동일 시간에 여러 값이 있는 경우)
            df_processed = df_processed.drop_duplicates(subset=['time'], keep='first')
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-40Hz) - ECG 생리학적 범위
                fs = 130  # ECG 샘플링 레이트 (130Hz)
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ecg'].values, 
                    lowcut=0.5, 
                    highcut=40, 
                    fs=fs
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
                
                print(f"  → {participant_id}: 웨이블릿 디노이징 적용 완료")
                
                # 3. 시각화 (양끝 10초)
                if self.save_plots:
                    self.visualize_signal_comparison(
                        participant_id=participant_id,
                        signal_type='ECG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임: time, ecg (디노이징된 값)
            df_processed = df_processed[['time', 'ecg']]
            
            print(f"✓ {participant_id}: ECG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {participant_id}: ECG 전처리 중 오류 발생 - {str(e)}")
            return None
    
    def preprocess_ppg(self, participant_id: str) -> Optional[pd.DataFrame]:
        """
        Galaxy Watch의 PPG 데이터 전처리
        
        Args:
            participant_id: 피험자 ID (예: P01, P02, ...)
            
        Returns:
            전처리된 PPG 데이터프레임 (칼럼: time, ppg, ppg_denoised)
            데이터가 없으면 None 반환
        """
        ppg_path = self.input_dir / participant_id / "GalaxyWatch" / "PPG.csv"
        
        if not ppg_path.exists():
            print(f"⚠️  {participant_id}: PPG 파일이 존재하지 않습니다.")
            return None
        
        try:
            # PPG 데이터 로드
            df = pd.read_csv(ppg_path)
            
            # 칼럼명 확인 및 변경
            # timestamp (ms) -> time (초 단위로 변환)
            # ppg (ADC value) -> ppg
            df_processed = pd.DataFrame({
                'time': df['timestamp'] / 1000.0,  # ms -> seconds
                'ppg': df['ppg']  # ADC 값 유지
            })
            
            # status 필터링: 모든 데이터 포함 (0, 500, -999 모두)
            # -999: 다른 센서(BIA) 작동 중이지만 PPG 신호는 유효 → 포함
            # 0: 정상 값 → 포함
            # 500: STATUS 인터페이스 미지원 (구형 소프트웨어) → 포함
            # 웨이블릿 디노이징이 신호 품질을 보장하므로 모든 데이터 활용
            if 'status' in df.columns:
                # 실제로는 거의 모든 status를 포함 (비정상적인 값만 제외)
                # 경험적으로 유효한 status: -999, 0, 500
                valid_status = df['status'].isin([-999, 0, 500])
                
                # 추가: 다른 status 값이 있는지 확인
                unique_status = df['status'].unique()
                other_status = [s for s in unique_status if s not in [-999, 0, 500]]
                if other_status:
                    print(f"  ℹ️  {participant_id}: 추가 status 값 발견: {other_status}")
                
                df_processed = df_processed[valid_status].reset_index(drop=True)
                removed = len(df) - len(df_processed)
                if removed > 0:
                    print(f"  → {participant_id}: 비정상 상태 데이터 {removed}개 제거")
            
            # 시간 순서대로 정렬
            df_processed = df_processed.sort_values('time').reset_index(drop=True)
            
            # 시간을 0초부터 시작하도록 정규화
            df_processed['time'] = df_processed['time'] - df_processed['time'].iloc[0]
            
            # 중복 제거
            df_processed = df_processed.drop_duplicates(subset=['time'], keep='first')
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-8Hz) - PPG 생리학적 범위
                fs = 25  # PPG 샘플링 레이트 (25Hz)
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ppg'].values, 
                    lowcut=0.5, 
                    highcut=8, 
                    fs=fs
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
                
                print(f"  → {participant_id}: 웨이블릿 디노이징 적용 완료")
                
                # 3. 시각화 (양끝 10초)
                if self.save_plots:
                    self.visualize_signal_comparison(
                        participant_id=participant_id,
                        signal_type='PPG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임: time, ppg (디노이징된 값)
            df_processed = df_processed[['time', 'ppg']]
            
            print(f"✓ {participant_id}: PPG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {participant_id}: PPG 전처리 중 오류 발생 - {str(e)}")
            return None
    
    def save_processed_data(self, participant_id: str, 
                          ecg_df: Optional[pd.DataFrame], 
                          ppg_df: Optional[pd.DataFrame]) -> None:
        """
        전처리된 데이터를 파일로 저장
        
        Args:
            participant_id: 피험자 ID
            ecg_df: ECG 데이터프레임
            ppg_df: PPG 데이터프레임
        """
        participant_output_dir = self.output_dir / participant_id
        participant_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ECG 데이터 저장
        if ecg_df is not None:
            ecg_output_path = participant_output_dir / "ecg.csv"
            ecg_df.to_csv(ecg_output_path, index=False)
            print(f"  💾 ECG 저장: {ecg_output_path}")
        
        # PPG 데이터 저장
        if ppg_df is not None:
            ppg_output_path = participant_output_dir / "ppg.csv"
            ppg_df.to_csv(ppg_output_path, index=False)
            print(f"  💾 PPG 저장: {ppg_output_path}")
    
    def process_all_participants(self) -> None:
        """모든 피험자의 데이터 전처리"""
        # 피험자 디렉토리 목록 가져오기 (P01, P02, ..., P24)
        participant_dirs = sorted([d for d in self.input_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith('P')])
        
        print(f"\n{'='*60}")
        print(f"GalaxyPPG 데이터셋 전처리 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 피험자 수: {len(participant_dirs)}")
        print(f"{'='*60}\n")
        
        success_count = 0
        ecg_success = 0
        ppg_success = 0
        
        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            print(f"\n[{participant_id}] 전처리 시작...")
            
            # ECG 전처리
            ecg_df = self.preprocess_ecg(participant_id)
            if ecg_df is not None:
                ecg_success += 1
            
            # PPG 전처리
            ppg_df = self.preprocess_ppg(participant_id)
            if ppg_df is not None:
                ppg_success += 1
            
            # 저장
            if ecg_df is not None or ppg_df is not None:
                self.save_processed_data(participant_id, ecg_df, ppg_df)
                success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"전처리 완료!")
        print(f"{'='*60}")
        print(f"성공한 피험자: {success_count}/{len(participant_dirs)}")
        print(f"ECG 데이터 처리: {ecg_success}명")
        print(f"PPG 데이터 처리: {ppg_success}명")
        print(f"출력 위치: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    # 실제 사용 시 아래 경로를 수정하세요
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\GalaxyPPG\Dataset"  # 데이터셋 경로
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\galaxysppg"     # 출력 경로
    
    # 웨이블릿 디노이징 설정
    APPLY_DENOISING = True      # True: 디노이징 적용, False: 원본만 저장
    WAVELET_TYPE = 'db4'        # 웨이블릿 종류 (db4, sym4, coif4 등)
    DECOMPOSITION_LEVEL = 5     # 분해 레벨 (3-6 권장)
    
    # 시각화 설정
    SAVE_PLOTS = True           # True: 전처리 전후 비교 그래프 저장
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\plots\galaxysppg"            # None이면 자동 설정 (OUTPUT_DIR/../visualization)
                                # 직접 지정: "/mnt/user-data/outputs/my_plots"
    
    # 전처리 실행
    preprocessor = GalaxyPPGPreprocessor(
        INPUT_DIR, 
        OUTPUT_DIR,
        apply_denoising=APPLY_DENOISING,
        wavelet=WAVELET_TYPE,
        level=DECOMPOSITION_LEVEL,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR
    )
    preprocessor.process_all_participants()


if __name__ == "__main__":
    main()
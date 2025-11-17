"""
PPG Field Study 데이터셋 전처리 스크립트

이 스크립트는 PPG Field Study 데이터셋의 pkl 파일에서 ECG와 PPG(BVP) 데이터를 
추출하고 GalaxyPPG와 동일한 전처리 파이프라인을 적용합니다.

주요 기능:
1. pkl 파일에서 ECG와 BVP(PPG) 데이터 추출
2. 웨이블릿 디노이징 적용
3. 피험자별 CSV 파일 생성
4. 시작/중간/끝 10초 시각화
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class PPGFieldStudyPreprocessor:
    """PPG Field Study 데이터셋 전처리 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str,
                 apply_denoising: bool = True,
                 wavelet: str = 'db4',
                 level: int = 5,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None):
        """
        Args:
            input_dir: 입력 데이터셋 디렉토리 (PPG_FieldStudy)
            output_dir: 출력 디렉토리
            apply_denoising: 웨이블릿 디노이징 적용 여부
            wavelet: 웨이블릿 종류
            level: 분해 레벨
            save_plots: 시각화 저장 여부
            plots_dir: 시각화 저장 디렉토리
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
                self.plots_dir = self.output_dir.parent / "visualization_ppgfieldstudy"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pkl_file(self, pkl_path: Path) -> Optional[Dict[str, Any]]:
        """
        pkl 파일 로드
        
        Args:
            pkl_path: pkl 파일 경로
            
        Returns:
            pkl 데이터 딕셔너리
        """
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return data
        except Exception as e:
            print(f"❌ pkl 파일 로드 실패: {str(e)}")
            return None
    
    def wavelet_denoise(self, signal_data: np.ndarray, 
                       wavelet: str = 'db4', 
                       level: int = 5,
                       threshold_mode: str = 'soft') -> np.ndarray:
        """
        웨이블릿 변환을 이용한 신호 디노이징 (베이스라인 복원 포함)
        
        Args:
            signal_data: 입력 신호
            wavelet: 웨이블릿 종류
            level: 분해 레벨
            threshold_mode: 임계값 처리 방식
            
        Returns:
            디노이징된 신호
        """
        # 원본 신호의 평균값 저장 (베이스라인 복원용)
        original_mean = np.mean(signal_data)
        
        # 웨이블릿 분해
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # 노이즈 레벨 추정 (MAD)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold 계산
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # 임계값 적용
        coeffs_thresholded = [coeffs[0]]
        for coeff in coeffs[1:]:
            coeffs_thresholded.append(
                pywt.threshold(coeff, threshold, mode=threshold_mode)
            )
        
        # 웨이블릿 재구성
        denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
        
        # 길이 조정
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]
        elif len(denoised_signal) < len(signal_data):
            denoised_signal = np.pad(denoised_signal, 
                                    (0, len(signal_data) - len(denoised_signal)), 
                                    mode='edge')
        
        # 베이스라인 복원
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
    
    def preprocess_ecg(self, participant_id: str, ecg_data: np.ndarray, 
                      sampling_rate: float = 700.0) -> Optional[pd.DataFrame]:
        """
        ECG 데이터 전처리
        
        Args:
            participant_id: 피험자 ID
            ecg_data: ECG 신호 배열
            sampling_rate: 샘플링 레이트 (Hz)
            
        Returns:
            전처리된 DataFrame (time, ecg)
        """
        try:
            # 다차원 배열인 경우 1차원으로 변환
            if len(ecg_data.shape) > 1:
                # 다중 채널이면 첫 번째 채널만 사용
                ecg_data = ecg_data.flatten() if ecg_data.shape[0] == 1 else ecg_data[:, 0]
            
            # 시간 배열 생성 (0초부터 시작)
            time = np.arange(len(ecg_data)) / sampling_rate
            
            # 데이터프레임 생성
            df_processed = pd.DataFrame({
                'time': time,
                'ecg': ecg_data.flatten()  # 확실하게 1차원으로
            })
            
            print(f"  → {participant_id}: ECG 데이터 형태 - {len(df_processed)} 샘플, {df_processed['time'].max():.1f}초")
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-40Hz)
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ecg'].values,
                    lowcut=0.5,
                    highcut=40,
                    fs=sampling_rate
                )
                
                # 2. Wavelet denoising
                denoised_signal = self.wavelet_denoise(
                    filtered_signal,
                    wavelet=self.wavelet,
                    level=self.level
                )
                
                # 시각화용 원본 저장
                original_signal = df_processed['ecg'].values.copy()
                
                # 디노이징된 신호로 교체
                df_processed['ecg'] = denoised_signal
                
                print(f"  → {participant_id}: ECG 웨이블릿 디노이징 적용 완료")
                
                # 시각화
                if self.save_plots:
                    self.visualize_signal_comparison(
                        participant_id=participant_id,
                        signal_type='ECG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임
            df_processed = df_processed[['time', 'ecg']]
            
            print(f"✓ {participant_id}: ECG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {participant_id}: ECG 전처리 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_ppg(self, participant_id: str, ppg_data: np.ndarray,
                      sampling_rate: float = 64.0) -> Optional[pd.DataFrame]:
        """
        PPG(BVP) 데이터 전처리
        
        Args:
            participant_id: 피험자 ID
            ppg_data: PPG(BVP) 신호 배열
            sampling_rate: 샘플링 레이트 (Hz)
            
        Returns:
            전처리된 DataFrame (time, ppg)
        """
        try:
            # 다차원 배열인 경우 1차원으로 변환
            if len(ppg_data.shape) > 1:
                # 다중 채널이면 첫 번째 채널만 사용
                ppg_data = ppg_data.flatten() if ppg_data.shape[0] == 1 else ppg_data[:, 0]
            
            # 시간 배열 생성 (0초부터 시작)
            time = np.arange(len(ppg_data)) / sampling_rate
            
            # 데이터프레임 생성
            df_processed = pd.DataFrame({
                'time': time,
                'ppg': ppg_data.flatten()  # 확실하게 1차원으로
            })
            
            print(f"  → {participant_id}: PPG 데이터 형태 - {len(df_processed)} 샘플, {df_processed['time'].max():.1f}초")
            
            # 웨이블릿 디노이징 적용
            if self.apply_denoising:
                # 1. Bandpass filter (0.5-8Hz)
                filtered_signal = self.apply_bandpass_filter(
                    df_processed['ppg'].values,
                    lowcut=0.5,
                    highcut=8,
                    fs=sampling_rate
                )
                
                # 2. Wavelet denoising
                denoised_signal = self.wavelet_denoise(
                    filtered_signal,
                    wavelet=self.wavelet,
                    level=self.level
                )
                
                # 시각화용 원본 저장
                original_signal = df_processed['ppg'].values.copy()
                
                # 디노이징된 신호로 교체
                df_processed['ppg'] = denoised_signal
                
                print(f"  → {participant_id}: PPG 웨이블릿 디노이징 적용 완료")
                
                # 시각화
                if self.save_plots:
                    self.visualize_signal_comparison(
                        participant_id=participant_id,
                        signal_type='PPG',
                        time=df_processed['time'].values,
                        original=original_signal,
                        denoised=denoised_signal,
                        window_seconds=10.0
                    )
            
            # 최종 데이터프레임
            df_processed = df_processed[['time', 'ppg']]
            
            print(f"✓ {participant_id}: PPG 데이터 {len(df_processed)}개 샘플 전처리 완료")
            return df_processed
            
        except Exception as e:
            print(f"❌ {participant_id}: PPG 전처리 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_processed_data(self, participant_id: str,
                          ecg_df: Optional[pd.DataFrame],
                          ppg_df: Optional[pd.DataFrame]) -> None:
        """
        전처리된 데이터를 CSV 파일로 저장
        
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
    
    def process_participant(self, participant_id: str) -> None:
        """
        특정 피험자의 데이터 처리
        
        Args:
            participant_id: 피험자 ID (예: S1, S2, ...)
        """
        print(f"\n[{participant_id}] 전처리 시작...")
        
        # pkl 파일 경로
        pkl_path = self.input_dir / participant_id / f"{participant_id}.pkl"
        
        if not pkl_path.exists():
            print(f"⚠️  {participant_id}: pkl 파일이 존재하지 않습니다.")
            return
        
        # pkl 파일 로드
        data = self.load_pkl_file(pkl_path)
        if data is None:
            return
        
        # 데이터 구조 확인
        print(f"  → {participant_id}: pkl 파일 로드 완료")
        if 'signal' in data:
            print(f"  → signal keys: {list(data['signal'].keys())}")
            if 'chest' in data['signal']:
                print(f"  → chest keys: {list(data['signal']['chest'].keys())}")
            if 'wrist' in data['signal']:
                print(f"  → wrist keys: {list(data['signal']['wrist'].keys())}")
        
        # ECG 데이터 추출 (chest -> ECG)
        ecg_df = None
        try:
            ecg_signal = data['signal']['chest']['ECG']
            print(f"  → {participant_id}: ECG 데이터 형태 = {ecg_signal.shape}, dtype = {ecg_signal.dtype}")
            ecg_df = self.preprocess_ecg(participant_id, ecg_signal, sampling_rate=700.0)
        except Exception as e:
            print(f"⚠️  {participant_id}: ECG 데이터 추출 실패 - {str(e)}")
            import traceback
            traceback.print_exc()
        
        # PPG(BVP) 데이터 추출 (wrist -> BVP)
        ppg_df = None
        try:
            ppg_signal = data['signal']['wrist']['BVP']
            print(f"  → {participant_id}: PPG 데이터 형태 = {ppg_signal.shape}, dtype = {ppg_signal.dtype}")
            ppg_df = self.preprocess_ppg(participant_id, ppg_signal, sampling_rate=64.0)
        except Exception as e:
            print(f"⚠️  {participant_id}: PPG 데이터 추출 실패 - {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 저장
        if ecg_df is not None or ppg_df is not None:
            self.save_processed_data(participant_id, ecg_df, ppg_df)
    
    def process_all_participants(self) -> None:
        """모든 피험자의 데이터 전처리 (S1~S15)"""
        print(f"\n{'='*60}")
        print(f"PPG Field Study 데이터셋 전처리 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 피험자 수: 15 (S1~S15)")
        print(f"{'='*60}\n")
        
        success_count = 0
        ecg_success = 0
        ppg_success = 0
        
        # S1 ~ S15 처리
        for i in range(1, 16):
            participant_id = f"S{i}"
            
            self.process_participant(participant_id)
            
            # 성공 여부 확인
            participant_output_dir = self.output_dir / participant_id
            if participant_output_dir.exists():
                if (participant_output_dir / "ecg.csv").exists():
                    ecg_success += 1
                if (participant_output_dir / "ppg.csv").exists():
                    ppg_success += 1
                if (participant_output_dir / "ecg.csv").exists() or (participant_output_dir / "ppg.csv").exists():
                    success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"전처리 완료!")
        print(f"{'='*60}")
        print(f"성공한 피험자: {success_count}/15")
        print(f"ECG 데이터 처리: {ecg_success}명")
        print(f"PPG 데이터 처리: {ppg_success}명")
        print(f"출력 위치: {self.output_dir}")
        if self.save_plots:
            print(f"시각화 위치: {self.plots_dir}")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\ppg+dalia\data\PPG_FieldStudy"  # pkl 파일들이 있는 경로
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\ppgdalia"  # 출력 경로
    
    # 웨이블릿 디노이징 설정
    APPLY_DENOISING = True
    WAVELET_TYPE = 'db4'
    DECOMPOSITION_LEVEL = 5
    
    # 시각화 설정
    SAVE_PLOTS = True
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\plots\ppgdalia"  # 시각화 저장 경로
    
    # 전처리 실행
    preprocessor = PPGFieldStudyPreprocessor(
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
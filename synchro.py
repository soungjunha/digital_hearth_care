"""
ECG-PPG 데이터 동기화 스크립트

피험자별로 분류된 ECG와 PPG 데이터를 동기화하여 단일 CSV 파일로 저장합니다.
- 샘플링 레이트: 256Hz로 통일
- ECG와 PPG를 하나의 CSV 파일에 저장 (time, ecg, ppg)
- 파일명: Subject_01, Subject_02, ... (모든 데이터셋 통합하여 번호 매김)
- 시작 시간이 다를 경우 시간 정렬
- 길이가 다를 경우 짧은 쪽에 맞춰 트리밍
- 여러 데이터셋을 자동으로 처리하여 통합 (galaxysppg, mimic, wesad, wildppg 등)

디렉토리 구조:
    입력 (data_set_csv/):
        galaxysppg/
            P02/
                ecg.csv  (time, ecg)
                ppg.csv  (time, ppg)
            P03/
                ecg.csv
                ppg.csv
        mimic/
            mimic_perform_train_all_001/
                ecg.csv
                ppg.csv
        wesad/
            S2/
                ecg.csv
                ppg.csv
    
    출력 (synchronized/):
        Subject_01.csv  (galaxysppg P02 - time, ecg, ppg - 256Hz)
        Subject_02.csv  (galaxysppg P03)
        Subject_03.csv  (galaxysppg P04)
        ...
        Subject_25.csv  (mimic_perform_train_all_001)
        Subject_26.csv  (mimic_perform_train_all_002)
        ...
        Subject_40.csv  (wesad S2)
        ...
    
    시각화 (plots/synchronization/):
        Subject_01_sync_comparison.png
        Subject_02_sync_comparison.png
        ...
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ECGPPGSynchronizer:
    """ECG-PPG 데이터 동기화 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 target_fs: float = 256.0,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None,
                 subject_counter: int = 1):
        """
        Args:
            input_dir: 입력 디렉토리 (피험자별 폴더 포함)
            output_dir: 출력 디렉토리
            target_fs: 목표 샘플링 레이트 (기본값: 256Hz)
            save_plots: 동기화 전후 비교 시각화 저장 여부
            plots_dir: 시각화 저장 디렉토리
            subject_counter: 피험자 번호 시작값 (Subject_01, Subject_02, ...)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_fs = target_fs
        self.save_plots = save_plots
        self.subject_counter = subject_counter
        
        if self.save_plots:
            if plots_dir is not None:
                self.plots_dir = Path(plots_dir)
            else:
                self.plots_dir = self.output_dir / "sync_visualization"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_sampling_rate(self, time: np.ndarray) -> float:
        """
        시간 배열로부터 샘플링 레이트 추정
        
        Args:
            time: 시간 배열 (초)
            
        Returns:
            샘플링 레이트 (Hz)
        """
        time_diff = np.diff(time)
        median_interval = np.median(time_diff)
        fs = 1.0 / median_interval if median_interval > 0 else 1.0
        return fs
    
    def resample_signal(self, time: np.ndarray, signal: np.ndarray, 
                       target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        신호를 목표 샘플링 레이트로 리샘플링
        
        Args:
            time: 원본 시간 배열
            signal: 원본 신호
            target_fs: 목표 샘플링 레이트
            
        Returns:
            (리샘플링된 시간, 리샘플링된 신호)
        """
        # 새로운 시간 배열 생성
        duration = time[-1] - time[0]
        num_samples = int(duration * target_fs)
        new_time = np.linspace(time[0], time[-1], num_samples)
        
        # 스플라인 보간을 사용한 리샘플링
        interpolator = interpolate.interp1d(time, signal, kind='cubic', 
                                           bounds_error=False, fill_value='extrapolate')
        new_signal = interpolator(new_time)
        
        return new_time, new_signal
    
    def align_signals(self, ecg_time: np.ndarray, ecg_signal: np.ndarray,
                     ppg_time: np.ndarray, ppg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ECG와 PPG 신호를 시간적으로 정렬
        
        Args:
            ecg_time: ECG 시간 배열
            ecg_signal: ECG 신호
            ppg_time: PPG 시간 배열
            ppg_signal: PPG 신호
            
        Returns:
            (정렬된 시간, 정렬된 ECG, 정렬된 PPG)
        """
        # 공통 시작 시간과 종료 시간 찾기
        start_time = max(ecg_time[0], ppg_time[0])
        end_time = min(ecg_time[-1], ppg_time[-1])
        
        # ECG 마스크
        ecg_mask = (ecg_time >= start_time) & (ecg_time <= end_time)
        aligned_ecg_time = ecg_time[ecg_mask]
        aligned_ecg_signal = ecg_signal[ecg_mask]
        
        # PPG 마스크
        ppg_mask = (ppg_time >= start_time) & (ppg_time <= end_time)
        aligned_ppg_time = ppg_time[ppg_mask]
        aligned_ppg_signal = ppg_signal[ppg_mask]
        
        return aligned_ecg_time, aligned_ecg_signal, aligned_ppg_time, aligned_ppg_signal
    
    def synchronize_signals(self, ecg_df: pd.DataFrame, ppg_df: pd.DataFrame) -> pd.DataFrame:
        """
        ECG와 PPG 신호 동기화하여 단일 데이터프레임으로 반환
        
        Args:
            ecg_df: ECG 데이터프레임 (time, ecg)
            ppg_df: PPG 데이터프레임 (time, ppg)
            
        Returns:
            동기화된 데이터프레임 (time, ecg, ppg)
        """
        # 1. 샘플링 레이트 추정
        ecg_fs = self.estimate_sampling_rate(ecg_df['time'].values)
        ppg_fs = self.estimate_sampling_rate(ppg_df['time'].values)
        
        print(f"  ℹ️  원본 샘플링 레이트 - ECG: {ecg_fs:.2f} Hz, PPG: {ppg_fs:.2f} Hz")
        
        # 2. 목표 샘플링 레이트 사용
        target_fs = self.target_fs
        
        print(f"  → 목표 샘플링 레이트: {target_fs:.2f} Hz")
        
        # 3. 리샘플링 (필요한 경우)
        ecg_time = ecg_df['time'].values
        ecg_signal = ecg_df['ecg'].values
        ppg_time = ppg_df['time'].values
        ppg_signal = ppg_df['ppg'].values
        
        if abs(ecg_fs - target_fs) > 1.0:  # 1Hz 이상 차이나면 리샘플링
            print(f"  → ECG 리샘플링 중... ({ecg_fs:.2f} Hz → {target_fs:.2f} Hz)")
            ecg_time, ecg_signal = self.resample_signal(ecg_time, ecg_signal, target_fs)
        
        if abs(ppg_fs - target_fs) > 1.0:
            print(f"  → PPG 리샘플링 중... ({ppg_fs:.2f} Hz → {target_fs:.2f} Hz)")
            ppg_time, ppg_signal = self.resample_signal(ppg_time, ppg_signal, target_fs)
        
        # 4. 시간 정렬 (공통 시간 구간 추출)
        ecg_time_aligned, ecg_signal_aligned, ppg_time_aligned, ppg_signal_aligned = \
            self.align_signals(ecg_time, ecg_signal, ppg_time, ppg_signal)
        
        # 5. 길이 맞추기 (더 짧은 쪽에 맞춤)
        min_length = min(len(ecg_time_aligned), len(ppg_time_aligned))
        
        ecg_time_final = ecg_time_aligned[:min_length]
        ecg_signal_final = ecg_signal_aligned[:min_length]
        ppg_signal_final = ppg_signal_aligned[:min_length]
        
        # 6. 시간을 0부터 시작하도록 정규화
        time_offset = ecg_time_final[0]
        ecg_time_final = ecg_time_final - time_offset
        
        # 7. 단일 데이터프레임 생성 (time, ecg, ppg)
        combined_df = pd.DataFrame({
            'time': ecg_time_final,
            'ecg': ecg_signal_final,
            'ppg': ppg_signal_final
        })
        
        print(f"  ✓ 동기화 완료 - 최종 샘플 수: {min_length}, 지속 시간: {ecg_time_final[-1]:.2f}초")
        
        return combined_df
    
    def visualize_synchronization(self, participant_id: str,
                                  ecg_before: pd.DataFrame, ppg_before: pd.DataFrame,
                                  combined_after: pd.DataFrame) -> None:
        """
        동기화 전후 비교 시각화 (처음 30초만 표시)
        
        Args:
            participant_id: 피험자 ID
            ecg_before: 동기화 전 ECG
            ppg_before: 동기화 전 PPG
            combined_after: 동기화 후 통합 데이터 (time, ecg, ppg)
        """
        if not self.save_plots:
            return
        
        # 처음 30초만 표시
        window_seconds = 30.0
        
        # Before
        ecg_before_mask = ecg_before['time'] <= window_seconds
        ppg_before_mask = ppg_before['time'] <= window_seconds
        
        # After
        after_mask = combined_after['time'] <= window_seconds
        
        # 플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{participant_id} - Signal Synchronization: Before vs After', 
                     fontsize=16, fontweight='bold')
        
        # Before - ECG
        axes[0, 0].plot(ecg_before['time'][ecg_before_mask], 
                       ecg_before['ecg'][ecg_before_mask], 'b-', linewidth=0.8)
        axes[0, 0].set_title('Before Sync - ECG', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('ECG Signal')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, window_seconds)
        
        # Before - PPG
        axes[0, 1].plot(ppg_before['time'][ppg_before_mask], 
                       ppg_before['ppg'][ppg_before_mask], 'g-', linewidth=0.8)
        axes[0, 1].set_title('Before Sync - PPG', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('PPG Signal')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, window_seconds)
        
        # After - ECG
        axes[1, 0].plot(combined_after['time'][after_mask], 
                       combined_after['ecg'][after_mask], 'r-', linewidth=0.8)
        axes[1, 0].set_title('After Sync - ECG', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('ECG Signal')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, window_seconds)
        
        # After - PPG
        axes[1, 1].plot(combined_after['time'][after_mask], 
                       combined_after['ppg'][after_mask], 'm-', linewidth=0.8)
        axes[1, 1].set_title('After Sync - PPG', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('PPG Signal')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, window_seconds)
        
        plt.tight_layout()
        
        # 저장 (폴더 구분 없이 바로 plots 디렉토리에 저장)
        plot_filename = self.plots_dir / f"{participant_id}_sync_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 {participant_id}: 동기화 시각화 저장 완료")
    
    def process_participant(self, participant_dir: Path) -> bool:
        """
        단일 피험자 데이터 동기화
        
        Args:
            participant_dir: 피험자 디렉토리
            
        Returns:
            성공 여부
        """
        participant_id = participant_dir.name
        ecg_path = participant_dir / "ecg.csv"
        ppg_path = participant_dir / "ppg.csv"
        
        # 파일 존재 확인
        if not ecg_path.exists():
            print(f"⚠️  {participant_id}: ecg.csv 파일이 존재하지 않습니다.")
            return False
        
        if not ppg_path.exists():
            print(f"⚠️  {participant_id}: ppg.csv 파일이 존재하지 않습니다.")
            return False
        
        try:
            # 데이터 로드
            ecg_df = pd.read_csv(ecg_path)
            ppg_df = pd.read_csv(ppg_path)
            
            print(f"  ✓ 데이터 로드 - ECG: {len(ecg_df)} 샘플, PPG: {len(ppg_df)} 샘플")
            
            # 칼럼 확인
            if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
                print(f"⚠️  {participant_id}: ECG 파일에 'time', 'ecg' 칼럼이 필요합니다.")
                return False
            
            if 'time' not in ppg_df.columns or 'ppg' not in ppg_df.columns:
                print(f"⚠️  {participant_id}: PPG 파일에 'time', 'ppg' 칼럼이 필요합니다.")
                return False
            
            # 동기화 전 데이터 복사 (시각화용)
            ecg_before = ecg_df.copy()
            ppg_before = ppg_df.copy()
            
            # 동기화 수행 (단일 데이터프레임 반환)
            combined_df = self.synchronize_signals(ecg_df, ppg_df)
            
            # Subject_XX 형식의 파일명 생성
            subject_name = f"Subject_{self.subject_counter:02d}"
            
            # 시각화
            if self.save_plots:
                self.visualize_synchronization(subject_name, ecg_before, ppg_before, 
                                              combined_df)
            
            # 저장 (Subject_XX.csv 형식으로 폴더 구분 없이 바로 저장)
            combined_output_path = self.output_dir / f"{subject_name}.csv"
            combined_df.to_csv(combined_output_path, index=False)
            
            print(f"  💾 저장 완료: {combined_output_path.name} (원본: {participant_id})")
            print(f"  📊 데이터 형식: time, ecg, ppg ({len(combined_df)} 샘플)")
            
            # 카운터 증가
            self.subject_counter += 1
            
            return True
            
        except Exception as e:
            print(f"❌ {participant_id}: 동기화 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_participants(self) -> None:
        """모든 피험자 데이터 동기화"""
        # 피험자 디렉토리 목록 가져오기
        participant_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        
        print(f"\n{'='*60}")
        print(f"ECG-PPG 데이터 동기화 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 피험자 수: {len(participant_dirs)}")
        print(f"목표 샘플링 레이트: {self.target_fs} Hz")
        print(f"출력 형식: 단일 CSV 파일 (time, ecg, ppg)")
        print(f"{'='*60}\n")
        
        success_count = 0
        
        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            print(f"\n[{participant_id}] 동기화 시작...")
            
            if self.process_participant(participant_dir):
                success_count += 1
                print(f"✓ {participant_id}: 동기화 완료")
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"동기화 완료!")
        print(f"{'='*60}")
        print(f"성공한 피험자: {success_count}/{len(participant_dirs)}")
        print(f"출력 위치: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    # data_set_csv 내의 각 데이터셋 폴더를 처리
    BASE_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv"  # 기본 디렉토리
    
    # 처리할 데이터셋 목록
    DATASETS = ['galaxysppg', 'mimic', 'ppgdalia', 'senssmarttech', 'wesad', 'wildppg']
    
    # 출력 디렉토리
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\synchronized"
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\synchronized\plots\synchronization"
    
    # 동기화 설정
    TARGET_FS = 256.0  # 256Hz로 통일
    SAVE_PLOTS = True  # True: 동기화 전후 비교 시각화 저장
    
    # Subject 카운터 초기화
    subject_counter = 1
    
    # 각 데이터셋별로 처리
    for dataset in DATASETS:
        input_dir = os.path.join(BASE_DIR, dataset)
        
        # 디렉토리가 존재하지 않으면 스킵
        if not os.path.exists(input_dir):
            print(f"\n⚠️  {dataset} 디렉토리가 존재하지 않습니다. 스킵합니다.")
            continue
        
        print(f"\n{'='*70}")
        print(f"데이터셋: {dataset.upper()}")
        print(f"현재 Subject 번호: {subject_counter}")
        print(f"{'='*70}")
        
        # 동기화 실행 (카운터를 전달하여 연속적으로 번호 매기기)
        synchronizer = ECGPPGSynchronizer(
            input_dir,
            OUTPUT_DIR,  # 모든 데이터셋이 같은 출력 디렉토리 사용
            target_fs=TARGET_FS,
            save_plots=SAVE_PLOTS,
            plots_dir=PLOTS_DIR,
            subject_counter=subject_counter
        )
        synchronizer.process_all_participants()
        
        # 다음 데이터셋을 위해 카운터 업데이트
        subject_counter = synchronizer.subject_counter
    
    print(f"\n{'='*70}")
    print(f"전체 데이터셋 동기화 완료!")
    print(f"총 피험자 수: {subject_counter - 1}명")
    print(f"출력 위치: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
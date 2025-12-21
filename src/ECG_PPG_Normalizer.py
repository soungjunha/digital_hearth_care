"""
ECG-PPG 데이터 Min-Max 정규화 스크립트

동기화된 ECG와 PPG 데이터를 Min-Max 정규화합니다.
- Min-Max 정규화: 0~1 범위로 스케일링
- Formula: (x - min) / (max - min)
- CNN-GRU 오토인코더 기반 심장질환 이상징후 탐지에 최적화

특징:
- ECG와 PPG를 개별적으로 정규화 (서로 다른 스케일)
- 각 피험자별로 독립적으로 정규화 (개인차 고려)
- 신호의 형태와 상대적 진폭 보존
- [0, 1] 범위로 통일하여 딥러닝 모델 학습에 적합

입력: synchronized/ 폴더의 Subject_XX.csv 파일들
출력: normalized/ 폴더에 정규화된 파일 저장
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ECGPPGNormalizer:
    """ECG-PPG 데이터 Min-Max 정규화 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None,
                 clip_values: bool = True):
        """
        Args:
            input_dir: 입력 디렉토리 (동기화된 데이터)
            output_dir: 출력 디렉토리
            save_plots: 정규화 전후 비교 시각화 저장 여부
            plots_dir: 시각화 저장 디렉토리
            clip_values: 극단적 이상치 클리핑 여부 (0~1 범위 강제)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_plots = save_plots
        self.clip_values = clip_values
        
        # 시각화 저장 디렉토리
        if self.save_plots:
            if plots_dir is not None:
                self.plots_dir = Path(plots_dir)
            else:
                self.plots_dir = self.output_dir / "visualization"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def minmax_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Min-Max 정규화: [0, 1] 범위로 스케일링
        
        Formula: (x - min) / (max - min)
        
        Args:
            signal: 입력 신호
            
        Returns:
            정규화된 신호 (0~1 범위)
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        # 모든 값이 동일한 경우 (상수 신호)
        if max_val - min_val == 0:
            print("  ⚠️  경고: 신호의 모든 값이 동일합니다. 0으로 설정합니다.")
            return np.zeros_like(signal)
        
        # Min-Max 정규화
        normalized = (signal - min_val) / (max_val - min_val)
        
        # 극단적 이상치 클리핑 (선택사항)
        if self.clip_values:
            normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def visualize_normalization(self, subject_id: str,
                                original_ecg: np.ndarray,
                                original_ppg: np.ndarray,
                                normalized_ecg: np.ndarray,
                                normalized_ppg: np.ndarray,
                                time: np.ndarray,
                                window_seconds: float = 10.0) -> None:
        """
        정규화 전후 비교 시각화 (처음 10초)
        
        Args:
            subject_id: 피험자 ID
            original_ecg: 원본 ECG
            original_ppg: 원본 PPG
            normalized_ecg: 정규화된 ECG
            normalized_ppg: 정규화된 PPG
            time: 시간 배열
            window_seconds: 표시할 시간 윈도우 (초)
        """
        if not self.save_plots:
            return
        
        # 처음 window_seconds만 표시
        mask = time <= window_seconds
        
        # 플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{subject_id} - Min-Max Normalization: Before vs After', 
                     fontsize=16, fontweight='bold')
        
        # Before - ECG
        axes[0, 0].plot(time[mask], original_ecg[mask], 'b-', linewidth=0.8)
        axes[0, 0].set_title('Original ECG', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('ECG Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, window_seconds)
        
        # After - ECG
        axes[1, 0].plot(time[mask], normalized_ecg[mask], 'r-', linewidth=0.8)
        axes[1, 0].set_title('Normalized ECG [0, 1]', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Normalized ECG')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, window_seconds)
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        # Before - PPG
        axes[0, 1].plot(time[mask], original_ppg[mask], 'g-', linewidth=0.8)
        axes[0, 1].set_title('Original PPG', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('PPG Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, window_seconds)
        
        # After - PPG
        axes[1, 1].plot(time[mask], normalized_ppg[mask], 'm-', linewidth=0.8)
        axes[1, 1].set_title('Normalized PPG [0, 1]', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Normalized PPG')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, window_seconds)
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        plot_filename = self.plots_dir / f"{subject_id}_minmax_normalization.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 시각화 저장: {plot_filename.name}")
    
    def process_file(self, csv_path: Path) -> bool:
        """
        단일 CSV 파일 정규화
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            성공 여부
        """
        subject_id = csv_path.stem
        
        try:
            # 데이터 로드
            df = pd.read_csv(csv_path)
            
            print(f"\n[{subject_id}] Min-Max 정규화 시작...")
            print(f"  ✓ 데이터 로드: {len(df)} 샘플")
            
            # 칼럼 확인
            required_columns = ['time', 'ecg', 'ppg']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️  {subject_id}: 필수 칼럼 누락 (time, ecg, ppg)")
                return False
            
            # 원본 데이터 복사
            original_ecg = df['ecg'].values.copy()
            original_ppg = df['ppg'].values.copy()
            time = df['time'].values
            
            # ECG Min-Max 정규화
            print(f"  → ECG 정규화 중...")
            normalized_ecg = self.minmax_normalize(original_ecg)
            
            # PPG Min-Max 정규화
            print(f"  → PPG 정규화 중...")
            normalized_ppg = self.minmax_normalize(original_ppg)
            
            print(f"  ✓ Min-Max 정규화 완료")
            
            # 통계 출력
            print(f"  📊 ECG 통계:")
            print(f"     원본    - Min: {original_ecg.min():.4f}, Max: {original_ecg.max():.4f}, "
                  f"Mean: {original_ecg.mean():.4f}, Std: {original_ecg.std():.4f}")
            print(f"     정규화  - Min: {normalized_ecg.min():.4f}, Max: {normalized_ecg.max():.4f}, "
                  f"Mean: {normalized_ecg.mean():.4f}, Std: {normalized_ecg.std():.4f}")
            
            print(f"  📊 PPG 통계:")
            print(f"     원본    - Min: {original_ppg.min():.4f}, Max: {original_ppg.max():.4f}, "
                  f"Mean: {original_ppg.mean():.4f}, Std: {original_ppg.std():.4f}")
            print(f"     정규화  - Min: {normalized_ppg.min():.4f}, Max: {normalized_ppg.max():.4f}, "
                  f"Mean: {normalized_ppg.mean():.4f}, Std: {normalized_ppg.std():.4f}")
            
            # 시각화
            if self.save_plots:
                self.visualize_normalization(
                    subject_id,
                    original_ecg,
                    original_ppg,
                    normalized_ecg,
                    normalized_ppg,
                    time
                )
            
            # 정규화된 데이터프레임 생성
            df_normalized = pd.DataFrame({
                'time': time,
                'ecg': normalized_ecg,
                'ppg': normalized_ppg
            })
            
            # 저장
            output_path = self.output_dir / f"{subject_id}.csv"
            df_normalized.to_csv(output_path, index=False)
            
            print(f"  💾 저장 완료: {output_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ {subject_id}: 정규화 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_files(self) -> None:
        """모든 CSV 파일 정규화"""
        # CSV 파일 목록 가져오기
        csv_files = sorted(list(self.input_dir.glob("*.csv")))
        
        if not csv_files:
            print(f"⚠️  {self.input_dir}에 CSV 파일이 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print(f"ECG-PPG Min-Max 정규화 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"정규화 방법: Min-Max [0, 1]")
        print(f"클리핑 활성화: {self.clip_values}")
        print(f"총 파일 수: {len(csv_files)}")
        print(f"{'='*60}")
        
        success_count = 0
        
        for csv_file in csv_files:
            if self.process_file(csv_file):
                success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"Min-Max 정규화 완료!")
        print(f"{'='*60}")
        print(f"성공: {success_count}/{len(csv_files)} 파일")
        print(f"출력 위치: {self.output_dir}")
        print(f"정규화 범위: [0.0, 1.0]")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\synchronized"  # 동기화된 데이터
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\normalized"  # 정규화된 데이터
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\data_set_csv\plots\normalization"  # 시각화
    
    # 정규화 설정
    SAVE_PLOTS = True  # True: 정규화 전후 비교 시각화 저장
    CLIP_VALUES = True  # True: 극단적 이상치를 0~1 범위로 클리핑
    
    # Min-Max 정규화 실행
    normalizer = ECGPPGNormalizer(
        INPUT_DIR,
        OUTPUT_DIR,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR,
        clip_values=CLIP_VALUES
    )
    normalizer.process_all_files()


if __name__ == "__main__":
    main()

    
    def process_file(self, csv_path: Path) -> bool:
        """
        단일 CSV 파일 정규화
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            성공 여부
        """
        subject_id = csv_path.stem
        
        try:
            # 데이터 로드
            df = pd.read_csv(csv_path)
            
            print(f"\n[{subject_id}] Min-Max 정규화 시작...")
            print(f"  ✓ 데이터 로드: {len(df)} 샘플")
            
            # 칼럼 확인
            required_columns = ['time', 'ecg', 'ppg']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️  {subject_id}: 필수 칼럼 누락 (time, ecg, ppg)")
                return False
            
            # 원본 데이터 복사
            original_ecg = df['ecg'].values.copy()
            original_ppg = df['ppg'].values.copy()
            time = df['time'].values
            
            # ECG Min-Max 정규화
            print(f"  → ECG 정규화 중...")
            normalized_ecg = self.minmax_normalize(original_ecg)
            
            # PPG Min-Max 정규화
            print(f"  → PPG 정규화 중...")
            normalized_ppg = self.minmax_normalize(original_ppg)
            
            print(f"  ✓ Min-Max 정규화 완료")
            
            # 통계 출력
            print(f"  📊 ECG 통계:")
            print(f"     원본    - Min: {original_ecg.min():.4f}, Max: {original_ecg.max():.4f}, "
                  f"Mean: {original_ecg.mean():.4f}, Std: {original_ecg.std():.4f}")
            print(f"     정규화  - Min: {normalized_ecg.min():.4f}, Max: {normalized_ecg.max():.4f}, "
                  f"Mean: {normalized_ecg.mean():.4f}, Std: {normalized_ecg.std():.4f}")
            
            print(f"  📊 PPG 통계:")
            print(f"     원본    - Min: {original_ppg.min():.4f}, Max: {original_ppg.max():.4f}, "
                  f"Mean: {original_ppg.mean():.4f}, Std: {original_ppg.std():.4f}")
            print(f"     정규화  - Min: {normalized_ppg.min():.4f}, Max: {normalized_ppg.max():.4f}, "
                  f"Mean: {normalized_ppg.mean():.4f}, Std: {normalized_ppg.std():.4f}")
            
            # 시각화
            if self.save_plots:
                self.visualize_normalization(
                    subject_id,
                    original_ecg,
                    original_ppg,
                    normalized_ecg,
                    normalized_ppg,
                    time
                )
            
            # 정규화된 데이터프레임 생성
            df_normalized = pd.DataFrame({
                'time': time,
                'ecg': normalized_ecg,
                'ppg': normalized_ppg
            })
            
            # 저장
            output_path = self.output_dir / f"{subject_id}.csv"
            df_normalized.to_csv(output_path, index=False)
            
            print(f"  💾 저장 완료: {output_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ {subject_id}: 정규화 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_files(self) -> None:
        """모든 CSV 파일 정규화"""
        # CSV 파일 목록 가져오기
        csv_files = sorted(list(self.input_dir.glob("*.csv")))
        
        if not csv_files:
            print(f"⚠️  {self.input_dir}에 CSV 파일이 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print(f"ECG-PPG Min-Max 정규화 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"정규화 방법: Min-Max [0, 1]")
        print(f"클리핑 활성화: {self.clip_values}")
        print(f"총 파일 수: {len(csv_files)}")
        print(f"{'='*60}")
        
        success_count = 0
        
        for csv_file in csv_files:
            if self.process_file(csv_file):
                success_count += 1
        
        # 최종 결과 출력
        print(f"\n{'='*60}")
        print(f"Min-Max 정규화 완료!")
        print(f"{'='*60}")
        print(f"성공: {success_count}/{len(csv_files)} 파일")
        print(f"출력 위치: {self.output_dir}")
        print(f"정규화 범위: [0.0, 1.0]")
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\synchronized"  # 동기화된 데이터
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\normalized"  # 정규화된 데이터
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\normalized\plots"  # 시각화
    
    # 정규화 설정
    SAVE_PLOTS = True  # True: 정규화 전후 비교 시각화 저장
    CLIP_VALUES = True  # True: 극단적 이상치를 0~1 범위로 클리핑
    
    # Min-Max 정규화 실행
    normalizer = ECGPPGNormalizer(
        INPUT_DIR,
        OUTPUT_DIR,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR,
        clip_values=CLIP_VALUES
    )
    normalizer.process_all_files()


if __name__ == "__main__":
    main()

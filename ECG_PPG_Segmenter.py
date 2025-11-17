"""
ECG-PPG 데이터 슬라이딩 윈도우 분할 스크립트

정규화된 ECG와 PPG 데이터를 고정 길이 윈도우로 분할합니다.
- 윈도우 길이: 기본 10초 (변경 가능)
- 오버랩: 기본 5초 (변경 가능)
- 마지막 세그먼트 처리: 오버랩이 부족하면 가능한 만큼만 오버랩
- 파일명: 모든 피험자에 걸쳐 연속 번호 (seg_0000000.csv, seg_0000001.csv, ...)

입력: normalized/ 폴더의 Subject_XX.csv 파일들
출력: segmented/ 폴더에 분할된 파일들 저장 (연속 번호, 7자리)

예시:
    Subject_01.csv → seg_0000000.csv, seg_0000001.csv, seg_0000002.csv, ...
    Subject_02.csv → seg_0000020.csv, seg_0000021.csv, seg_0000022.csv, ...
    Subject_03.csv → seg_0000040.csv, seg_0000041.csv, seg_0000042.csv, ...
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ECGPPGSegmenter:
    """ECG-PPG 데이터 세그먼트 분할 클래스"""
    
    def __init__(self, input_dir: str, output_dir: str,
                 window_length: float = 10.0,
                 overlap: float = 5.0,
                 sampling_rate: float = 256.0,
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None):
        """
        Args:
            input_dir: 입력 디렉토리 (정규화된 데이터)
            output_dir: 출력 디렉토리
            window_length: 윈도우 길이 (초)
            overlap: 오버랩 길이 (초)
            sampling_rate: 샘플링 레이트 (Hz)
            save_plots: 세그먼트 시각화 저장 여부
            plots_dir: 시각화 저장 디렉토리
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_length = window_length  # 초
        self.overlap = overlap  # 초
        self.sampling_rate = sampling_rate  # Hz
        
        # 샘플 수로 변환
        self.window_samples = int(window_length * sampling_rate)
        self.overlap_samples = int(overlap * sampling_rate)
        self.step_samples = self.window_samples - self.overlap_samples
        
        self.save_plots = save_plots
        
        # 전역 세그먼트 카운터 (모든 파일에 걸쳐 연속 번호)
        self.global_segment_counter = 0
        
        # 시각화 저장 디렉토리
        if self.save_plots:
            if plots_dir is not None:
                self.plots_dir = Path(plots_dir)
            else:
                self.plots_dir = self.output_dir / "visualization"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"세그먼트 분할 설정")
        print(f"{'='*60}")
        print(f"윈도우 길이: {window_length}초 ({self.window_samples} 샘플)")
        print(f"오버랩: {overlap}초 ({self.overlap_samples} 샘플)")
        print(f"스텝 크기: {window_length - overlap}초 ({self.step_samples} 샘플)")
        print(f"샘플링 레이트: {sampling_rate} Hz")
        print(f"{'='*60}\n")
    
    def create_segments(self, time: np.ndarray, ecg: np.ndarray, ppg: np.ndarray) -> List[dict]:
        """
        슬라이딩 윈도우로 데이터 분할
        
        Args:
            time: 시간 배열
            ecg: ECG 신호
            ppg: PPG 신호
            
        Returns:
            세그먼트 리스트 (각 세그먼트는 dict)
        """
        segments = []
        total_samples = len(time)
        
        # 기본 슬라이딩 윈도우
        start_idx = 0
        local_segment_idx = 0  # 현재 파일 내 로컬 인덱스
        
        while start_idx < total_samples:
            end_idx = start_idx + self.window_samples
            
            # 윈도우가 데이터 범위를 벗어나는 경우
            if end_idx > total_samples:
                # 남은 샘플이 윈도우 길이보다 짧은 경우
                remaining_samples = total_samples - start_idx
                
                # 남은 샘플이 너무 적으면 (윈도우의 50% 미만) 건너뛰기
                if remaining_samples < self.window_samples * 0.5:
                    print(f"    ⚠️  마지막 {remaining_samples} 샘플 ({remaining_samples/self.sampling_rate:.2f}초)은 "
                          f"윈도우 길이의 50% 미만이므로 생략합니다.")
                    break
                
                # 마지막 세그먼트: 데이터 끝에서 역으로 윈도우 크기만큼 자르기
                start_idx = total_samples - self.window_samples
                end_idx = total_samples
                
                # 이전 세그먼트와의 실제 오버랩 계산
                if local_segment_idx > 0:
                    prev_end = segments[-1]['end_idx']
                    actual_overlap = prev_end - start_idx
                    actual_overlap_time = actual_overlap / self.sampling_rate
                    
                    print(f"    ℹ️  마지막 세그먼트: 오버랩 {actual_overlap_time:.2f}초 "
                          f"(원래: {self.overlap}초)")
                
                # 마지막 세그먼트 추가
                segment = {
                    'global_segment_idx': self.global_segment_counter,  # 전역 인덱스
                    'local_segment_idx': local_segment_idx,  # 로컬 인덱스
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': time[start_idx],
                    'end_time': time[end_idx - 1],
                    'time': time[start_idx:end_idx],
                    'ecg': ecg[start_idx:end_idx],
                    'ppg': ppg[start_idx:end_idx],
                    'num_samples': end_idx - start_idx
                }
                segments.append(segment)
                self.global_segment_counter += 1
                break
            
            # 일반 세그먼트 추가
            segment = {
                'global_segment_idx': self.global_segment_counter,  # 전역 인덱스
                'local_segment_idx': local_segment_idx,  # 로컬 인덱스
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': time[start_idx],
                'end_time': time[end_idx - 1],
                'time': time[start_idx:end_idx],
                'ecg': ecg[start_idx:end_idx],
                'ppg': ppg[start_idx:end_idx],
                'num_samples': end_idx - start_idx
            }
            segments.append(segment)
            
            # 카운터 증가
            self.global_segment_counter += 1
            local_segment_idx += 1
            
            # 다음 시작 위치
            start_idx += self.step_samples
        
        return segments
    
    def visualize_segments(self, subject_id: str, segments: List[dict], 
                          max_segments_to_plot: int = 5) -> None:
        """
        세그먼트 분할 시각화 (처음 몇 개만)
        
        Args:
            subject_id: 피험자 ID
            segments: 세그먼트 리스트
            max_segments_to_plot: 시각화할 최대 세그먼트 수
        """
        if not self.save_plots or not segments:
            return
        
        num_to_plot = min(len(segments), max_segments_to_plot)
        
        # 플롯 생성
        fig, axes = plt.subplots(num_to_plot, 2, figsize=(16, 3*num_to_plot))
        if num_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{subject_id} - First {num_to_plot} Segments (Window: {self.window_length}s, Overlap: {self.overlap}s)', 
                     fontsize=14, fontweight='bold')
        
        for i in range(num_to_plot):
            segment = segments[i]
            time = segment['time']
            ecg = segment['ecg']
            ppg = segment['ppg']
            global_idx = segment['global_segment_idx']
            
            # ECG
            axes[i, 0].plot(time, ecg, 'b-', linewidth=0.8)
            axes[i, 0].set_title(f'Segment {global_idx} - ECG ({segment["start_time"]:.2f}s - {segment["end_time"]:.2f}s)', 
                                fontsize=10, fontweight='bold')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('ECG')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_ylim(-0.05, 1.05)
            
            # PPG
            axes[i, 1].plot(time, ppg, 'r-', linewidth=0.8)
            axes[i, 1].set_title(f'Segment {global_idx} - PPG ({segment["start_time"]:.2f}s - {segment["end_time"]:.2f}s)', 
                                fontsize=10, fontweight='bold')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('PPG')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # 저장
        plot_filename = self.plots_dir / f"{subject_id}_segments.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 시각화 저장: {plot_filename.name}")
    
    def save_segments(self, subject_id: str, segments: List[dict]) -> None:
        """
        세그먼트를 개별 CSV 파일로 저장
        
        Args:
            subject_id: 피험자 ID
            segments: 세그먼트 리스트
        """
        for segment in segments:
            # 파일명: seg_0000000.csv (전역 인덱스 사용, 7자리)
            filename = f"seg_{segment['global_segment_idx']:07d}.csv"
            output_path = self.output_dir / filename
            
            # 데이터프레임 생성
            df_segment = pd.DataFrame({
                'time': segment['time'],
                'ecg': segment['ecg'],
                'ppg': segment['ppg']
            })
            
            # 시간을 세그먼트 시작점 기준으로 재설정 (0부터 시작)
            df_segment['time'] = df_segment['time'] - df_segment['time'].iloc[0]
            
            # 저장
            df_segment.to_csv(output_path, index=False)
        
        print(f"  💾 {len(segments)}개 세그먼트 저장 완료 (seg_{segments[0]['global_segment_idx']:07d} ~ seg_{segments[-1]['global_segment_idx']:07d})")
    
    def process_file(self, csv_path: Path) -> bool:
        """
        단일 CSV 파일 세그먼트 분할
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            성공 여부
        """
        subject_id = csv_path.stem
        
        try:
            # 데이터 로드
            df = pd.read_csv(csv_path)
            
            print(f"\n[{subject_id}] 세그먼트 분할 시작...")
            print(f"  ✓ 데이터 로드: {len(df)} 샘플 ({len(df)/self.sampling_rate:.2f}초)")
            
            # 칼럼 확인
            required_columns = ['time', 'ecg', 'ppg']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️  {subject_id}: 필수 칼럼 누락 (time, ecg, ppg)")
                return False
            
            # 데이터 추출
            time = df['time'].values
            ecg = df['ecg'].values
            ppg = df['ppg'].values
            
            # 세그먼트 생성
            segments = self.create_segments(time, ecg, ppg)
            
            if not segments:
                print(f"⚠️  {subject_id}: 생성된 세그먼트가 없습니다.")
                return False
            
            print(f"  ✓ 세그먼트 생성: {len(segments)}개")
            print(f"    - 각 세그먼트: {self.window_length}초 ({self.window_samples} 샘플)")
            print(f"    - 오버랩: {self.overlap}초 ({self.overlap_samples} 샘플)")
            
            # 시각화
            if self.save_plots:
                self.visualize_segments(subject_id, segments)
            
            # 저장
            self.save_segments(subject_id, segments)
            
            return True
            
        except Exception as e:
            print(f"❌ {subject_id}: 세그먼트 분할 중 오류 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_files(self) -> None:
        """모든 CSV 파일 세그먼트 분할"""
        # CSV 파일 목록 가져오기
        csv_files = sorted(list(self.input_dir.glob("*.csv")))
        
        if not csv_files:
            print(f"⚠️  {self.input_dir}에 CSV 파일이 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print(f"ECG-PPG 세그먼트 분할 시작")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 파일 수: {len(csv_files)}")
        print(f"{'='*60}")
        
        success_count = 0
        
        for csv_file in csv_files:
            if self.process_file(csv_file):
                success_count += 1
        
        # 최종 결과 출력
        total_segments = self.global_segment_counter  # 전역 카운터가 총 세그먼트 수
        
        print(f"\n{'='*60}")
        print(f"세그먼트 분할 완료!")
        print(f"{'='*60}")
        print(f"처리된 파일: {success_count}/{len(csv_files)}")
        print(f"생성된 세그먼트: {total_segments}개 (seg_0000000 ~ seg_{total_segments-1:07d})")
        print(f"출력 위치: {self.output_dir}")
        print(f"{'='*60}\n")

def main():
    """메인 실행 함수"""
    # 경로 설정
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\normalized"  # 정규화된 데이터
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\segmented"  # 세그먼트 데이터
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\segmented\plots"  # 시각화
    
    # 세그먼트 설정
    WINDOW_LENGTH = 10.0  # 윈도우 길이 (초)
    OVERLAP = 5.0  # 오버랩 (초)
    SAMPLING_RATE = 256.0  # 샘플링 레이트 (Hz)
    SAVE_PLOTS = True  # True: 세그먼트 시각화 저장
    
    # 세그먼트 분할 실행
    segmenter = ECGPPGSegmenter(
        INPUT_DIR,
        OUTPUT_DIR,
        window_length=WINDOW_LENGTH,
        overlap=OVERLAP,
        sampling_rate=SAMPLING_RATE,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR
    )
    segmenter.process_all_files()


if __name__ == "__main__":
    main()
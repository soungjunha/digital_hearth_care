"""
ECG-PPG 데이터 슬라이딩 윈도우 분할 스크립트 (Edge Padding 적용)

정규화된 ECG와 PPG 데이터를 고정 길이 윈도우로 분할합니다.
- 윈도우 길이: 기본 30초 (변경 가능)
- 오버랩: 기본 15초 (변경 가능)
- 마지막 세그먼트 처리: Edge padding (마지막 값 반복) - 성능 저하 최소화
- 파일명: 모든 피험자에 걸쳐 연속 번호 (seg_0000000.csv, seg_0000001.csv, ...)

입력: normalized/ 폴더의 Subject_XX.csv 파일들
출력: segmented/ 폴더에 분할된 파일들 저장 (연속 번호, 7자리)

개선사항:
    - Edge padding: 신호의 마지막 값을 반복하여 자연스러운 연속성 유지
    - 0.1초 미만 부족 시 padding 적용 (모델 성능 저하 최소화)
    - 0.1초 이상 부족 시 역방향 윈도우 생성
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
    """ECG-PPG 데이터 세그먼트 분할 클래스 (Edge Padding 지원)"""
    
    def __init__(self, input_dir: str, output_dir: str,
                 window_length: float = 30.0,
                 overlap: float = 15.0,
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
        
        # 통계 정보
        self.stats = {
            'total_segments': 0,
            'padded_segments': 0,
            'reduced_overlap_segments': 0,
            'discarded_segments': 0,
            'max_padding_samples': 0,
            'total_padding_samples': 0
        }
        
        # 시각화 저장 디렉토리
        if self.save_plots:
            if plots_dir is not None:
                self.plots_dir = Path(plots_dir)
            else:
                self.plots_dir = self.output_dir / "visualization"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"세그먼트 분할 설정 (적응형 오버랩 전략)")
        print(f"{'='*70}")
        print(f"윈도우 길이: {window_length}초 ({self.window_samples} 샘플)")
        print(f"오버랩: {overlap}초 ({self.overlap_samples} 샘플)")
        print(f"스텝 크기: {window_length - overlap}초 ({self.step_samples} 샘플)")
        print(f"샘플링 레이트: {sampling_rate} Hz")
        print(f"")
        print(f"처리 전략:")
        print(f"  1. 전체 데이터 < 윈도우 - 1초: 생략")
        print(f"  2. 전체 데이터 >= 윈도우 - 1초: Edge Padding (1초 미만)")
        print(f"  3. 남은 데이터 >= 오버랩의 30% ({overlap*0.3:.1f}초): 오버랩 감소")
        print(f"  4. 남은 데이터 < 오버랩의 30% ({overlap*0.3:.1f}초): 생략")
        print(f"{'='*70}\n")
    
    def create_segments(self, time: np.ndarray, ecg: np.ndarray, ppg: np.ndarray) -> List[dict]:
        """
        슬라이딩 윈도우로 데이터 분할 (적응형 오버랩 전략)
        
        마지막 세그먼트 처리 전략:
        1. 전체 데이터가 윈도우보다 짧음: Edge Padding (최소 비율 이상인 경우만)
        2. 남은 데이터가 오버랩의 30%보다 길면: 오버랩 감소
        3. 남은 데이터가 오버랩의 30%보다 짧으면: 생략
        
        Args:
            time: 시간 배열
            ecg: ECG 신호
            ppg: PPG 신호
            
        Returns:
            세그먼트 리스트 (각 세그먼트는 dict)
        """
        segments = []
        total_samples = len(time)
        
        # 전체 데이터가 윈도우보다 짧은 경우
        if total_samples < self.window_samples:
            total_seconds = total_samples / self.sampling_rate
            shortage_seconds = (self.window_samples - total_samples) / self.sampling_rate
            
            # 1초 이상 짧으면 생략
            if shortage_seconds >= 1.0:
                print(f"    ⚠️  전체 데이터 {total_samples}샘플 ({total_seconds:.2f}초)이 "
                      f"윈도우 길이보다 {shortage_seconds:.2f}초 짧아서 생략합니다.")
                self.stats['discarded_segments'] += 1
                return segments
            
            # 1초 미만 부족 → Edge Padding
            shortage = self.window_samples - total_samples
            
            print(f"    ℹ️  전체 데이터 {total_seconds:.2f}초 → {shortage_seconds:.3f}초 Edge Padding 적용")
            
            # Edge Padding
            time_interval = 1.0 / self.sampling_rate
            time_padding = np.arange(1, shortage + 1) * time_interval + time[-1]
            time_padded = np.concatenate([time, time_padding])
            
            ecg_padded = np.concatenate([ecg, np.full(shortage, ecg[-1])])
            ppg_padded = np.concatenate([ppg, np.full(shortage, ppg[-1])])
            
            self.stats['padded_segments'] += 1
            self.stats['total_padding_samples'] += shortage
            if shortage > self.stats['max_padding_samples']:
                self.stats['max_padding_samples'] = shortage
            
            segment = {
                'global_segment_idx': self.global_segment_counter,
                'local_segment_idx': 0,
                'start_idx': 0,
                'end_idx': total_samples,
                'start_time': time[0],
                'end_time': time[-1],
                'time': time_padded,
                'ecg': ecg_padded,
                'ppg': ppg_padded,
                'num_samples': self.window_samples,
                'padded': True,
                'padding_type': 'edge',
                'padded_samples': shortage,
                'padded_seconds': shortage_seconds,
                'reduced_overlap': False
            }
            segments.append(segment)
            self.global_segment_counter += 1
            return segments
        
        # 일반 슬라이딩 윈도우
        start_idx = 0
        local_segment_idx = 0
        
        # 오버랩의 30% 계산 (최소 허용 남은 데이터)
        min_remaining_samples = int(self.overlap_samples * 0.3)
        min_remaining_seconds = min_remaining_samples / self.sampling_rate
        
        while start_idx < total_samples:
            end_idx = start_idx + self.window_samples
            
            # 윈도우가 데이터 범위를 벗어나는 경우
            if end_idx > total_samples:
                remaining_samples = total_samples - start_idx
                remaining_seconds = remaining_samples / self.sampling_rate
                
                # === 단순화된 마지막 세그먼트 처리 ===
                
                # 남은 데이터가 오버랩의 30%보다 짧으면 생략
                if remaining_samples < min_remaining_samples:
                    print(f"    ⚠️  마지막 {remaining_samples}샘플 ({remaining_seconds:.2f}초)은 "
                          f"오버랩의 30% ({min_remaining_seconds:.2f}초) 미만이므로 생략합니다.")
                    self.stats['discarded_segments'] += 1
                    break
                
                # 남은 데이터가 오버랩의 30% 이상이면 오버랩 감소
                # 새로운 시작점 = 끝 - 윈도우 길이
                new_start_idx = total_samples - self.window_samples
                
                # 이전 세그먼트와의 실제 오버랩 계산
                if local_segment_idx > 0:
                    prev_end = segments[-1]['end_idx']
                    actual_overlap = prev_end - new_start_idx
                    actual_overlap_time = actual_overlap / self.sampling_rate
                    
                    print(f"    ℹ️  마지막 세그먼트: {remaining_seconds:.2f}초 남음 → "
                          f"오버랩 감소 전략 (오버랩 {actual_overlap_time:.2f}초, "
                          f"원래 {self.overlap}초)")
                else:
                    # 첫 번째 세그먼트인데 윈도우를 초과하는 경우
                    actual_overlap = 0
                    actual_overlap_time = 0.0
                    print(f"    ℹ️  첫 세그먼트: {remaining_seconds:.2f}초 남음 → "
                          f"역방향 윈도우 생성")
                
                self.stats['reduced_overlap_segments'] += 1
                
                segment = {
                    'global_segment_idx': self.global_segment_counter,
                    'local_segment_idx': local_segment_idx,
                    'start_idx': new_start_idx,
                    'end_idx': total_samples,
                    'start_time': time[new_start_idx],
                    'end_time': time[-1],
                    'time': time[new_start_idx:],
                    'ecg': ecg[new_start_idx:],
                    'ppg': ppg[new_start_idx:],
                    'num_samples': self.window_samples,
                    'padded': False,
                    'reduced_overlap': True,
                    'actual_overlap_samples': actual_overlap if local_segment_idx > 0 else 0,
                    'actual_overlap_seconds': actual_overlap_time
                }
                segments.append(segment)
                self.global_segment_counter += 1
                break
            
            # 일반 세그먼트 추가
            segment = {
                'global_segment_idx': self.global_segment_counter,
                'local_segment_idx': local_segment_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': time[start_idx],
                'end_time': time[end_idx - 1],
                'time': time[start_idx:end_idx],
                'ecg': ecg[start_idx:end_idx],
                'ppg': ppg[start_idx:end_idx],
                'num_samples': end_idx - start_idx,
                'padded': False,
                'reduced_overlap': False
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
        세그먼트 분할 시각화 (처음 몇 개 + padding된 세그먼트)
        
        Args:
            subject_id: 피험자 ID
            segments: 세그먼트 리스트
            max_segments_to_plot: 시각화할 최대 일반 세그먼트 수
        """
        if not self.save_plots or not segments:
            return
        
        # 일반 세그먼트와 padding된 세그먼트 분리
        normal_segments = [s for s in segments if not s.get('padded', False)]
        padded_segments = [s for s in segments if s.get('padded', False)]
        
        # 시각화할 세그먼트 선택
        segments_to_plot = normal_segments[:max_segments_to_plot] + padded_segments
        num_to_plot = len(segments_to_plot)
        
        if num_to_plot == 0:
            return
        
        # 플롯 생성
        fig, axes = plt.subplots(num_to_plot, 2, figsize=(16, 3*num_to_plot))
        if num_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{subject_id} - Segments (Window: {self.window_length}s, Overlap: {self.overlap}s)', 
                     fontsize=14, fontweight='bold')
        
        for i, segment in enumerate(segments_to_plot):
            time = segment['time']
            ecg = segment['ecg']
            ppg = segment['ppg']
            global_idx = segment['global_segment_idx']
            is_padded = segment.get('padded', False)
            
            # 제목 생성
            if is_padded:
                padding_info = f" [PADDED: {segment['padded_samples']} samples]"
                title_color = 'red'
            else:
                padding_info = ""
                title_color = 'black'
            
            # ECG
            axes[i, 0].plot(time, ecg, 'b-', linewidth=0.8)
            if is_padded:
                # Padding 영역 표시
                padding_start_idx = len(time) - segment['padded_samples']
                axes[i, 0].axvspan(time[padding_start_idx], time[-1], 
                                   alpha=0.2, color='red', label='Padded region')
            
            title = f'Segment {global_idx} - ECG ({segment["start_time"]:.2f}s - {segment["end_time"]:.2f}s){padding_info}'
            axes[i, 0].set_title(title, fontsize=10, fontweight='bold', color=title_color)
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('ECG')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_ylim(-0.05, 1.05)
            if is_padded:
                axes[i, 0].legend(loc='upper right', fontsize=8)
            
            # PPG
            axes[i, 1].plot(time, ppg, 'r-', linewidth=0.8)
            if is_padded:
                # Padding 영역 표시
                padding_start_idx = len(time) - segment['padded_samples']
                axes[i, 1].axvspan(time[padding_start_idx], time[-1], 
                                   alpha=0.2, color='red', label='Padded region')
            
            title = f'Segment {global_idx} - PPG ({segment["start_time"]:.2f}s - {segment["end_time"]:.2f}s){padding_info}'
            axes[i, 1].set_title(title, fontsize=10, fontweight='bold', color=title_color)
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('PPG')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim(-0.05, 1.05)
            if is_padded:
                axes[i, 1].legend(loc='upper right', fontsize=8)
        
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
            
            # 저장 (메타데이터 없이)
            df_segment.to_csv(output_path, index=False)
        
        # 통계 정보 출력
        padded_count = sum(1 for s in segments if s.get('padded', False))
        reduced_overlap_count = sum(1 for s in segments if s.get('reduced_overlap', False))
        
        if padded_count > 0 or reduced_overlap_count > 0:
            print(f"  💾 {len(segments)}개 세그먼트 저장 완료 (seg_{segments[0]['global_segment_idx']:07d} ~ "
                  f"seg_{segments[-1]['global_segment_idx']:07d}) - Padded: {padded_count}개, Reduced Overlap: {reduced_overlap_count}개")
        else:
            print(f"  💾 {len(segments)}개 세그먼트 저장 완료 (seg_{segments[0]['global_segment_idx']:07d} ~ "
                  f"seg_{segments[-1]['global_segment_idx']:07d})")
    
    def process_file(self, csv_path: Path) -> bool:
        """
        단일 CSV 파일 세그먼트 분할
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            성공 여부
        """
        subject_id = csv_path.stem
        
        # 파일 존재 여부 확인
        if not csv_path.exists():
            print(f"⚠️  {subject_id}: 파일이 존재하지 않습니다 - {csv_path}")
            return False
        
        try:
            # 데이터 로드
            df = pd.read_csv(csv_path)
            
            print(f"\n[{subject_id}] 세그먼트 분할 시작...")
            print(f"  ✓ 데이터 로드: {len(df)} 샘플 ({len(df)/self.sampling_rate:.3f}초)")
            
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
            
            # 세그먼트 통계
            padded_segments = [s for s in segments if s.get('padded', False)]
            
            print(f"  ✓ 세그먼트 생성: {len(segments)}개 (Padded: {len(padded_segments)}개)")
            print(f"    - 각 세그먼트: {self.window_length}초 ({self.window_samples} 샘플)")
            print(f"    - 오버랩: {self.overlap}초 ({self.overlap_samples} 샘플)")
            
            if padded_segments:
                for seg in padded_segments:
                    print(f"    - Padded 세그먼트 {seg['global_segment_idx']}: "
                          f"{seg['padded_samples']}샘플 ({seg['padded_seconds']:.4f}초) padding")
            
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
        # CSV 파일 목록 가져오기 (실제 존재하는 파일만)
        csv_files = sorted([f for f in self.input_dir.glob("*.csv") if f.exists() and f.is_file()])
        
        if not csv_files:
            print(f"⚠️  {self.input_dir}에 CSV 파일이 없습니다.")
            return
        
        print(f"\n{'='*70}")
        print(f"ECG-PPG 세그먼트 분할 시작 (Edge Padding)")
        print(f"입력 디렉토리: {self.input_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"총 파일 수: {len(csv_files)}")
        
        # 디버깅: 파일 목록 일부 출력
        print(f"\n발견된 파일 예시 (처음 5개, 마지막 5개):")
        for f in csv_files[:5]:
            print(f"  - {f.name}")
        if len(csv_files) > 10:
            print(f"  ...")
            for f in csv_files[-5:]:
                print(f"  - {f.name}")
        
        print(f"{'='*70}")
        
        success_count = 0
        failed_files = []
        
        for csv_file in csv_files:
            if self.process_file(csv_file):
                success_count += 1
            else:
                failed_files.append(csv_file.name)
        
        # 최종 결과 출력
        total_segments = self.global_segment_counter
        
        print(f"\n{'='*70}")
        print(f"세그먼트 분할 완료!")
        print(f"{'='*70}")
        print(f"처리된 파일: {success_count}/{len(csv_files)}")
        if failed_files:
            print(f"실패한 파일: {len(failed_files)}개")
            for failed_file in failed_files[:10]:  # 최대 10개만 표시
                print(f"  - {failed_file}")
            if len(failed_files) > 10:
                print(f"  ... 외 {len(failed_files) - 10}개")
        
        if total_segments > 0:
            print(f"생성된 세그먼트: {total_segments}개 (seg_0000000 ~ seg_{total_segments-1:07d})")
            print(f"")
            print(f"📊 세그먼트 처리 통계:")
            print(f"  - 정상 세그먼트: {total_segments - self.stats['padded_segments'] - self.stats['reduced_overlap_segments']}개")
            print(f"  - Padded 세그먼트: {self.stats['padded_segments']}개 "
                  f"({self.stats['padded_segments']/total_segments*100:.2f}%)")
            print(f"  - Reduced Overlap 세그먼트: {self.stats['reduced_overlap_segments']}개 "
                  f"({self.stats['reduced_overlap_segments']/total_segments*100:.2f}%)")
            print(f"  - 생략된 세그먼트: {self.stats['discarded_segments']}개")
            
            if self.stats['padded_segments'] > 0:
                print(f"")
                print(f"📊 Padding 상세 통계:")
                print(f"  - 총 Padding 샘플: {self.stats['total_padding_samples']}개")
                print(f"  - 최대 Padding: {self.stats['max_padding_samples']}샘플 "
                      f"({self.stats['max_padding_samples']/self.sampling_rate:.3f}초)")
                print(f"  - 평균 Padding: {self.stats['total_padding_samples']/self.stats['padded_segments']:.1f}샘플 "
                      f"({self.stats['total_padding_samples']/self.stats['padded_segments']/self.sampling_rate:.3f}초)")
        else:
            print(f"생성된 세그먼트: 0개")
        
        print(f"")
        print(f"출력 위치: {self.output_dir}")
        print(f"{'='*70}\n")


def main():
    """메인 실행 함수"""
    # 경로 설정
    INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\normalized"  # 정규화된 데이터
    OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\segmented_30s"  # 30초 세그먼트 데이터
    PLOTS_DIR = r"F:\coding자료\coding\digital_hearth_care\segmented_30s\plots"  # 시각화
    
    # 세그먼트 설정 (30초 윈도우)
    WINDOW_LENGTH = 30.0  # 윈도우 길이 (초)
    OVERLAP = 15.0  # 오버랩 (초) - 50% overlap
    SAMPLING_RATE = 256.0  # 샘플링 레이트 (Hz)
    SAVE_PLOTS = True  # True: 세그먼트 시각화 저장
    
    # === 적응형 오버랩 전략 (30초 윈도우, 15초 오버랩 기준) ===
    # 
    # 오버랩의 30% = 4.5초
    # 
    # 예시 1: 29.5초 전체 데이터
    #   → Edge Padding (0.5초 padding, 1초 미만 부족)
    # 
    # 예시 2: 28.5초 전체 데이터
    #   → 생략 (1.5초 부족, 1초 이상)
    # 
    # 예시 3: 59초 데이터 (0-30, 15-45 생성 후 29초 남음)
    #   → 0-30초, 15-45초 생성
    #   → 마지막: 29-59초 (29초 > 4.5초 → 오버랩 감소, 14초 오버랩)
    # 
    # 예시 4: 54초 데이터 (0-30, 15-45 생성 후 24초 남음)
    #   → 0-30초, 15-45초 생성
    #   → 마지막: 24-54초 (24초 > 4.5초 → 오버랩 감소, 21초 오버랩)
    # 
    # 예시 5: 48초 데이터 (0-30, 15-45 생성 후 18초 남음)
    #   → 0-30초, 15-45초 생성
    #   → 마지막: 18-48초 (18초 > 4.5초 → 오버랩 감소, 27초 오버랩)
    # 
    # 예시 6: 47초 데이터 (0-30, 15-45 생성 후 17초 남음)
    #   → 0-30초, 15-45초 생성
    #   → 마지막: 17-47초 (17초 > 4.5초 → 오버랩 감소, 28초 오버랩)
    # 
    # 예시 7: 48초 데이터 (0-30, 15-45 생성 후 3초 남음)
    #   → 0-30초, 15-45초 생성
    #   → 마지막: 3초 < 4.5초 → 생략
    
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
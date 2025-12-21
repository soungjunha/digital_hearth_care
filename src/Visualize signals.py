import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def visualize_signal(csv_file, output_dir=None, time_duration=10):
    """
    CSV 파일의 앞/뒤 지정된 시간 구간을 시각화
    
    Args:
        csv_file: CSV 파일 경로
        output_dir: 이미지 저장 디렉토리 (None이면 화면에만 표시)
        time_duration: 시각화할 시간 구간 (초), 기본값 10초
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 시간 컬럼 찾기 (Time_sec, Time, time 등)
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        print(f"⚠️  시간 컬럼을 찾을 수 없습니다: {csv_file}")
        return
    
    # 신호 컬럼들 (시간 컬럼 제외)
    signal_cols = [col for col in df.columns if col != time_col]
    
    if not signal_cols:
        print(f"⚠️  신호 컬럼을 찾을 수 없습니다: {csv_file}")
        return
    
    time = df[time_col].values
    total_duration = time[-1] - time[0]
    
    print(f"\n파일: {Path(csv_file).name}")
    print(f"  총 샘플 수: {len(df)}")
    print(f"  총 시간: {total_duration:.2f} 초")
    print(f"  샘플링 주파수: {len(df) / total_duration:.2f} Hz")
    print(f"  신호 컬럼: {', '.join(signal_cols)}")
    print(f"  시각화 구간: 앞/뒤 {time_duration}초")
    
    # 지정된 시간만큼 데이터 추출
    first_mask = time <= (time[0] + time_duration)
    last_mask = time >= (time[-1] - time_duration)
    
    first_segment = df[first_mask]
    last_segment = df[last_mask]
    
    print(f"  앞 {time_duration}초 샘플 수: {len(first_segment)}")
    print(f"  뒤 {time_duration}초 샘플 수: {len(last_segment)}")
    
    # 각 신호별로 시각화
    for signal_col in signal_cols:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'{Path(csv_file).stem} - {signal_col}', fontsize=16, fontweight='bold')
        
        # 앞 구간 플롯
        axes[0].plot(first_segment[time_col], first_segment[signal_col], 'b-', linewidth=0.5)
        axes[0].set_xlabel('Time (sec)', fontsize=12)
        axes[0].set_ylabel(signal_col, fontsize=12)
        axes[0].set_title(f'First {time_duration} seconds', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(time[0], time[0] + time_duration)
        
        # 통계 정보 표시
        mean_first = first_segment[signal_col].mean()
        std_first = first_segment[signal_col].std()
        axes[0].text(0.02, 0.98, f'Mean: {mean_first:.2f}\nStd: {std_first:.2f}', 
                    transform=axes[0].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 뒤 구간 플롯
        axes[1].plot(last_segment[time_col], last_segment[signal_col], 'r-', linewidth=0.5)
        axes[1].set_xlabel('Time (sec)', fontsize=12)
        axes[1].set_ylabel(signal_col, fontsize=12)
        axes[1].set_title(f'Last {time_duration} seconds', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(time[-1] - time_duration, time[-1])
        
        # 통계 정보 표시
        mean_last = last_segment[signal_col].mean()
        std_last = last_segment[signal_col].std()
        axes[1].text(0.02, 0.98, f'Mean: {mean_last:.2f}\nStd: {std_last:.2f}', 
                    transform=axes[1].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{Path(csv_file).stem}_{signal_col}_{time_duration}sec.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ 저장: {output_file.name}")
            plt.close()
        else:
            # 윈도우 창으로 표시
            plt.show()


def visualize_directory(input_dir, output_dir=None, time_duration=10):
    """
    디렉토리 내의 모든 CSV 파일을 시각화
    
    Args:
        input_dir: CSV 파일들이 있는 디렉토리
        output_dir: 이미지 저장 디렉토리
        time_duration: 시각화할 시간 구간 (초)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"⚠️  디렉토리가 존재하지 않습니다: {input_path}")
        return
    
    # CSV 파일 찾기
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  CSV 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"=" * 80)
    print(f"발견된 CSV 파일 수: {len(csv_files)}")
    print(f"=" * 80)
    
    for csv_file in sorted(csv_files):
        try:
            visualize_signal(csv_file, output_dir, time_duration)
        except Exception as e:
            print(f"  ✗ 오류 발생: {str(e)}")
            continue
    
    print(f"\n" + "=" * 80)
    print("시각화 완료!")
    if output_dir:
        print(f"저장 위치: {output_dir}")
    print("=" * 80)


def visualize_all_participants(participants_dir, output_base_dir, time_duration=10):
    """
    모든 참가자 폴더의 CSV 파일들을 시각화
    
    Args:
        participants_dir: 참가자 폴더들이 있는 디렉토리
        output_base_dir: 이미지 저장 기본 디렉토리
        time_duration: 시각화할 시간 구간 (초)
    """
    participants_path = Path(participants_dir)
    
    if not participants_path.exists():
        print(f"⚠️  디렉토리가 존재하지 않습니다: {participants_path}")
        return
    
    # 참가자 폴더 찾기
    participant_folders = [f for f in participants_path.iterdir() if f.is_dir()]
    
    if not participant_folders:
        print(f"⚠️  참가자 폴더를 찾을 수 없습니다.")
        return
    
    print(f"=" * 80)
    print(f"전체 참가자 시각화 시작")
    print(f"참가자 폴더 수: {len(participant_folders)}")
    print(f"=" * 80)
    
    for participant_folder in sorted(participant_folders):
        participant_name = participant_folder.name
        print(f"\n[{participant_name}]")
        
        # 참가자별 출력 디렉토리
        output_dir = Path(output_base_dir) / participant_name
        
        # CSV 파일 찾기
        csv_files = list(participant_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"  ⚠️  CSV 파일 없음")
            continue
        
        for csv_file in sorted(csv_files):
            try:
                visualize_signal(csv_file, output_dir, time_duration)
            except Exception as e:
                print(f"  ✗ 오류: {csv_file.name} - {str(e)}")
                continue
    
    print(f"\n" + "=" * 80)
    print("전체 시각화 완료!")
    print(f"저장 위치: {output_base_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # ========================================
    # 설정: 시각화할 시간 구간 (초)
    # ========================================
    TIME_DURATION = 5  # 앞/뒤로 표시할 시간 (초)
    
    # ========================================
    # 설정: 시각화할 파일 경로를 지정하세요
    # ========================================
    
    # 옵션 1: 단일 파일 시각화 (윈도우로 표시)
    FILE_PATH = r"F:\coding자료\coding\digital_hearth_care\dataset\GalaxyPPG\Dataset\P02\PolarH10\ECG.csv"
    
    # 옵션 2: 디렉토리 내 모든 CSV 시각화 (윈도우로 표시)
    # DIR_PATH = r"F:\coding자료\coding\digital_hearth_care\dataset\CLAS_Database\CLAS_Database\CLAS\Participants\Part1"
    
    # 옵션 3: 전체 참가자 폴더 시각화 (파일로 저장)
    # PARTICIPANTS_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\CLAS_Database\CLAS_Database\CLAS\Participants"
    # OUTPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\dataset\visualizations"
    
    # ========================================
    # 실행할 모드를 선택하세요 (하나만 주석 해제)
    # ========================================
    
    # 모드 1: 단일 파일
    if 'FILE_PATH' in locals():
        print("=" * 80)
        print("단일 파일 시각화 모드")
        print(f"시각화 구간: 앞/뒤 {TIME_DURATION}초")
        print("=" * 80)
        visualize_signal(FILE_PATH, output_dir=None, time_duration=TIME_DURATION)
    
    # 모드 2: 디렉토리
    # if 'DIR_PATH' in locals():
    #     print("=" * 80)
    #     print("디렉토리 시각화 모드")
    #     print(f"시각화 구간: 앞/뒤 {TIME_DURATION}초")
    #     print("=" * 80)
    #     visualize_directory(DIR_PATH, output_dir=None, time_duration=TIME_DURATION)
    
    # 모드 3: 전체 참가자 (저장)
    # if 'PARTICIPANTS_DIR' in locals() and 'OUTPUT_DIR' in locals():
    #     print("=" * 80)
    #     print("전체 참가자 시각화 모드 (파일 저장)")
    #     print(f"시각화 구간: 앞/뒤 {TIME_DURATION}초")
    #     print("=" * 80)
    #     visualize_all_participants(PARTICIPANTS_DIR, OUTPUT_DIR, time_duration=TIME_DURATION)
    
    print("\n프로그램 종료")
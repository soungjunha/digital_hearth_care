import os
import pandas as pd
import numpy as np
from scipy import signal
import h5py
from pathlib import Path
from tqdm import tqdm

# ==================== 경로 설정 (여기서 수정하세요) ====================
INPUT_DIR = r"F:\coding자료\coding\digital_hearth_care\test_set"  # 입력 CSV 파일들이 있는 디렉토리
OUTPUT_BASE_DIR = r"F:\coding자료\coding\digital_hearth_care\test_set\preprocess_test"  # 출력 베이스 디렉토리

# 출력 하위 디렉토리 (자동 생성됨)
OUTPUT_10SEC_DIR = os.path.join(OUTPUT_BASE_DIR, "10sec_test_segments")
OUTPUT_30SEC_DIR = os.path.join(OUTPUT_BASE_DIR, "30sec_test_segments")
OUTPUT_HDF5_10SEC = os.path.join(OUTPUT_BASE_DIR, "10sec_test_data.h5")
OUTPUT_HDF5_30SEC = os.path.join(OUTPUT_BASE_DIR, "30sec_test_data.h5")

# ==================== 전처리 파라미터 ====================
TARGET_FS = 256  # 목표 샘플링 주파수 (Hz)
WINDOW_10SEC = 10  # 10초 윈도우
WINDOW_30SEC = 30  # 30초 윈도우
# ====================================================================


def create_directories():
    """필요한 디렉토리 생성"""
    os.makedirs(OUTPUT_10SEC_DIR, exist_ok=True)
    os.makedirs(OUTPUT_30SEC_DIR, exist_ok=True)
    print(f"✓ 출력 디렉토리 생성 완료:")
    print(f"  - {OUTPUT_10SEC_DIR}")
    print(f"  - {OUTPUT_30SEC_DIR}")


def load_and_rename_columns(csv_path):
    """CSV 로드 및 컬럼명 변경"""
    df = pd.read_csv(csv_path)
    
    # 컬럼명 변경: (Time_sec, PPG, ECG) -> (time, ecg, ppg)
    df.columns = ['time', 'ppg', 'ecg']
    
    # 컬럼 순서 변경: (time, ecg, ppg)
    df = df[['time', 'ecg', 'ppg']]
    
    return df


def resample_signal(time, signal_data, target_fs):
    """신호를 목표 샘플링 주파수로 리샘플링"""
    # 원본 샘플링 주파수 계산
    time_diff = np.diff(time)
    original_fs = 1.0 / np.median(time_diff)
    
    # 리샘플링 비율 계산
    num_samples = len(signal_data)
    duration = time[-1] - time[0]
    target_num_samples = int(duration * target_fs)
    
    # scipy.signal.resample 사용
    resampled_signal = signal.resample(signal_data, target_num_samples)
    
    # 새로운 시간 축 생성 (0초부터 시작)
    resampled_time = np.linspace(0, duration, target_num_samples)
    
    return resampled_time, resampled_signal


def min_max_normalize(data):
    """Min-Max 정규화 [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
    
    normalized = (data - min_val) / (max_val - min_val)
    return normalized


def segment_data(df, window_sec, target_fs, overlap=0.5):
    """데이터를 지정된 윈도우 크기로 세그먼트 (50% 오버랩)"""
    window_samples = window_sec * target_fs
    step_samples = int(window_samples * (1 - overlap))  # 50% 오버랩
    total_samples = len(df)
    
    segments = []
    start_idx = 0
    
    while start_idx + window_samples <= total_samples:
        end_idx = start_idx + window_samples
        
        segment = df.iloc[start_idx:end_idx].copy()
        segment.reset_index(drop=True, inplace=True)
        segments.append(segment)
        
        start_idx += step_samples
    
    return segments


def process_single_file(csv_path):
    """단일 CSV 파일 전처리"""
    filename = os.path.basename(csv_path)
    base_name = os.path.splitext(filename)[0]
    
    # 1. CSV 로드 및 컬럼명 변경
    df = load_and_rename_columns(csv_path)
    
    # 2. 리샘플링
    time_resampled, ecg_resampled = resample_signal(
        df['time'].values, df['ecg'].values, TARGET_FS
    )
    _, ppg_resampled = resample_signal(
        df['time'].values, df['ppg'].values, TARGET_FS
    )
    
    # 3. Min-Max 정규화
    ecg_normalized = min_max_normalize(ecg_resampled)
    ppg_normalized = min_max_normalize(ppg_resampled)
    
    # 4. 정규화된 데이터프레임 생성
    df_processed = pd.DataFrame({
        'time': time_resampled,
        'ecg': ecg_normalized,
        'ppg': ppg_normalized
    })
    
    # 5. 10초 세그먼트 저장
    segments_10sec = segment_data(df_processed, WINDOW_10SEC, TARGET_FS)
    for i, seg in enumerate(segments_10sec):
        output_path = os.path.join(OUTPUT_10SEC_DIR, f"{base_name}_10sec_seg{i:04d}.csv")
        seg.to_csv(output_path, index=False)
    
    # 6. 30초 세그먼트 저장
    segments_30sec = segment_data(df_processed, WINDOW_30SEC, TARGET_FS)
    for i, seg in enumerate(segments_30sec):
        output_path = os.path.join(OUTPUT_30SEC_DIR, f"{base_name}_30sec_seg{i:04d}.csv")
        seg.to_csv(output_path, index=False)
    
    return len(segments_10sec), len(segments_30sec)


def create_hdf5_from_csv_folder(csv_folder, output_hdf5_path, window_sec):
    """CSV 폴더의 모든 파일을 통합 HDF5로 압축 (새로운 구조)"""
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"⚠ {csv_folder}에 CSV 파일이 없습니다.")
        return
    
    # 첫 번째 파일로 shape 확인
    first_df = pd.read_csv(os.path.join(csv_folder, csv_files[0]))
    sequence_length = len(first_df)
    n_samples = len(csv_files)
    
    # 데이터 수집용 리스트
    all_time = []
    all_ecg = []
    all_ppg = []
    all_filenames = []
    
    # 모든 CSV 파일 읽기
    for csv_file in tqdm(csv_files, desc=f"HDF5 생성 ({window_sec}초)"):
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        all_time.append(df['time'].values)
        all_ecg.append(df['ecg'].values)
        all_ppg.append(df['ppg'].values)
        all_filenames.append(csv_file)
    
    # numpy 배열로 변환
    time_array = np.array(all_time, dtype=np.float32)
    ecg_array = np.array(all_ecg, dtype=np.float32)
    ppg_array = np.array(all_ppg, dtype=np.float32)
    filenames_array = np.array(all_filenames, dtype='S')
    
    # HDF5 파일 생성
    with h5py.File(output_hdf5_path, 'w') as hf:
        # Attributes 설정
        hf.attrs['description'] = 'Preprocessed ECG and PPG signals for cardiac anomaly detection'
        hf.attrs['duration'] = window_sec
        hf.attrs['n_channels'] = 2
        hf.attrs['n_samples'] = n_samples
        hf.attrs['sampling_rate'] = TARGET_FS
        hf.attrs['sequence_length'] = sequence_length
        
        # Datasets 생성 (gzip 압축 레벨 4)
        hf.create_dataset('time', data=time_array, 
                         compression='gzip', compression_opts=4, dtype='float32')
        hf.create_dataset('ecg', data=ecg_array, 
                         compression='gzip', compression_opts=4, dtype='float32')
        hf.create_dataset('ppg', data=ppg_array, 
                         compression='gzip', compression_opts=4, dtype='float32')
        hf.create_dataset('filenames', data=filenames_array)
    
    print(f"✓ HDF5 파일 생성 완료: {output_hdf5_path}")
    print(f"  - Shape: ({n_samples}, {sequence_length})")
    print(f"  - 총 {n_samples}개 세그먼트 저장")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("PPG/ECG 전처리 파이프라인 시작")
    print("=" * 70)
    
    # 디렉토리 확인
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {INPUT_DIR}")
        return
    
    # 출력 디렉토리 생성
    create_directories()
    
    # 입력 CSV 파일 목록
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"❌ {INPUT_DIR}에 CSV 파일이 없습니다.")
        return
    
    print(f"\n총 {len(csv_files)}개의 CSV 파일 발견")
    print("-" * 70)
    
    # 각 파일 처리
    total_10sec = 0
    total_30sec = 0
    
    for csv_file in tqdm(csv_files, desc="파일 전처리"):
        csv_path = os.path.join(INPUT_DIR, csv_file)
        try:
            num_10sec, num_30sec = process_single_file(csv_path)
            total_10sec += num_10sec
            total_30sec += num_30sec
        except Exception as e:
            print(f"\n⚠ {csv_file} 처리 중 오류: {str(e)}")
            continue
    
    print("\n" + "-" * 70)
    print(f"✓ 전처리 완료:")
    print(f"  - 10초 세그먼트: {total_10sec}개")
    print(f"  - 30초 세그먼트: {total_30sec}개")
    
    # HDF5 압축
    print("\n" + "=" * 70)
    print("HDF5 압축 시작")
    print("=" * 70)
    
    create_hdf5_from_csv_folder(OUTPUT_10SEC_DIR, OUTPUT_HDF5_10SEC, WINDOW_10SEC)
    create_hdf5_from_csv_folder(OUTPUT_30SEC_DIR, OUTPUT_HDF5_30SEC, WINDOW_30SEC)
    
    print("\n" + "=" * 70)
    print("모든 작업 완료!")
    print("=" * 70)
    
    # 파일 크기 정보
    if os.path.exists(OUTPUT_HDF5_10SEC):
        size_10 = os.path.getsize(OUTPUT_HDF5_10SEC) / (1024**2)
        print(f"10초 HDF5 파일 크기: {size_10:.2f} MB")
    
    if os.path.exists(OUTPUT_HDF5_30SEC):
        size_30 = os.path.getsize(OUTPUT_HDF5_30SEC) / (1024**2)
        print(f"30초 HDF5 파일 크기: {size_30:.2f} MB")


if __name__ == "__main__":
    main()
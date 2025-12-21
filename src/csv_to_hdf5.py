import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ==========================================
# 📁 디렉토리 설정
# ==========================================
INPUT_DIR = r'F:\coding자료\coding\digital_hearth_care\test_set\IHD_10s'      # CSV 파일 디렉토리
OUTPUT_FILE = r'F:\coding자료\coding\digital_hearth_care\test_set\IHD_10s.h5'  # 출력 HDF5 파일

# 압축 설정
COMPRESSION = 'gzip'        # 'gzip', 'lzf', None
COMPRESSION_LEVEL = 4       # 0-9 (gzip만 해당)
# ==========================================


def csv_to_hdf5(input_dir, output_file, compression='gzip', compression_level=4):
    """
    CSV 파일들을 HDF5 포맷으로 변환
    
    Args:
        input_dir: CSV 파일들이 있는 디렉토리 경로
        output_file: 출력 HDF5 파일 경로
        compression: 압축 방식 ('gzip', 'lzf', None)
        compression_level: 압축 레벨 (0-9, gzip만 해당)
    """
    
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob('seg_*.csv'))
    
    if len(csv_files) == 0:
        print(f"❌ {input_dir}에서 seg_*.csv 파일을 찾을 수 없습니다.")
        return False
    
    print("="*70)
    print("📋 CSV → HDF5 변환 시작")
    print("="*70)
    print(f"📂 입력 디렉토리: {input_dir}")
    print(f"📁 출력 파일: {output_file}")
    print(f"📊 발견된 CSV 파일: {len(csv_files)}개")
    print(f"🗜️  압축 방식: {compression if compression else 'None'}")
    if compression == 'gzip':
        print(f"🗜️  압축 레벨: {compression_level}")
    print("="*70)
    
    # 첫 번째 파일로 형태 확인
    first_df = pd.read_csv(csv_files[0])
    n_samples = len(first_df)
    n_files = len(csv_files)
    
    print(f"\n📊 데이터 정보:")
    print(f"  - 파일당 샘플 수: {n_samples}")
    print(f"  - 컬럼: {list(first_df.columns)}")
    print(f"  - 전체 데이터 포인트: {n_files * n_samples:,}개")
    print()
    
    # 데이터 준비
    ecg_data = np.zeros((n_files, n_samples), dtype=np.float32)
    ppg_data = np.zeros((n_files, n_samples), dtype=np.float32)
    time_data = np.zeros((n_files, n_samples), dtype=np.float32)
    filenames = []
    
    # CSV 파일들 읽기
    print("📖 CSV 파일 읽는 중...")
    for idx, csv_file in enumerate(tqdm(csv_files, desc="Loading")):
        df = pd.read_csv(csv_file)
        
        ecg_data[idx] = df['ecg'].values
        ppg_data[idx] = df['ppg'].values
        time_data[idx] = df['time'].values
        filenames.append(csv_file.name)
    
    # HDF5에 데이터셋 저장
    print("\n💾 HDF5 파일로 저장 중...")
    
    # 출력 디렉토리 생성
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as hf:
        # ECG 데이터
        hf.create_dataset(
            'ecg',
            data=ecg_data,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None,
            dtype=np.float32
        )
        print("  ✓ ECG 데이터 저장 완료")
        
        # PPG 데이터
        hf.create_dataset(
            'ppg',
            data=ppg_data,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None,
            dtype=np.float32
        )
        print("  ✓ PPG 데이터 저장 완료")
        
        # Time 데이터
        hf.create_dataset(
            'time',
            data=time_data,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None,
            dtype=np.float32
        )
        print("  ✓ Time 데이터 저장 완료")
        
        # 파일명 저장 (메타데이터)
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('filenames', data=filenames, dtype=dt)
        print("  ✓ 파일명 저장 완료")
        
        # 메타데이터 속성 추가
        hf.attrs['n_samples'] = n_files
        hf.attrs['sequence_length'] = n_samples
        hf.attrs['sampling_rate'] = 256  # Hz
        hf.attrs['duration'] = 10  # seconds
        hf.attrs['n_channels'] = 2  # ECG, PPG
        hf.attrs['description'] = 'Preprocessed ECG and PPG signals for cardiac anomaly detection'
        print("  ✓ 메타데이터 저장 완료")
    
    # 결과 확인
    print("\n" + "="*70)
    print("📊 변환 결과")
    print("="*70)
    
    with h5py.File(output_file, 'r') as hf:
        print("\n🗂️  데이터셋:")
        for key in hf.keys():
            if key != 'filenames':
                dataset = hf[key]
                print(f"  - {key:10s}: shape={dataset.shape}, "
                      f"dtype={dataset.dtype}, compression={dataset.compression}")
        
        print(f"\n📝 메타데이터:")
        for key, value in hf.attrs.items():
            print(f"  - {key}: {value}")
        
        # 파일 크기 비교
        import os
        hdf5_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        csv_total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)  # MB
        
        print(f"\n💾 파일 크기:")
        print(f"  - CSV 전체:  {csv_total_size:.2f} MB")
        print(f"  - HDF5:      {hdf5_size:.2f} MB")
        print(f"  - 압축률:     {(1 - hdf5_size/csv_total_size)*100:.1f}% 절감 ✨")
    
    print("\n" + "="*70)
    print("✅ 변환 완료!")
    print(f"📁 저장 위치: {output_file}")
    print("="*70)
    
    return True


if __name__ == "__main__":
    # 변환 실행
    success = csv_to_hdf5(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL
    )
    
    if success:
        print("\n✨ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 변환 중 오류가 발생했습니다.")
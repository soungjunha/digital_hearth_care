import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ===== HDF5 데이터셋 클래스 =====
class HDF5CardiacDataset(Dataset):
    """HDF5 포맷 ECG, PPG 데이터셋 - 메모리 효율적"""
    
    def __init__(self, hdf5_file, indices=None, load_to_memory=True):
        """
        Args:
            hdf5_file: HDF5 파일 경로
            indices: 사용할 샘플 인덱스 (None이면 전체)
            load_to_memory: True면 전체 데이터를 메모리에 로드
        """
        self.hdf5_file = hdf5_file
        self.load_to_memory = load_to_memory
        
        # 메타데이터 읽기
        with h5py.File(hdf5_file, 'r') as hf:
            self.n_samples = hf.attrs['n_samples']
            self.sequence_length = hf.attrs['sequence_length']
            self.sampling_rate = hf.attrs.get('sampling_rate', 256)
            
            # 인덱스 설정
            if indices is None:
                self.indices = list(range(self.n_samples))
            else:
                self.indices = indices
            
            # 전체 데이터를 메모리에 로드
            if load_to_memory:
                self.ecg_data = hf['ecg'][:].astype(np.float32)
                self.ppg_data = hf['ppg'][:].astype(np.float32)
            else:
                self.ecg_data = None
                self.ppg_data = None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        if self.load_to_memory:
            ecg = self.ecg_data[real_idx]
            ppg = self.ppg_data[real_idx]
        else:
            with h5py.File(self.hdf5_file, 'r') as hf:
                ecg = hf['ecg'][real_idx]
                ppg = hf['ppg'][real_idx]
        
        # (2, sequence_length) 형태로 스택
        sample = np.stack([ecg, ppg], axis=0)
        sample = torch.FloatTensor(sample)
        
        return sample, sample


def create_train_val_datasets(hdf5_file, train_ratio=0.75, load_to_memory=True):
    """학습/검증 데이터셋 자동 분할"""
    with h5py.File(hdf5_file, 'r') as hf:
        n_samples = hf.attrs['n_samples']
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()
    
    train_dataset = HDF5CardiacDataset(hdf5_file, indices=train_indices, load_to_memory=load_to_memory)
    val_dataset = HDF5CardiacDataset(hdf5_file, indices=val_indices, load_to_memory=load_to_memory)
    
    return train_dataset, val_dataset


# ===== CNN-GRU 오토인코더 모델 =====
class CNNGRUAutoencoder(nn.Module):
    """CNN-GRU 기반 오토인코더"""
    
    def __init__(self, input_channels=2, sequence_length=2560, latent_dim=64):
        super(CNNGRUAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # ===== 인코더 =====
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.encoder_gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.encoder_fc = nn.Linear(128 * 2, latent_dim)
        
        # ===== 디코더 =====
        self.decoder_fc = nn.Linear(latent_dim, 128 * 40)
        
        self.decoder_gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=7, stride=4, padding=3, output_padding=3),
        )
    
    def encode(self, x):
        x = self.encoder_cnn(x)
        x = x.permute(0, 2, 1)
        _, hidden = self.encoder_gru(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        z = self.encoder_fc(hidden)
        return z
    
    def decode(self, z):
        batch_size = z.size(0)
        x = self.decoder_fc(z)
        x = x.view(batch_size, 40, 128)
        x, _ = self.decoder_gru(x)
        x = x.permute(0, 2, 1)
        x = self.decoder_cnn(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# ===== 이상 탐지기 =====
class AnomalyDetector:
    """재구성 오차 기반 이상 탐지"""
    
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.reconstruction_errors = []
    
    def calculate_reconstruction_error(self, original, reconstructed):
        mse = torch.mean((original - reconstructed) ** 2, dim=(1, 2))
        return mse
    
    def fit_threshold(self, dataloader, device='cpu'):
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                reconstructed, _ = self.model(data)
                error = self.calculate_reconstruction_error(data, reconstructed)
                errors.extend(error.cpu().numpy())
        
        self.reconstruction_errors = errors
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"이상 탐지 임계값 설정: {self.threshold:.6f} ({self.threshold_percentile}th percentile)")
        
        return self.threshold
    
    def detect(self, data, device='cpu'):
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(device)
            reconstructed, latent = self.model(data)
            error = self.calculate_reconstruction_error(data, reconstructed)
        
        is_anomaly = error > self.threshold
        
        return {
            'is_anomaly': is_anomaly.cpu().numpy(),
            'error': error.cpu().numpy(),
            'threshold': self.threshold,
            'reconstructed': reconstructed.cpu().numpy(),
            'latent': latent.cpu().numpy()
        }
    
    def get_warning_level(self, error):
        if error < self.threshold:
            return "정상", 0
        elif error < self.threshold * 1.5:
            return "경미한 이상", 1
        elif error < self.threshold * 2.0:
            return "주의", 2
        else:
            return "심각한 이상", 3


# ===== 학습 함수 =====
def train_autoencoder(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', model_save_path='best_model.pth'):
    """오토인코더 학습"""
    from tqdm import tqdm
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("✅ 첫 번째 배치 로딩 완료! 학습 진행 중...\n")
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0.0
        
        # 진행 상황 표시
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', ncols=100)
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 실시간 손실 표시
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', ncols=100, leave=False)
            for data, _ in pbar_val:
                data = data.to(device)
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
                
                pbar_val.set_postfix({'loss': f'{loss.item():.6f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} ⭐ (Best)')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses


# ===== 시각화 함수 =====
def visualize_reconstruction(original, reconstructed, sample_idx=0, save_path=None):
    """Compare original and reconstructed signals"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    axes[0].plot(original[sample_idx, 0, :], label='Original ECG', alpha=0.7, linewidth=1.5)
    axes[0].plot(reconstructed[sample_idx, 0, :], label='Reconstructed ECG', alpha=0.7, linewidth=1.5)
    axes[0].set_title('ECG Signal Reconstruction', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sample (256Hz)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(original[sample_idx, 1, :], label='Original PPG', alpha=0.7, linewidth=1.5)
    axes[1].plot(reconstructed[sample_idx, 1, :], label='Reconstructed PPG', alpha=0.7, linewidth=1.5)
    axes[1].set_title('PPG Signal Reconstruction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sample (256Hz)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_anomaly_distribution(errors, threshold, save_path=None):
    """Plot anomaly score distribution"""
    plt.figure(figsize=(12, 6))
    
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ===== 메인 실행 =====
if __name__ == "__main__":
    # ==========================================
    # 🎛️ 학습 설정
    # ==========================================
    
    # 📁 파일 경로
    HDF5_FILE = r'C:\Users\jerom\Downloads\model\dataset.h5'  # 입력 HDF5 파일
    MODEL_SAVE_PATH = r'C:\Users\jerom\Downloads\model\model_test.pth'  # 모델 저장 경로
    OUTPUT_DIR = r'C:\Users\jerom\Downloads\model\plots'  # 결과 저장 디렉토리
    
    # 🎯 하이퍼파라미터
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    LATENT_DIM = 128
    TRAIN_RATIO = 0.8
    LOAD_TO_MEMORY = True
    THRESHOLD_PERCENTILE = 95
    
    # 🖥️ 디바이스 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    
    print("="*70)
    print("🧠 심장질환 이상징후 탐지 모델 학습")
    print("="*70)
    print(f"📁 HDF5 파일: {HDF5_FILE}")
    print(f"💾 모델 저장: {MODEL_SAVE_PATH}")
    print(f"🖥️  디바이스: {DEVICE}")
    print(f"📊 배치 크기: {BATCH_SIZE}")
    print(f"🔄 에포크: {EPOCHS}")
    print(f"📈 학습률: {LEARNING_RATE}")
    print(f"🎯 잠재 차원: {LATENT_DIM}")
    print(f"📉 학습 비율: {TRAIN_RATIO*100:.0f}%")
    print(f"🚨 임계값 백분위: {THRESHOLD_PERCENTILE}")
    print("="*70)
    
    # ===== 1. HDF5 파일 로드 =====
    print("\nHDF5 파일 로드 중...")
    hdf5_file = HDF5_FILE
    
    with h5py.File(hdf5_file, 'r') as hf:
        print(f"전체 샘플 수: {hf.attrs['n_samples']}")
        print(f"시퀀스 길이: {hf.attrs['sequence_length']}")
        print(f"샘플링률: {hf.attrs['sampling_rate']} Hz")
    
    # ===== 2. 데이터셋 생성 =====
    train_dataset, val_dataset = create_train_val_datasets(
        hdf5_file, 
        train_ratio=TRAIN_RATIO,
        load_to_memory=LOAD_TO_MEMORY
    )
    
    # DataLoader 설정 (대용량 데이터셋 최적화)
    # Windows에서는 num_workers=0 필수!
    import platform
    is_windows = platform.system() == 'Windows'
    num_workers = 0 if is_windows else (4 if DEVICE.type == 'cuda' else 2)
    
    print(f"\nDataLoader 설정:")
    print(f"  - num_workers: {num_workers} ({'Windows 호환 모드' if is_windows else 'Linux/Mac 모드'})")
    print(f"  - pin_memory: {DEVICE.type == 'cuda'}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\n학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # ===== 3. 모델 생성 및 학습 =====
    print("\n모델 생성...")
    model = CNNGRUAutoencoder(
        input_channels=2,
        sequence_length=2560,
        latent_dim=LATENT_DIM
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print("\n학습 시작...")
    print("⏳ 첫 번째 배치 로딩 중... (10-30초 소요)")
    
    train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader,
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=DEVICE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # 학습 곡선 시각화
    plot_training_curves(
        train_losses, val_losses, 
        f'{OUTPUT_DIR}/training_curves_hdf5.png'
    )
    
    # ===== 4. 이상 탐지 설정 =====
    print("\n이상 탐지 임계값 설정 중...")
    detector = AnomalyDetector(model, threshold_percentile=THRESHOLD_PERCENTILE)
    detector.fit_threshold(train_loader, device=DEVICE)
    
    # 이상 점수 분포 시각화
    plot_anomaly_distribution(
        detector.reconstruction_errors,
        detector.threshold,
        f'{OUTPUT_DIR}/anomaly_distribution_hdf5.png'
    )
    
    # ===== 5. 전체 데이터 평가 =====
    print("\n전체 데이터 평가 중...")
    
    # 전체 데이터 로드
    full_dataset = HDF5CardiacDataset(hdf5_file, load_to_memory=True)
    full_loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
    
    for data, _ in full_loader:
        result = detector.detect(data, device=DEVICE)
        
        print("\n[전체 데이터 이상 탐지 결과]")
        print("-" * 60)
        
        with h5py.File(hdf5_file, 'r') as hf:
            filenames = [name.decode() if isinstance(name, bytes) else name 
                        for name in hf['filenames'][:]]
        
        for i, filename in enumerate(filenames):
            warning_level, severity = detector.get_warning_level(result['error'][i])
            anomaly_status = "🚨 이상" if result['is_anomaly'][i] else "✓ 정상"
            print(f"{filename:20s} | {anomaly_status:8s} | "
                  f"오차: {result['error'][i]:.6f} | {warning_level}")
        print("-" * 60)
        
        # 첫 번째 샘플 시각화
        visualize_reconstruction(
            data.cpu().numpy(),
            result['reconstructed'],
            sample_idx=0,
            save_path=f'{OUTPUT_DIR}/reconstruction_hdf5.png'
        )
        break
    
    print("\n" + "="*70)
    print("✅ HDF5 기반 모델 학습 및 평가 완료!")
    print(f"📁 저장 위치: {OUTPUT_DIR}")
    print(f"  - 모델: {MODEL_SAVE_PATH}")
    print(f"  - 학습 곡선: {OUTPUT_DIR}/training_curves_hdf5.png")
    print(f"  - 이상 분포: {OUTPUT_DIR}/anomaly_distribution_hdf5.png")
    print(f"  - 재구성 예시: {OUTPUT_DIR}/reconstruction_hdf5.png")
    print("="*70)
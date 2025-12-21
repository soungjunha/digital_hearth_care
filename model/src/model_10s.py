import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import platform
warnings.filterwarnings('ignore')

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac
else:
    plt.rc('font', family='NanumGothic')  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# ===== HDF5 데이터셋 클래스 =====
class HDF5CardiacDataset(Dataset):
    """HDF5 포맷 ECG, PPG 데이터셋 (표준 구조)"""
    
    def __init__(self, hdf5_file, indices=None, load_to_memory=True):
        self.hdf5_file = hdf5_file
        self.load_to_memory = load_to_memory
        
        with h5py.File(hdf5_file, 'r') as hf:
            # HDF5 파일 구조 확인
            print(f"\n📂 HDF5 파일 구조 확인: {Path(hdf5_file).name}")
            
            # 구조 타입 감지: 그룹 구조 vs 표준 구조
            keys = list(hf.keys())
            
            # 그룹 구조인지 확인 (seg0000, seg0001 형태)
            if keys and isinstance(hf[keys[0]], h5py.Group):
                print(f"   ⚠️  그룹 구조 감지! HDF5GroupCardiacDataset을 사용하세요.")
                raise ValueError("이 파일은 그룹 구조입니다. HDF5GroupCardiacDataset을 사용하세요.")
            
            print(f"   데이터셋 키: {keys}")
            if hf.attrs:
                print(f"   Attributes: {dict(hf.attrs)}")
            
            # 데이터셋 키 자동 감지
            if 'ecg' in hf.keys() and 'ppg' in hf.keys():
                ecg_key, ppg_key = 'ecg', 'ppg'
            elif 'ECG' in hf.keys() and 'PPG' in hf.keys():
                ecg_key, ppg_key = 'ECG', 'PPG'
            else:
                raise KeyError(f"ECG/PPG 데이터를 찾을 수 없습니다. 사용 가능한 키: {keys}")
            
            # attributes가 있으면 사용, 없으면 데이터 shape에서 추출
            if 'n_samples' in hf.attrs:
                self.n_samples = hf.attrs['n_samples']
                self.sequence_length = hf.attrs['sequence_length']
                self.sampling_rate = hf.attrs.get('sampling_rate', 256)
            else:
                self.n_samples = hf[ecg_key].shape[0]
                self.sequence_length = hf[ecg_key].shape[1]
                self.sampling_rate = 256
                print(f"   ⚠️  Attributes 없음. 자동 추출: n_samples={self.n_samples}, seq_len={self.sequence_length}")
            
            if indices is None:
                self.indices = list(range(self.n_samples))
            else:
                self.indices = indices
            
            if load_to_memory:
                self.ecg_data = hf[ecg_key][:].astype(np.float32)
                self.ppg_data = hf[ppg_key][:].astype(np.float32)
                print(f"   ✅ 메모리 로드 완료: ECG={self.ecg_data.shape}, PPG={self.ppg_data.shape}")
            else:
                self.ecg_data = None
                self.ppg_data = None
                self.ecg_key = ecg_key
                self.ppg_key = ppg_key
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        if self.load_to_memory:
            ecg = self.ecg_data[real_idx]
            ppg = self.ppg_data[real_idx]
        else:
            with h5py.File(self.hdf5_file, 'r') as hf:
                ecg_key = getattr(self, 'ecg_key', 'ecg')
                ppg_key = getattr(self, 'ppg_key', 'ppg')
                ecg = hf[ecg_key][real_idx]
                ppg = hf[ppg_key][real_idx]
        
        sample = np.stack([ecg, ppg], axis=0)
        sample = torch.FloatTensor(sample)
        
        return sample, sample


class HDF5GroupCardiacDataset(Dataset):
    """그룹 구조의 HDF5 포맷 ECG, PPG 데이터셋"""
    
    def __init__(self, hdf5_file, indices=None, load_to_memory=True):
        self.hdf5_file = hdf5_file
        self.load_to_memory = load_to_memory
        
        with h5py.File(hdf5_file, 'r') as hf:
            print(f"\n📂 그룹 구조 HDF5 파일: {Path(hdf5_file).name}")
            
            # 모든 그룹 이름 수집
            self.group_names = sorted([key for key in hf.keys() if isinstance(hf[key], h5py.Group)])
            self.n_samples = len(self.group_names)
            
            # 첫 번째 그룹에서 sequence_length 확인
            if self.n_samples > 0:
                first_group = hf[self.group_names[0]]
                self.sequence_length = first_group['ecg'].shape[0]
                self.sampling_rate = 256
            else:
                raise ValueError("그룹을 찾을 수 없습니다.")
            
            print(f"   그룹 개수: {self.n_samples}")
            print(f"   시퀀스 길이: {self.sequence_length}")
            print(f"   첫 번째 그룹: {self.group_names[0]}")
            print(f"   마지막 그룹: {self.group_names[-1]}")
            
            if indices is None:
                self.indices = list(range(self.n_samples))
            else:
                self.indices = indices
            
            # 메모리에 로드
            if load_to_memory:
                ecg_list = []
                ppg_list = []
                
                print(f"   메모리 로드 중...")
                for group_name in self.group_names:
                    group = hf[group_name]
                    ecg = group['ecg'][:].astype(np.float32)
                    ppg = group['ppg'][:].astype(np.float32)
                    ecg_list.append(ecg)
                    ppg_list.append(ppg)
                
                self.ecg_data = np.array(ecg_list, dtype=np.float32)
                self.ppg_data = np.array(ppg_list, dtype=np.float32)
                print(f"   ✅ 메모리 로드 완료: ECG={self.ecg_data.shape}, PPG={self.ppg_data.shape}")
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
                group_name = self.group_names[real_idx]
                ecg = hf[group_name]['ecg'][:].astype(np.float32)
                ppg = hf[group_name]['ppg'][:].astype(np.float32)
        
        sample = np.stack([ecg, ppg], axis=0)
        sample = torch.FloatTensor(sample)
        
        return sample, sample


def load_dataset_auto(hdf5_file, indices=None, load_to_memory=True):
    """HDF5 구조를 자동으로 감지하여 적절한 Dataset 반환"""
    with h5py.File(hdf5_file, 'r') as hf:
        keys = list(hf.keys())
        
        # 그룹 구조인지 확인
        if keys and isinstance(hf[keys[0]], h5py.Group):
            return HDF5GroupCardiacDataset(hdf5_file, indices, load_to_memory)
        else:
            return HDF5CardiacDataset(hdf5_file, indices, load_to_memory)


def create_train_val_test_datasets(hdf5_file, train_ratio=0.75, val_ratio=0.15, load_to_memory=True):
    """학습/검증/테스트 데이터셋 삼분할 (7.5:1.5:1)"""
    with h5py.File(hdf5_file, 'r') as hf:
        # 키 자동 감지
        if 'ecg' in hf.keys():
            ecg_key = 'ecg'
        elif 'ECG' in hf.keys():
            ecg_key = 'ECG'
        else:
            # 그룹 구조인 경우
            keys = list(hf.keys())
            if keys and isinstance(hf[keys[0]], h5py.Group):
                n_samples = len([k for k in keys if isinstance(hf[k], h5py.Group)])
            else:
                raise KeyError(f"ECG 데이터를 찾을 수 없습니다. 키: {list(hf.keys())}")
        
        if 'n_samples' in hf.attrs:
            n_samples = hf.attrs['n_samples']
        else:
            if 'ecg' in hf.keys():
                n_samples = hf['ecg'].shape[0]
            elif 'ECG' in hf.keys():
                n_samples = hf['ECG'].shape[0]
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    print(f"\n📦 데이터 분할 완료:")
    print(f"   전체: {n_samples:,}개")
    print(f"   학습: {len(train_indices):,}개 ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"   검증: {len(val_indices):,}개 ({len(val_indices)/n_samples*100:.1f}%)")
    print(f"   테스트: {len(test_indices):,}개 ({len(test_indices)/n_samples*100:.1f}%)")
    
    train_dataset = load_dataset_auto(hdf5_file, train_indices, load_to_memory)
    val_dataset = load_dataset_auto(hdf5_file, val_indices, load_to_memory)
    test_dataset = load_dataset_auto(hdf5_file, test_indices, load_to_memory)
    
    return train_dataset, val_dataset, test_dataset


# ===== CNN-GRU Autoencoder 모델 =====
class CNNGRUAutoencoder(nn.Module):
    def __init__(self, input_channels=2, sequence_length=2560, latent_dim=128):
        super().__init__()
        
        # Encoder: CNN + GRU
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.cnn_output_length = sequence_length // 8
        self.encoder_gru = nn.GRU(128, latent_dim, num_layers=2, batch_first=True, bidirectional=False)
        
        # Decoder: GRU + CNN
        self.decoder_gru = nn.GRU(latent_dim, 128, num_layers=2, batch_first=True, bidirectional=False)
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        x_cnn = self.encoder_cnn(x)
        x_cnn = x_cnn.permute(0, 2, 1)
        _, hidden = self.encoder_gru(x_cnn)
        
        # Decoder
        latent = hidden[-1].unsqueeze(1).repeat(1, self.cnn_output_length, 1)
        x_gru, _ = self.decoder_gru(latent)
        x_gru = x_gru.permute(0, 2, 1)
        reconstructed = self.decoder_cnn(x_gru)
        
        return reconstructed, hidden


# ===== 성능 평가 클래스 =====
class MedicalDevicePerformanceEvaluator:
    """식약처 체외진단의료기기 가이드라인 기반 성능 평가"""
    
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit_threshold(self, train_loader, device='cpu'):
        """정상 데이터 학습 세트로 임계값 설정"""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                reconstructed, _ = self.model(data)
                error = torch.mean((reconstructed - target) ** 2, dim=(1, 2)).cpu().numpy()
                errors.extend(error)
                
                del data, target, reconstructed, error
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"   임계값 (Percentile {self.threshold_percentile}): {self.threshold:.6f}")
        print(f"   학습 데이터 오차 범위: [{np.min(errors):.6f}, {np.max(errors):.6f}]")
    
    def evaluate_dataset(self, data_loader, true_label, dataset_name, device='cpu'):
        """데이터셋 평가 (true_label: 0=정상, 1=비정상)"""
        self.model.eval()
        errors = []
        labels = []
        predictions = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                reconstructed, _ = self.model(data)
                error = torch.mean((reconstructed - target) ** 2, dim=(1, 2)).cpu().numpy()
                errors.extend(error)
                labels.extend([true_label] * len(error))
                
                del data, target, reconstructed
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        errors = np.array(errors)
        predictions = (errors > self.threshold).astype(int)
        
        print(f"   {dataset_name} 오차 범위: [{errors.min():.6f}, {errors.max():.6f}]")
        print(f"   {dataset_name} 예측: 정상 {np.sum(predictions==0)}, 비정상 {np.sum(predictions==1)}")
        
        return errors, np.array(labels), predictions
    
    def calculate_medical_metrics(self, labels, predictions, errors):
        """식약처 가이드라인 기반 성능 지표"""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        fpr, tpr, _ = roc_curve(labels, errors)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(labels, errors)
        pr_auc = auc(recall, precision)
        
        return {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'Accuracy': accuracy,
            'F1_Score': f1,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'Threshold': self.threshold,
            'FPR': fpr,
            'TPR': tpr,
            'Precision_curve': precision,
            'Recall_curve': recall
        }


def plot_medical_performance_dashboard(metrics, normal_errors, abnormal_errors, save_path=None):
    """식약처 가이드라인 기반 시각화 대시보드"""
    
    # 대시보드 윈도우
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['예측: 정상', '예측: 비정상'],
                yticklabels=['실제: 정상', '실제: 비정상'])
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 2. 주요 지표 텍스트
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    metrics_text = f"""
    민감도: {metrics['Sensitivity']:.4f} ({metrics['Sensitivity']*100:.2f}%)
    특이도: {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)
    PPV:    {metrics['PPV']:.4f} ({metrics['PPV']*100:.2f}%)
    NPV:    {metrics['NPV']:.4f} ({metrics['NPV']*100:.2f}%)
    정확도: {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)
    F1:     {metrics['F1_Score']:.4f}
    
    ROC-AUC: {metrics['ROC_AUC']:.4f}
    PR-AUC:  {metrics['PR_AUC']:.4f}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=13, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax2.set_title('식약처 가이드라인 지표', fontsize=14, fontweight='bold')
    
    # 3. 재구성 오차 분포
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(normal_errors, bins=50, alpha=0.6, color='#3498DB', edgecolor='black',
             label=f'정상 (n={len(normal_errors)})', density=True)
    ax3.hist(abnormal_errors, bins=50, alpha=0.6, color='#E74C3C', edgecolor='black',
             label=f'비정상 (n={len(abnormal_errors)})', density=True)
    ax3.axvline(metrics['Threshold'], color='green', linestyle='--', linewidth=3,
                label=f"임계값 = {metrics['Threshold']:.6f}")
    ax3.set_xlabel('재구성 오차 (MSE)', fontsize=11)
    ax3.set_ylabel('밀도', fontsize=11)
    ax3.set_title('정상 vs 비정상 오차 분포', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. ROC Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(metrics['FPR'], metrics['TPR'], color='#E74C3C', linewidth=2.5,
             label=f'ROC (AUC = {metrics["ROC_AUC"]:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax4.fill_between(metrics['FPR'], metrics['TPR'], alpha=0.3, color='#E74C3C')
    ax4.set_xlabel('False Positive Rate', fontsize=11)
    ax4.set_ylabel('True Positive Rate', fontsize=11)
    ax4.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision-Recall Curve
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(metrics['Recall_curve'], metrics['Precision_curve'],
             color='#9B59B6', linewidth=2.5, label=f'PR (AUC = {metrics["PR_AUC"]:.3f})')
    ax5.fill_between(metrics['Recall_curve'], metrics['Precision_curve'], alpha=0.3, color='#9B59B6')
    ax5.set_xlabel('Recall', fontsize=11)
    ax5.set_ylabel('Precision', fontsize=11)
    ax5.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. 바이올린 플롯
    ax6 = fig.add_subplot(gs[1, 2])
    data_violin = [normal_errors, abnormal_errors]
    parts = ax6.violinplot(data_violin, positions=[1, 2], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.6)
    ax6.set_xticks([1, 2])
    ax6.set_xticklabels(['정상', '비정상'])
    ax6.set_ylabel('재구성 오차 (MSE)', fontsize=11)
    ax6.set_title('오차 분포 (Violin Plot)', fontsize=14, fontweight='bold')
    ax6.axhline(metrics['Threshold'], color='green', linestyle='--', linewidth=2, label='임계값')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. 박스플롯
    ax7 = fig.add_subplot(gs[2, 0])
    bp = ax7.boxplot(data_violin, labels=['정상', '비정상'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='black'),
                     medianprops=dict(color='red', linewidth=2))
    ax7.set_ylabel('재구성 오차 (MSE)', fontsize=11)
    ax7.set_title('오차 분포 (Box Plot)', fontsize=14, fontweight='bold')
    ax7.axhline(metrics['Threshold'], color='green', linestyle='--', linewidth=2, label='임계값')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. 통계 요약
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    stats_text = f"""
    정상 데이터:
      평균: {np.mean(normal_errors):.6f}
      중앙값: {np.median(normal_errors):.6f}
      표준편차: {np.std(normal_errors):.6f}
      범위: [{np.min(normal_errors):.6f}, {np.max(normal_errors):.6f}]
    
    비정상 데이터:
      평균: {np.mean(abnormal_errors):.6f}
      중앙값: {np.median(abnormal_errors):.6f}
      표준편차: {np.std(abnormal_errors):.6f}
      범위: [{np.min(abnormal_errors):.6f}, {np.max(abnormal_errors):.6f}]
    """
    ax8.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax8.set_title('통계 요약', fontsize=14, fontweight='bold')
    
    # 9. 성능 지표 막대 그래프
    ax9 = fig.add_subplot(gs[2, 2])
    metrics_names = ['민감도', '특이도', 'PPV', 'NPV', '정확도', 'F1']
    metrics_values = [metrics['Sensitivity'], metrics['Specificity'],
                      metrics['PPV'], metrics['NPV'], metrics['Accuracy'], metrics['F1_Score']]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    bars = ax9.barh(metrics_names, metrics_values, color=colors, edgecolor='black')
    ax9.set_xlim(0, 1)
    ax9.set_xlabel('값', fontsize=11)
    ax9.set_title('성능 지표 요약', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, metrics_values)):
        ax9.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.suptitle('식약처 체외진단의료기기 성능 평가 대시보드', fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 대시보드 저장: {save_path}")
    
    # --- [추가된 부분: 5개 개별 윈도우 생성] ---
    try:
        print("\n--- 개별 윈도우 생성 시작 ---")
        
        # 1. Confusion Matrix (새 윈도우)
        plt.figure(figsize=(8, 6))
        cm = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['예측: 정상', '예측: 비정상'],
                    yticklabels=['실제: 정상', '실제: 비정상'],
                    annot_kws={"size": 16})
        plt.title('Confusion Matrix (개별 윈도우)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 2. 주요 지표 (새 윈도우)
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111)
        ax.axis('off')
        metrics_text_large = f"""
        📊 식약처 가이드라인 성능 지표
        
        민감도 (Sensitivity):  {metrics['Sensitivity']:.4f} ({metrics['Sensitivity']*100:.2f}%)
        특이도 (Specificity):  {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)
        
        PPV (양성 예측도):     {metrics['PPV']:.4f} ({metrics['PPV']*100:.2f}%)
        NPV (음성 예측도):     {metrics['NPV']:.4f} ({metrics['NPV']*100:.2f}%)
        
        정확도 (Accuracy):     {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)
        F1 Score:              {metrics['F1_Score']:.4f}
        
        ROC-AUC:               {metrics['ROC_AUC']:.4f}
        PR-AUC:                {metrics['PR_AUC']:.4f}
        
        임계값 (Threshold):    {metrics['Threshold']:.6f}
        """
        ax.text(0.5, 0.5, metrics_text_large, fontsize=16, verticalalignment='center',
                horizontalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1))
        
        # 3. ROC Curve (새 윈도우)
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['FPR'], metrics['TPR'], color='#E74C3C', linewidth=3,
                 label=f'ROC (AUC = {metrics["ROC_AUC"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.fill_between(metrics['FPR'], metrics['TPR'], alpha=0.3, color='#E74C3C')
        plt.xlabel('False Positive Rate (1 - 특이도)', fontsize=12)
        plt.ylabel('True Positive Rate (민감도)', fontsize=12)
        plt.title('ROC Curve (개별 윈도우)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curve (새 윈도우)
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['Recall_curve'], metrics['Precision_curve'],
                 color='#9B59B6', linewidth=2.5, label=f'PR (AUC = {metrics["PR_AUC"]:.3f})')
        plt.fill_between(metrics['Recall_curve'], metrics['Precision_curve'], alpha=0.3, color='#9B59B6')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (개별 윈도우)', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 5. 재구성 오차 분포 (새 윈도우)
        plt.figure(figsize=(10, 6))
        plt.hist(normal_errors, bins=50, alpha=0.6, color='#3498DB', edgecolor='black',
                 label=f'정상 (n={len(normal_errors)})', density=True)
        plt.hist(abnormal_errors, bins=50, alpha=0.6, color='#E74C3C', edgecolor='black',
                 label=f'비정상 (n={len(abnormal_errors)})', density=True)
        plt.axvline(metrics['Threshold'], color='green', linestyle='--', linewidth=3,
                    label=f"임계값 = {metrics['Threshold']:.6f}")
        plt.xlabel('재구성 오차 (MSE)', fontsize=12)
        plt.ylabel('밀도', fontsize=12)
        plt.title('정상 vs 비정상 재구성 오차 분포 (개별 윈도우)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        print("--- ✅ 개별 윈도우 5개 생성 완료 ---")
        
    except Exception as e:
        print(f"--- ⚠️ 개별 윈도우 생성 중 오류 발생: {e} ---")
    # --- [추가된 부분 끝] ---

    # 기존 대시보드 윈도우 (1개) + 개별 윈도우 (5개) = 총 6개 윈도우가 뜸
    plt.show()


def print_performance_report(metrics):
    """성능 평가 리포트"""
    print("\n" + "="*80)
    print("📋 식약처 체외진단의료기기 성능 평가 리포트")
    print("="*80)
    print(f"\n[Confusion Matrix]")
    print(f"  TP: {metrics['TP']:5d}  |  TN: {metrics['TN']:5d}  |  FP: {metrics['FP']:5d}  |  FN: {metrics['FN']:5d}")
    print(f"\n[주요 성능 지표]")
    print(f"  민감도: {metrics['Sensitivity']:.4f} ({metrics['Sensitivity']*100:.2f}%)")
    print(f"  특이도: {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)")
    print(f"  PPV:    {metrics['PPV']:.4f} ({metrics['PPV']*100:.2f}%)")
    print(f"  NPV:    {metrics['NPV']:.4f} ({metrics['NPV']*100:.2f}%)")
    print(f"  정확도: {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"  F1:     {metrics['F1_Score']:.4f}")
    print(f"\n[AUC]")
    print(f"  ROC: {metrics['ROC_AUC']:.4f}  |  PR: {metrics['PR_AUC']:.4f}")
    print("="*80 + "\n")


def train_autoencoder(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', model_save_path='best_model.pth'):
    """오토인코더 학습"""
    from tqdm import tqdm
    
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for data, target in tqdm(train_loader, desc=f'[Epoch {epoch+1}/{epochs}] Training'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            del data, target, reconstructed, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'[Epoch {epoch+1}/{epochs}] Validation'):
                data, target = data.to(device), target.to(device)
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, target)
                val_loss += loss.item()
                del data, target, reconstructed, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch [{epoch+1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f} ⭐')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}')


# ===== 메인 =====
if __name__ == "__main__":
    # 파일 경로 (사용자가 수정)
    NORMAL_HDF5_FILE = r'F:\coding자료\coding\digital_hearth_care\model_2\dataset_10sec.h5'
    
    # ===== 비정상 데이터셋 5개 =====
    ABNORMAL_HDF5_FILES = [
        r'F:\coding자료\coding\digital_hearth_care\model_2\AF_10s.h5',
        r'F:\coding자료\coding\digital_hearth_care\model_2\Arrhythmia_10s.h5',
        r'F:\coding자료\coding\digital_hearth_care\model_2\HF_10s.h5',
        r'F:\coding자료\coding\digital_hearth_care\model_2\hypertension_10s.h5',
        r'F:\coding자료\coding\digital_hearth_care\model_2\IHD_10s.h5'
    ]
    
    MODEL_SAVE_PATH = r'F:\coding자료\coding\digital_hearth_care\model_2\10sec\model_test.pth'
    OUTPUT_DIR = r'F:\coding자료\coding\digital_hearth_care\model_2\10sec\performance'
    
    # 하이퍼파라미터
    BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    LATENT_DIM = 128
    LOAD_TO_MEMORY = True
    THRESHOLD_PERCENTILE = 95
    
    # 데이터 분할 비율 (7.5:1.5:1)
    TRAIN_RATIO = 0.75  # 75%
    VAL_RATIO = 0.15    # 15%
    TEST_RATIO = 0.10   # 10%
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("🏥 심장 이상징후 탐지 - 식약처 가이드라인 기반 성능 평가 (비정상 데이터셋 5개)")
    print("="*80)
    print(f"📁 정상 데이터: {NORMAL_HDF5_FILE}")
    print(f"📁 비정상 데이터셋 개수: {len(ABNORMAL_HDF5_FILES)}개")
    for i, filepath in enumerate(ABNORMAL_HDF5_FILES, 1):
        print(f"   {i}. {filepath}")
    print(f"🖥️  디바이스: {DEVICE}")
    print("="*80)
    
    # 비정상 파일 존재 확인
    missing_files = []
    for filepath in ABNORMAL_HDF5_FILES:
        if not Path(filepath).exists():
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n❌ 다음 비정상 데이터 파일이 없습니다:")
        for filepath in missing_files:
            print(f"   - {filepath}")
        print(f"\n   비정상 데이터 HDF5 파일을 준비해주세요.")
        exit(1)
    
    # 학습 (또는 기존 모델 로드)
    print("\n[1단계] 모델 준비")
    print("-" * 80)
    
    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        NORMAL_HDF5_FILE, train_ratio=0.75, val_ratio=0.15, load_to_memory=LOAD_TO_MEMORY
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = CNNGRUAutoencoder(input_channels=2, sequence_length=2560, latent_dim=LATENT_DIM)
    
    if Path(MODEL_SAVE_PATH).exists():
        print(f"✅ 기존 모델 로드: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        print(f"⚠️  기존 모델 없음. 새로 학습합니다...")
        model = model.to(DEVICE)
        train_autoencoder(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, MODEL_SAVE_PATH)
    
    # 성능 평가
    print("\n[2단계] 성능 평가 (테스트 데이터만 사용)")
    print("-" * 80)
    print("⚠️  주의: 학습에 사용되지 않은 테스트 데이터만 평가합니다!")
    
    evaluator = MedicalDevicePerformanceEvaluator(model, THRESHOLD_PERCENTILE)
    
    # 학습 데이터로 임계값 설정
    print("\n🎯 학습 데이터로 임계값 설정")
    evaluator.fit_threshold(train_loader, device=DEVICE)
    
    # ===== 비정상 데이터 5개 평가 =====
    print("\n📊 비정상 데이터셋 5개 평가 시작")
    print("-" * 80)
    
    all_abnormal_errors = []
    all_abnormal_labels = []
    all_abnormal_preds = []
    
    for i, abnormal_file in enumerate(ABNORMAL_HDF5_FILES, 1):
        print(f"\n[비정상 데이터셋 {i}/{len(ABNORMAL_HDF5_FILES)}]")
        print(f"📂 파일: {Path(abnormal_file).name}")
        
        abnormal_dataset = load_dataset_auto(abnormal_file, load_to_memory=LOAD_TO_MEMORY)
        abnormal_loader = DataLoader(abnormal_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
        
        errors, labels, preds = evaluator.evaluate_dataset(
            abnormal_loader, true_label=1, dataset_name=f"비정상_{i}", device=DEVICE
        )
        
        all_abnormal_errors.append(errors)
        all_abnormal_labels.append(labels)
        all_abnormal_preds.append(preds)
        
        print(f"   ✅ 평가 완료: {len(errors)}개 샘플")
    
    # 모든 비정상 데이터 결합
    abnormal_errors = np.concatenate(all_abnormal_errors)
    abnormal_labels = np.concatenate(all_abnormal_labels)
    abnormal_preds = np.concatenate(all_abnormal_preds)
    
    print("\n" + "="*80)
    print(f"📊 비정상 데이터 통합 결과")
    print(f"   총 비정상 샘플: {len(abnormal_errors):,}개")
    print(f"   예측 결과: 정상 {np.sum(abnormal_preds==0):,}개, 비정상 {np.sum(abnormal_preds==1):,}개")
    print("="*80)
    
    # ===== 정상 데이터를 비정상 데이터와 같은 양으로 샘플링 =====
    print("\n📊 정상 데이터 테스트 세트 평가 (비정상 데이터와 동일한 샘플 수)")
    print("-" * 80)
    
    total_abnormal_count = len(abnormal_errors)
    
    # 테스트 데이터셋에서 비정상 데이터 개수만큼만 샘플링
    if len(test_dataset) < total_abnormal_count:
        print(f"⚠️  경고: 테스트 데이터({len(test_dataset)}개)가 비정상 데이터({total_abnormal_count}개)보다 적습니다.")
        print(f"   사용 가능한 모든 테스트 데이터를 사용합니다.")
        sampled_indices = list(range(len(test_dataset)))
    else:
        # 랜덤 샘플링 (재현성을 위해 seed 설정)
        np.random.seed(42)
        sampled_indices = np.random.choice(len(test_dataset), total_abnormal_count, replace=False).tolist()
        print(f"   정상 테스트 데이터 {len(test_dataset):,}개 중 {total_abnormal_count:,}개 샘플링")
    
    # 샘플링된 인덱스로 새로운 데이터셋 생성
    sampled_test_dataset = load_dataset_auto(NORMAL_HDF5_FILE, indices=[test_dataset.indices[i] for i in sampled_indices], load_to_memory=LOAD_TO_MEMORY)
    sampled_test_loader = DataLoader(sampled_test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    
    normal_errors, normal_labels, normal_preds = evaluator.evaluate_dataset(
        sampled_test_loader, true_label=0, dataset_name="정상 (Test-Sampled)", device=DEVICE
    )
    
    print("\n" + "="*80)
    print(f"📊 최종 데이터 균형 확인")
    print(f"   정상 샘플: {len(normal_errors):,}개")
    print(f"   비정상 샘플: {len(abnormal_errors):,}개")
    print(f"   샘플 비율: 1:1 (균형 맞춤)")
    print("="*80)
    
    # 전체 결합
    all_errors = np.concatenate([normal_errors, abnormal_errors])
    all_labels = np.concatenate([normal_labels, abnormal_labels])
    all_preds = np.concatenate([normal_preds, abnormal_preds])
    
    # 성능 지표 계산
    metrics = evaluator.calculate_medical_metrics(all_labels, all_preds, all_errors)
    
    # 결과 출력 및 시각화
    print_performance_report(metrics)
    plot_medical_performance_dashboard(metrics, normal_errors, abnormal_errors,
                                       save_path=f'{OUTPUT_DIR}/medical_performance_dashboard_5datasets.png')
    
    print("\n✅ 평가 완료!")
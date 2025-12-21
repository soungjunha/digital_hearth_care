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
            print(f"\n📂 HDF5 파일: {Path(hdf5_file).name}")
            
            keys = list(hf.keys())
            
            # 그룹 구조인지 확인
            if keys and isinstance(hf[keys[0]], h5py.Group):
                print(f"   ⚠️  그룹 구조 감지! HDF5GroupCardiacDataset을 사용하세요.")
                raise ValueError("이 파일은 그룹 구조입니다. HDF5GroupCardiacDataset을 사용하세요.")
            
            # 데이터셋 키 자동 감지
            if 'ecg' in hf.keys() and 'ppg' in hf.keys():
                ecg_key, ppg_key = 'ecg', 'ppg'
            elif 'ECG' in hf.keys() and 'PPG' in hf.keys():
                ecg_key, ppg_key = 'ECG', 'PPG'
            else:
                raise KeyError(f"ECG/PPG 데이터를 찾을 수 없습니다. 키: {keys}")
            
            if 'n_samples' in hf.attrs:
                self.n_samples = hf.attrs['n_samples']
                self.sequence_length = hf.attrs['sequence_length']
                self.sampling_rate = hf.attrs.get('sampling_rate', 256)
            else:
                self.n_samples = hf[ecg_key].shape[0]
                self.sequence_length = hf[ecg_key].shape[1]
                self.sampling_rate = 256
            
            if indices is None:
                self.indices = list(range(self.n_samples))
            else:
                self.indices = indices
            
            if load_to_memory:
                self.ecg_data = hf[ecg_key][:].astype(np.float32)
                self.ppg_data = hf[ppg_key][:].astype(np.float32)
                print(f"   ✅ 로드 완료: {self.n_samples}개, 길이={self.sequence_length}, ECG={self.ecg_data.shape}")
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
            print(f"\n📂 그룹 구조 HDF5: {Path(hdf5_file).name}")
            
            self.group_names = sorted([key for key in hf.keys() if isinstance(hf[key], h5py.Group)])
            self.n_samples = len(self.group_names)
            
            if self.n_samples > 0:
                first_group = hf[self.group_names[0]]
                self.sequence_length = first_group['ecg'].shape[0]
                self.sampling_rate = 256
            else:
                raise ValueError("그룹을 찾을 수 없습니다.")
            
            if indices is None:
                self.indices = list(range(self.n_samples))
            else:
                self.indices = indices
            
            if load_to_memory:
                ecg_list = []
                ppg_list = []
                
                for group_name in self.group_names:
                    group = hf[group_name]
                    ecg = group['ecg'][:].astype(np.float32)
                    ppg = group['ppg'][:].astype(np.float32)
                    ecg_list.append(ecg)
                    ppg_list.append(ppg)
                
                self.ecg_data = np.array(ecg_list, dtype=np.float32)
                self.ppg_data = np.array(ppg_list, dtype=np.float32)
                print(f"   ✅ 로드 완료: {self.n_samples}개, 길이={self.sequence_length}")
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
        
        if keys and isinstance(hf[keys[0]], h5py.Group):
            return HDF5GroupCardiacDataset(hdf5_file, indices, load_to_memory)
        else:
            return HDF5CardiacDataset(hdf5_file, indices, load_to_memory)


def create_train_val_test_datasets(hdf5_file, train_ratio=0.75, val_ratio=0.15, load_to_memory=True):
    """학습/검증/테스트 데이터셋 삼분할 (7.5:1.5:1)"""
    with h5py.File(hdf5_file, 'r') as hf:
        if 'ecg' in hf.keys():
            ecg_key = 'ecg'
        elif 'ECG' in hf.keys():
            ecg_key = 'ECG'
        else:
            keys = list(hf.keys())
            if keys and isinstance(hf[keys[0]], h5py.Group):
                n_samples = len([k for k in keys if isinstance(hf[k], h5py.Group)])
            else:
                raise KeyError(f"ECG 데이터를 찾을 수 없습니다. 키: {list(hf.keys())}")
        
        if 'n_samples' in hf.attrs:
            n_samples = hf.attrs['n_samples']
        else:
            if 'ecg_key' in locals():
                n_samples = hf[ecg_key].shape[0]
    
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    print(f"\n📊 데이터 분할 (총 {n_samples}개)")
    print(f"   Train: {len(train_indices)}개 ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"   Val:   {len(val_indices)}개 ({len(val_indices)/n_samples*100:.1f}%)")
    print(f"   Test:  {len(test_indices)}개 ({len(test_indices)/n_samples*100:.1f}%)")
    
    train_dataset = load_dataset_auto(hdf5_file, indices=train_indices, load_to_memory=load_to_memory)
    val_dataset = load_dataset_auto(hdf5_file, indices=val_indices, load_to_memory=load_to_memory)
    test_dataset = load_dataset_auto(hdf5_file, indices=test_indices, load_to_memory=load_to_memory)
    
    return train_dataset, val_dataset, test_dataset


# ===== CNN-GRU 오토인코더 모델 (30초용) =====
class CNNGRUAutoencoder30s(nn.Module):
    """CNN-GRU 기반 오토인코더 (30초 = 7680 samples)"""
    
    def __init__(self, input_channels=2, sequence_length=7680, latent_dim=128):
        super(CNNGRUAutoencoder30s, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encoder CNN: 7680 → 120
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=4, padding=3),  # 7680 → 1920
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),  # 1920 → 480
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),  # 480 → 120
            nn.BatchNorm1d(128),
            nn.ReLU(),
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
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 120)
        
        self.decoder_gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        
        # Decoder CNN: 120 → 7680
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=4, padding=2, output_padding=3),  # 120 → 480
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4, padding=2, output_padding=3),  # 480 → 1920
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=7, stride=4, padding=3, output_padding=3),  # 1920 → 7680
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
        x = x.view(batch_size, 120, 128)
        x, _ = self.decoder_gru(x)
        x = x.permute(0, 2, 1)
        x = self.decoder_cnn(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# ===== 의료기기 성능 평가 클래스 =====
class MedicalDevicePerformanceEvaluator:
    """식약처 가이드라인 기반 의료기기 성능 평가"""
    
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
        
        print(f"정상 데이터로 임계값 설정 중...")
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                reconstructed, _ = self.model(data)
                error = self.calculate_reconstruction_error(data, reconstructed)
                errors.extend(error.cpu().numpy())
                
                del data, reconstructed, error
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.reconstruction_errors = errors
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"✅ 이상 탐지 임계값: {self.threshold:.6f} ({self.threshold_percentile}th percentile)")
        print(f"   정상 데이터 오차 범위: [{np.min(errors):.6f}, {np.max(errors):.6f}]")
        print(f"   정상 데이터 평균 오차: {np.mean(errors):.6f}")
        
        return self.threshold
    
    def evaluate_dataset(self, dataloader, true_label, dataset_name, device='cpu'):
        self.model.eval()
        all_errors = []
        
        print(f"\n📊 {dataset_name} 데이터 평가 중...")
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                reconstructed, _ = self.model(data)
                error = self.calculate_reconstruction_error(data, reconstructed)
                
                all_errors.extend(error.cpu().numpy())
                
                del data, reconstructed, error
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        all_errors = np.array(all_errors)
        predictions = (all_errors > self.threshold).astype(int)
        true_labels = np.full(len(all_errors), true_label)
        
        print(f"✅ {dataset_name} 평가 완료: {len(all_errors)}개")
        print(f"   오차 범위: [{np.min(all_errors):.6f}, {np.max(all_errors):.6f}]")
        print(f"   평균 오차: {np.mean(all_errors):.6f}")
        print(f"   이상 예측: {np.sum(predictions)}개 ({np.sum(predictions)/len(predictions)*100:.2f}%)")
        
        return all_errors, true_labels, predictions
    
    def calculate_medical_metrics(self, y_true, y_pred, scores):
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0, 0]
                fp, fn, tp = 0, 0, 0
            else:
                tp = cm[0, 0]
                tn, fp, fn = 0, 0, 0
        else:
            raise ValueError(f"Unexpected confusion matrix: {cm.shape}")
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
        except:
            fpr, tpr = [0, 1], [0, 1]
            roc_auc = 0.5
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)
        except:
            precision, recall = [1, 0], [0, 1]
            pr_auc = 0.5
        
        return {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Sensitivity': sensitivity, 'Specificity': specificity,
            'PPV': ppv, 'NPV': npv, 'Accuracy': accuracy, 'F1_Score': f1_score,
            'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'Threshold': self.threshold,
            'FPR': fpr, 'TPR': tpr,
            'Precision_curve': precision, 'Recall_curve': recall
        }


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

# (이 파일의 다른 함수들은 생략... HDF5CardiacDataset, CNNGRUAutoencoder30s 등...)
# (아래는 수정된 시각화 함수입니다)

# ===== 성능 시각화 (동일) =====
def plot_medical_performance_dashboard(metrics, normal_errors, abnormal_errors, save_path=None):
    """식약처 가이드라인 기반 성능 대시보드"""
    
    # 1. --- 전체 대시보드 그리기 (Figure 1) ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['정상(Pred)', '이상(Pred)'],
                yticklabels=['정상(True)', '이상(True)'],
                cbar_kws={'label': '샘플 수'}, annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
    
    # 2. 성능 지표
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_names = ['민감도\n(Sensitivity)', '특이도\n(Specificity)', 'PPV', 'NPV', '정확도\n(Accuracy)', 'F1']
    metrics_values = [metrics['Sensitivity'], metrics['Specificity'], 
                      metrics['PPV'], metrics['NPV'], metrics['Accuracy'], metrics['F1_Score']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    bars = ax2.barh(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlim([0, 1.0])
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_title('식약처 가이드라인 주요 성능 지표', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0.95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='목표: 95%')
    ax2.legend(fontsize=10)
    
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        color = 'green' if value >= 0.95 else 'red' if value < 0.85 else 'orange'
        ax2.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=11, fontweight='bold', color=color)
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(metrics['FPR'], metrics['TPR'], color='#E74C3C', linewidth=2.5, 
             label=f'ROC (AUC = {metrics["ROC_AUC"]:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    ax3.fill_between(metrics['FPR'], metrics['TPR'], alpha=0.3, color='#E74C3C')
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. PR Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(metrics['Recall_curve'], metrics['Precision_curve'], 
             color='#9B59B6', linewidth=2.5, label=f'PR (AUC = {metrics["PR_AUC"]:.3f})')
    ax4.fill_between(metrics['Recall_curve'], metrics['Precision_curve'], alpha=0.3, color='#9B59B6')
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. 재구성 오차 분포
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.hist(normal_errors, bins=50, alpha=0.6, color='#3498DB', edgecolor='black',
             label=f'정상 (n={len(normal_errors)})', density=True)
    ax5.hist(abnormal_errors, bins=50, alpha=0.6, color='#E74C3C', edgecolor='black',
             label=f'비정상 (n={len(abnormal_errors)})', density=True)
    ax5.axvline(metrics['Threshold'], color='green', linestyle='--', linewidth=3,
                label=f"임계값 = {metrics['Threshold']:.6f}")
    ax5.set_xlabel('재구성 오차 (MSE)', fontsize=12)
    ax5.set_ylabel('밀도', fontsize=12)
    ax5.set_title('정상 vs 비정상 재구성 오차 분포', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 성능 지표 테이블
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    table_data = [
        ['지표', '값', '설명'],
        ['민감도 (Sensitivity)', f'{metrics["Sensitivity"]:.4f}', '실제 이상 중 양성 판정 비율'],
        ['특이도 (Specificity)', f'{metrics["Specificity"]:.4f}', '실제 정상 중 음성 판정 비율'],
        ['양성 예측도 (PPV)', f'{metrics["PPV"]:.4f}', '양성 판정 중 실제 이상 비율'],
        ['음성 예측도 (NPV)', f'{metrics["NPV"]:.4f}', '음성 판정 중 실제 정상 비율'],
        ['정확도 (Accuracy)', f'{metrics["Accuracy"]:.4f}', '전체 중 올바른 판정 비율'],
        ['F1 Score', f'{metrics["F1_Score"]:.4f}', 'Precision과 Recall의 조화평균'],
        ['ROC AUC', f'{metrics["ROC_AUC"]:.4f}', 'ROC 곡선 아래 면적'],
        ['PR AUC', f'{metrics["PR_AUC"]:.4f}', 'PR 곡선 아래 면적'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.25, 0.15, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    for i in range(1, len(table_data)):
        for j in range(3):
            table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            table[(i, j)].set_edgecolor('black')
    
    ax6.set_title('의료기기 성능 지표 상세', fontsize=14, fontweight='bold', pad=20)
    
    # 7. 종합 평가
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    cohen_d = (np.mean(abnormal_errors) - np.mean(normal_errors)) / np.sqrt((np.std(normal_errors)**2 + np.std(abnormal_errors)**2) / 2)
    
    summary_text = f"""
    ═══════════════════════════════════════════════════════════════════════════════════
    
    📊 식약처 체외진단의료기기 성능 평가 요약 (30초 모델)
    
    [Confusion Matrix]  TP: {metrics['TP']:5d}  |  TN: {metrics['TN']:5d}  |  FP: {metrics['FP']:5d}  |  FN: {metrics['FN']:5d}
    
    [성능 지표]  민감도: {metrics['Sensitivity']*100:.2f}%  |  특이도: {metrics['Specificity']*100:.2f}%  |  정확도: {metrics['Accuracy']*100:.2f}%
    
    [재구성 오차]  정상 평균: {np.mean(normal_errors):.6f}  |  비정상 평균: {np.mean(abnormal_errors):.6f}
    
    [분리도]  Cohen's d = {cohen_d:.3f}  (효과크기: {'Large' if abs(cohen_d) > 0.8 else 'Medium' if abs(cohen_d) > 0.5 else 'Small'})
    
    ═══════════════════════════════════════════════════════════════════════════════════
    """
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('심장 이상징후 탐지 (30초) - 식약처 가이드라인 기반 성능 평가', fontsize=16, fontweight='bold', y=0.995)

    
    # 2. --- [수정됨] 개별 차트 저장 및 팝업 (Figure 2~6) ---
    
    # 공통 헬퍼 함수 정의
    def save_current_figure(directory, stem, name):
        if directory:
            try:
                filepath = directory / f"{stem}_{name}.png"
                plt.gcf().savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"   ✅ {filepath.name} 저장 완료")
            except Exception as e:
                print(f"   ⚠️  {name}.png 저장 실패: {e}")

    # 저장 경로 설정
    if save_path:
        p = Path(save_path)
        directory = p.parent
        stem = p.stem # 예: 'medical_performance_dashboard_30s'
        directory.mkdir(parents=True, exist_ok=True)
        print(f"--- 📝 개별 차트 저장 위치: {directory} ---")
    else:
        directory = None
        stem = None

    # --- 1. Confusion Matrix (새 윈도우, Figure 2) ---
    plt.figure(figsize=(8, 6))
    cm_data = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
    ax_cm_new = sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['정상(Pred)', '이상(Pred)'],
                            yticklabels=['정상(True)', '이상(True)'],
                            cbar_kws={'label': '샘플 수'}, annot_kws={'size': 14, 'weight': 'bold'})
    ax_cm_new.set_title('Confusion Matrix (30s, 개별 윈도우)', fontsize=14, fontweight='bold', pad=10)
    save_current_figure(directory, stem, "1_ConfusionMatrix")

    # --- 2. 성능 지표 (새 윈도우, Figure 3) ---
    plt.figure(figsize=(10, 7))
    metrics_names_popup = ['민감도\n(Sensitivity)', '특이도\n(Specificity)', 'PPV', 'NPV', '정확도\n(Accuracy)', 'F1']
    metrics_values_popup = [metrics['Sensitivity'], metrics['Specificity'], 
                            metrics['PPV'], metrics['NPV'], metrics['Accuracy'], metrics['F1_Score']]
    colors_popup = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    ax_metrics_new = plt.barh(metrics_names_popup, metrics_values_popup, color=colors_popup, edgecolor='black', linewidth=1.5)
    plt.xlim([0, 1.0])
    plt.xlabel('Score', fontsize=12)
    plt.title('식약처 가이드라인 주요 성능 지표 (30s, 개별 윈도우)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0.95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='목표: 95%')
    plt.legend(fontsize=10)
    for i, (bar, value) in enumerate(zip(ax_metrics_new, metrics_values_popup)):
        color = 'green' if value >= 0.95 else 'red' if value < 0.85 else 'orange'
        plt.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=11, fontweight='bold', color=color)
    save_current_figure(directory, stem, "2_PerformanceMetrics")

    # --- 3. ROC Curve (새 윈도우, Figure 4) ---
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['FPR'], metrics['TPR'], color='#E74C3C', linewidth=2.5, 
             label=f'ROC (AUC = {metrics["ROC_AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    plt.fill_between(metrics['FPR'], metrics['TPR'], alpha=0.3, color='#E74C3C')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (30s, 개별 윈도우)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    save_current_figure(directory, stem, "3_ROCCurve")

    # --- 4. PR Curve (새 윈도우, Figure 5) ---
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['Recall_curve'], metrics['Precision_curve'], 
             color='#9B59B6', linewidth=2.5, label=f'PR (AUC = {metrics["PR_AUC"]:.3f})')
    plt.fill_between(metrics['Recall_curve'], metrics['Precision_curve'], alpha=0.3, color='#9B59B6')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (30s, 개별 윈도우)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    save_current_figure(directory, stem, "4_PRCurve")

    # --- 5. 재구성 오차 분포 (새 윈도우, Figure 6) ---
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.6, color='#3498DB', edgecolor='black',
             label=f'정상 (n={len(normal_errors)})', density=True)
    plt.hist(abnormal_errors, bins=50, alpha=0.6, color='#E74C3C', edgecolor='black',
             label=f'비정상 (n={len(abnormal_errors)})', density=True)
    plt.axvline(metrics['Threshold'], color='green', linestyle='--', linewidth=3,
                label=f"임계값 = {metrics['Threshold']:.6f}")
    plt.xlabel('재구성 오차 (MSE)', fontsize=12)
    plt.ylabel('밀도', fontsize=12)
    plt.title('정상 vs 비정상 재구성 오차 분포 (30s, 개별 윈도우)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    save_current_figure(directory, stem, "5_ErrorDistribution")
    
    # --- 3. [수정됨] 전체 대시보드 저장 ---
    if directory:
        try:
            # 전체 대시보드(Figure 1) 저장
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"--- ✅ (전체) {p.name} 저장 완료 ---")
        except Exception as e:
            print(f"--- ⚠️ (전체) {p.name} 저장 실패: {e} ---")

    # --- 4. 모든 윈도우 띄우기 ---
    plt.show()



def print_performance_report(metrics):
    """성능 평가 리포트"""
    print("\n" + "="*80)
    print("📋 식약처 체외진단의료기기 성능 평가 리포트 (30초 모델)")
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
    NORMAL_HDF5_FILE = r'C:\Users\jerom\Downloads\model\dataset_30s.h5'
    ABNORMAL_HDF5_FILE = r'C:\Users\jerom\Downloads\model\30sec_test_data.h5'
    MODEL_SAVE_PATH = r'C:\Users\jerom\Downloads\model\30sec\model_test.pth'
    OUTPUT_DIR = r'C:\Users\jerom\Downloads\model\performance\30sec'
    
    # 하이퍼파라미터
    BATCH_SIZE = 64  # 30초는 데이터가 크므로 배치 크기 감소
    EVAL_BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    LATENT_DIM = 128
    LOAD_TO_MEMORY = True
    THRESHOLD_PERCENTILE = 95
    
    # 데이터 분할 비율
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.15
    TEST_RATIO = 0.10
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("🏥 심장 이상징후 탐지 (30초) - 식약처 가이드라인 기반 성능 평가")
    print("="*80)
    print(f"📁 정상 데이터: {NORMAL_HDF5_FILE}")
    print(f"📁 비정상 데이터: {ABNORMAL_HDF5_FILE}")
    print(f"🖥️  디바이스: {DEVICE}")
    print("="*80)
    
    # 파일 존재 확인
    if not Path(ABNORMAL_HDF5_FILE).exists():
        print(f"\n❌ 비정상 데이터 파일이 없습니다: {ABNORMAL_HDF5_FILE}")
        exit(1)
    
    # 학습
    print("\n[1단계] 모델 준비")
    print("-" * 80)
    
    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        NORMAL_HDF5_FILE, train_ratio=0.75, val_ratio=0.15, load_to_memory=LOAD_TO_MEMORY
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = CNNGRUAutoencoder30s(input_channels=2, sequence_length=7680, latent_dim=LATENT_DIM)
    
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
    
    print("\n🎯 학습 데이터로 임계값 설정")
    evaluator.fit_threshold(train_loader, device=DEVICE)
    
    print("\n📊 정상 데이터 테스트 세트 평가")
    normal_errors, normal_labels, normal_preds = evaluator.evaluate_dataset(
        test_loader, true_label=0, dataset_name="정상 (Test)", device=DEVICE
    )
    
    # 비정상 데이터 평가
    abnormal_dataset = load_dataset_auto(ABNORMAL_HDF5_FILE, load_to_memory=LOAD_TO_MEMORY)
    abnormal_loader = DataLoader(abnormal_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    abnormal_errors, abnormal_labels, abnormal_preds = evaluator.evaluate_dataset(
        abnormal_loader, true_label=1, dataset_name="비정상", device=DEVICE
    )
    
    # 전체 결합
    all_errors = np.concatenate([normal_errors, abnormal_errors])
    all_labels = np.concatenate([normal_labels, abnormal_labels])
    all_preds = np.concatenate([normal_preds, abnormal_preds])
    
    # 성능 지표 계산
    metrics = evaluator.calculate_medical_metrics(all_labels, all_preds, all_errors)
    
    # 결과 출력 및 시각화
    print_performance_report(metrics)
    plot_medical_performance_dashboard(metrics, normal_errors, abnormal_errors,
                                       save_path=f'{OUTPUT_DIR}/medical_performance_dashboard_30s.png')
    
    print("\n✅ 평가 완료!")

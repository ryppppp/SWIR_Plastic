import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# 配置参数
# ----------------------------
TRAIN_DIR = "./dataset/Multispectral/train"
TEST_DIR = "./dataset/Multispectral/test"       
PATCH_SIZE = 5
BANDS = ['1140','1200','1245','1310']
BAND_COUNT = len(BANDS)
LABEL_MAP = {
    # 0: "ABS",
    1: "PA",
    2: "PET",
    3: "PMMA",
    4: "PP",
    5: "PS",
    6: "PVC",
    7: "PC"
}

BATCH_SIZE = 512
NUM_WORKERS = 4
LR = 5e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
TEST_SPLIT = 0.8
PLOTS_DIR = "./sort/results"
BEST_MODEL_PATH = "./sort/results/best_model.pth"

SPECTRAL_ONLY = True  # 仅用光谱分支

os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# 数据读取与预处理
# ----------------------------
def load_csvs_to_images(csv_dir, band_cols=BANDS):
    """
    读取目录中的所有 CSV，
    返回 list of dict: [{'image': HxWxC numpy array, 'label': HxW numpy array}, ...]
    CSV columns must include x,y, band_cols..., label
    """
    result = []
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if len(csv_paths) == 0:
        raise ValueError(f"No csv found in {csv_dir}")
    for p in csv_paths:
        df = pd.read_csv(p)
        # ensure ints
        xs = df['x'].astype(int).values
        ys = df['y'].astype(int).values
        max_x = xs.max()
        max_y = ys.max()
        H = max_y + 1
        W = max_x + 1
        # init arrays
        img = np.zeros((H, W, len(band_cols)), dtype=np.float32)
        labels = np.full((H, W), fill_value=-1, dtype=np.int64)
        for _, row in df.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            # fill bands in order
            bands = [row[c] for c in band_cols]
            img[y, x, :] = np.array(bands, dtype=np.float32)
            labels[y, x] = int(row['label'])
        result.append({'image': img, 'label': labels, 'path': p})
    return result

def snv(vec: np.ndarray) -> np.ndarray:
    """
    标准正态变量转换（SNV）：对单个光谱向量做去均值/除以标准差。
    输入 vec: shape=(C,)
    返回: shape=(C,)
    """
    m = vec.mean()
    s = vec.std()
    if s < 1e-6:
        s = 1e-6
    return (vec - m) / s

def compute_spectral_features(v):
    """
    输入: v = [b1,b2,b3,b4]  shape=(4,)
    输出: 增强后的特征向量  shape=(17,)
    """
    b1, b2, b3, b4 = v
    
    # -------- 基础比值特征 --------
    ratio1 = b2 / (b1 + 1e-6)
    ratio2 = b3 / (b2 + 1e-6)
    ratio3 = b4 / (b3 + 1e-6)
    
    # -------- 光谱斜率 --------
    slope12 = b2 - b1
    slope23 = b3 - b2
    slope34 = b4 - b3
    
    # -------- 二阶差分曲率 --------
    curv1 = b3 - 2*b2 + b1
    curv2 = b4 - 2*b3 + b2
    
    # -------- 吸收深度 (模拟)--------
    # 局部线性插值: 深度 = 真实值 - 线性预测值
    depth_mid = b2 - ((b1 + b3) / 2)
    depth_mid2 = b3 - ((b2 + b4) / 2)
    
    # -------- Gaussian 简化参数 a1, a2 --------
    # 用两个手工特征替代复杂拟合
    a1 = (b2 + b3) / 2 - b1
    a2 = (b2 + b3) / 2 - b4
    
    # -------- Spectral Angle --------
    # 与全1向量的夹角
    u = np.array([1,1,1,1], dtype=np.float32)
    angle = np.arccos(
        np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-6)
    )
    
    feature_list = [
        # 原始光谱
        b1, b2, b3, b4,
        # 比值
        ratio1, ratio2, ratio3,
        # 斜率
        slope12, slope23, slope34,
        # 曲率
        curv1, curv2,
        # 吸收深度
        depth_mid, depth_mid2,
        # Gaussian a1,a2
        a1, a2,
        # spectral angle
        angle
    ]
    return np.array(feature_list, dtype=np.float32)

def build_dataset_from_images(image_list, patch_size=PATCH_SIZE):
    """
    将所有图像转换为像素样本列表：
    - spectral_vectors: N x C
    - patches: N x patch_size x patch_size x C
    - labels: N
    """
    half = patch_size // 2
    spectral_list = []
    patch_list = []
    label_list = []
    for item in image_list:
        img = item['image']  # H x W x C
        lab = item['label']  # H x W
        H, W, C = img.shape
        # pad image for patches
        pad_img = np.pad(img, ((half, half), (half, half), (0,0)), mode='reflect')
        for y in range(H):
            for x in range(W):
                label = lab[y, x]
                if label < 0:
                    continue    #跳过无标签像素
                raw = img[y, x, :].astype(np.float32)     # shape = (4,)
                snv_raw = snv(raw)                        # 对该像素的4维光谱做SNV
                feat = compute_spectral_features(snv_raw) # 基于SNV后的光谱派生17维特征
                # extract patch centered at (y,x) from padded image
                y0 = y
                x0 = x
                patch = pad_img[y0:y0+patch_size, x0:x0+patch_size, :].astype(np.float32)
                spectral_list.append(feat)
                patch_list.append(patch)
                label_list.append(int(label))
    spectral_np = np.stack(spectral_list, axis=0)   # N x C
    patch_np = np.stack(patch_list, axis=0)         # N x p x p x C
    unique_labels = sorted(set(label_list))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    labels_np = np.array([label_map[l] for l in label_list], dtype=np.int64)
    return spectral_np, patch_np, labels_np

# def build_train_test_from_images(image_list, patch_size=PATCH_SIZE, train_ratio=0.5, seed=RANDOM_SEED):
#     """
#     对每张图（每个CSV）内部按像素随机拆分：train_ratio（默认50%）为训练，其余为测试。
#     返回：
#     - X_spec_train, X_patch_train, y_train
#     - X_spec_test,  X_patch_test,  y_test
#     """
#     rng = np.random.default_rng(seed)
#     half = patch_size // 2

#     spec_train, patch_train, y_train = [], [], []
#     spec_test,  patch_test,  y_test  = [], [], []

#     for item in image_list:
#         img = item['image']  # H x W x C
#         lab = item['label']  # H x W
#         H, W, C = img.shape

#         # collect valid pixel coords for this image
#         coords = [(y, x) for y in range(H) for x in range(W) if lab[y, x] >= 0]
#         if len(coords) == 0:
#             continue
#         idx = np.arange(len(coords))
#         rng.shuffle(idx)
#         cut = int(len(idx) * train_ratio)
#         train_idx = idx[:cut]
#         test_idx  = idx[cut:]

#         # pad once per image
#         pad_img = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')

#         # helper to extract features + patch for a pixel
#         def extract(y, x):
#             raw = img[y, x, :].astype(np.float32)
#             snv_raw = snv(raw)
#             feat = compute_spectral_features(snv_raw)
#             patch = pad_img[y:y+patch_size, x:x+patch_size, :].astype(np.float32)
#             label = int(lab[y, x])
#             return feat, patch, label

#         # fill train split
#         for k in train_idx:
#             y0, x0 = coords[k]
#             feat, pch, lbl = extract(y0, x0)
#             spec_train.append(feat)
#             patch_train.append(pch)
#             y_train.append(lbl)

#         # fill test split
#         for k in test_idx:
#             y0, x0 = coords[k]
#             feat, pch, lbl = extract(y0, x0)
#             spec_test.append(feat)
#             patch_test.append(pch)
#             y_test.append(lbl)

#     # numpy stacks
#     X_spec_train = np.stack(spec_train, axis=0) if spec_train else np.empty((0, 17), dtype=np.float32)
#     X_patch_train = np.stack(patch_train, axis=0) if patch_train else np.empty((0, patch_size, patch_size, BAND_COUNT), dtype=np.float32)
#     y_train_np = np.array(y_train, dtype=np.int64) if y_train else np.empty((0,), dtype=np.int64)

#     X_spec_test  = np.stack(spec_test, axis=0) if spec_test else np.empty((0, 17), dtype=np.float32)
#     X_patch_test = np.stack(patch_test, axis=0) if patch_test else np.empty((0, patch_size, patch_size, BAND_COUNT), dtype=np.float32)
#     y_test_np  = np.array(y_test, dtype=np.int64) if y_test else np.empty((0,), dtype=np.int64)

#     # 将标签映射到紧凑的0..K-1（确保训练/测试一致）
#     unique_labels = sorted(set(y_train_np.tolist() + y_test_np.tolist()))
#     label_map = {lab: i for i, lab in enumerate(unique_labels)}
#     y_train_np = np.array([label_map[l] for l in y_train_np], dtype=np.int64)
#     y_test_np  = np.array([label_map[l] for l in y_test_np], dtype=np.int64)

#     return X_spec_train, X_patch_train, y_train_np, X_spec_test, X_patch_test, y_test_np

def bootstrap_train_test_from_images(image_list, patch_size=PATCH_SIZE, seed=RANDOM_SEED):
    """
    对每张图像（每个CSV文件）执行 Bootstrap 采样：
    - 训练集：对图内像素进行“有放回”采样 N 次
    - 测试集：Out-Of-Bag(OOB)，即所有没有被采样到的像素
    """
    rng = np.random.default_rng(seed)
    half = patch_size // 2

    spec_train, patch_train, y_train = [], [], []
    spec_test, patch_test, y_test = [], [], []

    for item in image_list:
        img = item['image']
        lab = item['label']
        H, W, C = img.shape

        # 收集所有有效像素坐标
        coords = [(y, x) for y in range(H) for x in range(W) if lab[y, x] >= 0]
        N = len(coords)
        if N == 0:
            continue

        # ---------- Bootstrap 有放回抽样 ----------
        # 训练集：N 次抽样
        train_indices = rng.integers(low=0, high=N, size=N)

        # OOB 测试集：未出现在 train_indices 中的样本
        in_train = set(train_indices.tolist())
        all_idx = set(range(N))
        test_indices = list(all_idx - in_train)  # OOB

        # pad 图像一次
        pad_img = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')

        # 抽取方法
        def extract(y, x):
            raw = img[y, x, :].astype(np.float32)
            snv_raw = snv(raw)
            feat = compute_spectral_features(snv_raw)
            patch = pad_img[y:y+patch_size, x:x+patch_size, :].astype(np.float32)
            label = int(lab[y, x])
            return feat, patch, label

        # 构建训练集
        for k in train_indices:
            y0, x0 = coords[k]
            feat, pch, lbl = extract(y0, x0)
            spec_train.append(feat)
            patch_train.append(pch)
            y_train.append(lbl)

        # 构建 OOB 测试集
        for k in test_indices:
            y0, x0 = coords[k]
            feat, pch, lbl = extract(y0, x0)
            spec_test.append(feat)
            patch_test.append(pch)
            y_test.append(lbl)

    # stack
    X_spec_train = np.stack(spec_train, axis=0)
    X_patch_train = np.stack(patch_train, axis=0)
    y_train_np = np.array(y_train, dtype=np.int64)

    X_spec_test = np.stack(spec_test, axis=0)
    X_patch_test = np.stack(patch_test, axis=0)
    y_test_np = np.array(y_test, dtype=np.int64)

    # label 映射
    unique_labels = sorted(set(y_train_np.tolist() + y_test_np.tolist()))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y_train_np = np.array([label_map[l] for l in y_train_np], dtype=np.int64)
    y_test_np = np.array([label_map[l] for l in y_test_np], dtype=np.int64)

    return X_spec_train, X_patch_train, y_train_np, X_spec_test, X_patch_test, y_test_np

# ----------------------------
# Dataset 类
# ----------------------------
class TwoBranchDataset(Dataset):
    def __init__(self, spectral_np, patch_np, labels_np, spectral_scaler=None, with_index=False):
        """
        spectral_np: N x C
        patch_np: N x p x p x C
        labels_np: N
        spectral_scaler: sklearn scaler to normalize spectral vectors (optional)
        """
        assert spectral_np.shape[0] == patch_np.shape[0] == labels_np.shape[0]
        self.spectral = spectral_np
        self.patch = patch_np
        self.labels = labels_np
        # flatten patch channels first for convenience in transform to torch (C,p,p)
        # but keep as numpy and convert in __getitem__
        self.spectral_scaler = spectral_scaler
        if spectral_scaler is not None:
            self.spectral = spectral_scaler.transform(self.spectral)
        self.with_index = with_index  

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        spec = self.spectral[idx]
        pat = self.patch[idx]
        spec_t = torch.from_numpy(spec).float()
        pat_t = torch.from_numpy(pat).permute(2,0,1).float()
        label_t = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.with_index:         # 返回索引      
            return spec_t, pat_t, label_t, torch.tensor(idx, dtype=torch.long)
        return spec_t, pat_t, label_t

# ----------------------------
# 模型：Two-Branch Fusion
# ----------------------------
# class SpectralBranch(nn.Module):
#     def __init__(self, in_dim=17, hidden=[64,32], dropout=0.2):
#         super().__init__()
#         layers = []
#         prev = in_dim
#         for h in hidden:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout))
#             prev = h
#         self.mlp = nn.Sequential(*layers)
#         self.out_dim = prev
#     def forward(self, x):
#         return self.mlp(x)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B x C x L
        b, c, l = x.size()
        w = x.mean(dim=2)            # 全局平均池化 → B x C
        w = self.fc(w).unsqueeze(2)  # → B x C x 1
        return x * w                 # 通道注意力
        

class SpectralBranch(nn.Module):
    """
    输入：17维光谱特征 → reshape 成 (B,1,17)
    结构：1D CNN → SE 注意力 → FC → 输出 32 维
    """
    def __init__(self, in_dim=17):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.se = SEBlock(32)

        # 最终 flatten → FC
        self.fc = nn.Sequential(
            nn.Linear(32 * in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        self.out_dim = 32

    def forward(self, x):
        # x: B x 17
        x = x.unsqueeze(1)     # → B x 1 x 17
        h = self.conv(x)       # → B x 32 x 17
        h = self.se(h)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        return h
    
class SpatialBranch(nn.Module):
    def __init__(self, in_channels=BAND_COUNT, base_channels=32, patch_size=PATCH_SIZE):
        super().__init__()
        # small conv net that reduces spatial dims to 1x1 by conv+pool, then flatten
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
        )
        # compute resulting spatial size after two poolings
        s = patch_size
        s = (s + 1) // 2 if s%2==1 else s//2
        # safer compute by integer division:
        s = patch_size // 2 // 2
        if s < 1:
            s = 1
        self._out_feat = (base_channels*2) * s * s
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._out_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        # x: B x C x p x p
        h = self.conv(x)
        h = self.fc(h)
        return h



class TwoBranchNet(nn.Module):
    def __init__(self, num_classes, patch_size=PATCH_SIZE, band_count=BAND_COUNT):
        super().__init__()
        self.spectral_only = SPECTRAL_ONLY

        self.spectral_branch = SpectralBranch(in_dim=17)
        if not self.spectral_only:
            self.spatial_branch = SpatialBranch(in_channels=BAND_COUNT, base_channels=32, patch_size=patch_size)

        fusion_dim = self.spectral_branch.out_dim if self.spectral_only else (self.spectral_branch.out_dim + 128)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, spec, patch):
        s_feat = self.spectral_branch(spec)
        if self.spectral_only:
            fused = s_feat
        else:
            p_feat = self.spatial_branch(patch)
            fused = torch.cat([s_feat, p_feat], dim=1)
        out = self.classifier(fused)
        return out

# ----------------------------
# 训练与评估函数
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for spec, patch, labels in loader:
        spec = spec.to(device)
        patch = patch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(spec, patch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for spec, patch, labels in loader:
            spec = spec.to(device)
            patch = patch.to(device)
            labels = labels.to(device)
            outputs = model(spec, patch)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# 可视化混淆矩阵
def plot_confusion(cm, classes, normalize, title, save_path):
    if normalize:
        cm = cm.astype('float')
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# 测试集评估与可视化
def evaluate_on_test(model, test_loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []

    mis_idx, mis_true, mis_pred = [], [], []  # 新增

    with torch.no_grad():
        for batch in test_loader:
            # 兼容是否返回索引
            if len(batch) == 4:
                spec, patch, labels, idx = batch
            else:
                spec, patch, labels = batch
                idx = None

            spec = spec.to(device); patch = patch.to(device)
            outputs = model(spec, patch)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(labels.numpy())

            # 记录误分类
            if idx is not None:
                idx_np = idx.cpu().numpy()
                labels_np = labels.cpu().numpy()
                mis_mask = preds != labels_np
                if np.any(mis_mask):
                    mis_idx.extend(idx_np[mis_mask].tolist())
                    mis_true.extend(labels_np[mis_mask].tolist())
                    mis_pred.extend(preds[mis_mask].tolist())

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    test_acc = (y_true == y_pred).mean()

    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    target_names = [LABEL_MAP.get(int(l), str(l)) for l in labels_present]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    report = classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, digits=4)

    # 保存误分类CSV
    if len(mis_idx) > 0:
        df_mis = pd.DataFrame({
            'dataset_index': mis_idx,
            'true': mis_true,
            'pred': mis_pred,
        })
        df_mis['true_name'] = df_mis['true'].apply(lambda x: LABEL_MAP.get(int(x), str(int(x))))
        df_mis['pred_name'] = df_mis['pred'].apply(lambda x: LABEL_MAP.get(int(x), str(int(x))))
        mis_path = os.path.join(PLOTS_DIR, "misclassified_test_samples.csv")
        df_mis.to_csv(mis_path, index=False, encoding="utf-8")
        print(f"Saved {len(df_mis)} misclassified samples to: {mis_path}")

    report_path = os.path.join(PLOTS_DIR, "test_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n"); f.write(report)

    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    cm_norm_path = os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png")
    plot_confusion(cm, target_names, normalize=False, title="Confusion Matrix", save_path=cm_path)
    plot_confusion(cm, target_names, normalize=True, title="Confusion Matrix (Normalized)", save_path=cm_norm_path)
    return test_acc, report

def spectral_feature_names():
    return [
        'b1','b2','b3','b4',
        'ratio2/1','ratio3/2','ratio4/3',
        'slope12','slope23','slope34',
        'curv1','curv2',
        'depth_mid','depth_mid2',
        'gauss_a1','gauss_a2',
        'angle'
    ]

def permutation_feature_importance(model, loader, device, num_features=17):
    model.eval()
    names = spectral_feature_names()
    importances = np.zeros(num_features, dtype=np.float32)

    # 计算基线准确率
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for spec, patch, labels in loader:
            spec = spec.to(device)
            patch = patch.to(device)
            outputs = model(spec, patch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(labels.numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    baseline_acc = (y_true == y_pred).mean()

    # 逐特征置换
    rng = np.random.default_rng(123)
    for k in range(num_features):
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for spec, patch, labels in loader:
                    spec_perm = spec.clone()
                    # 置换该特征维度（打乱同一batch内该列）
                    col = spec_perm[:, k].cpu().numpy()
                    rng.shuffle(col)
                    spec_perm[:, k] = torch.from_numpy(col).to(spec_perm.dtype).to(device)

                    outputs = model(spec_perm.to(device), patch.to(device))
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    y_pred_all.append(preds)
                    y_true_all.append(labels.numpy())
            y_true_p = np.concatenate(y_true_all)
            y_pred_p = np.concatenate(y_pred_all)
            acc_p = (y_true_p == y_pred_p).mean()
            importances[k] = max(0.0, baseline_acc - acc_p)

    return names, importances, baseline_acc

def plot_feature_importance(names, importances, baseline_acc, save_path):
    idx = np.argsort(importances)[::-1]
    names_sorted = [names[i] for i in idx]
    imps_sorted = importances[idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(imps_sorted)), imps_sorted, color="#4C78A8")
    plt.xticks(range(len(imps_sorted)), names_sorted, rotation=45, ha='right')
    plt.ylabel('Accuracy drop')
    plt.title(f'Permutation Feature Importance (baseline acc={baseline_acc:.4f})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# 主流程：读取数据、split、训练、测试
# ----------------------------
def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 1) 合并两批 CSV
    print("Loading TRAIN CSVs from:", TRAIN_DIR)
    train_imgs = load_csvs_to_images(TRAIN_DIR, band_cols=BANDS)
    print("Loading TEST CSVs from:", TEST_DIR)
    test_imgs = load_csvs_to_images(TEST_DIR, band_cols=BANDS)
    all_images = train_imgs + test_imgs
    print(f"Total CSV files merged: {len(all_images)}")

    # 整体 Bootstrap：训练为有放回采样，测试为 OOB
    X_spec_train, X_patch_train, y_train, X_spec_test, X_patch_test, y_test = bootstrap_train_test_from_images(
        all_images, patch_size=PATCH_SIZE, seed=RANDOM_SEED
    )
    print("Bootstrap train samples:", X_spec_train.shape[0])
    print("Bootstrap OOB test samples:", X_spec_test.shape[0])
    print("Class distribution (train):", Counter(y_train.tolist()))
    print("Class distribution (OOB test):", Counter(y_test.tolist()))

    # 标准化仅在训练集拟合
    scaler = StandardScaler()
    X_spec_train = scaler.fit_transform(X_spec_train)
    X_spec_test = scaler.transform(X_spec_test)

    # 从训练集中再切出一小部分作为验证集（例如10%）
    X_spec_train2, X_spec_val, X_patch_train2, X_patch_val, y_train2, y_val = train_test_split(
        X_spec_train, X_patch_train, y_train,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
    X_spec_train = X_spec_train2
    X_patch_train = X_patch_train2
    y_train = y_train2

    # 构建数据集/加载器
    train_ds = TwoBranchDataset(X_spec_train, X_patch_train, y_train, spectral_scaler=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    val_ds = TwoBranchDataset(X_spec_val, X_patch_val, y_val, spectral_scaler=None)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    test_ds = TwoBranchDataset(X_spec_test, X_patch_test, y_test, spectral_scaler=None, with_index=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 建模与训练
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    print("Num classes:", num_classes)

    model = TwoBranchNet(num_classes=num_classes, patch_size=PATCH_SIZE, band_count=len(BANDS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            tag = " (save best)"
        else:
            tag = ""

        print(f"Epoch {epoch:02d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}{tag}")

    print("Training finished. Best val loss:", best_val_loss)

    # 4) 测试集评估与可视化（OOB）
    if os.path.isfile(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        test_acc, report = evaluate_on_test(model, test_loader, DEVICE, num_classes)
        print(f"Final Test Accuracy (OOB): {test_acc:.4f}")
        print(report)
    else:
        print(f"Best model not found at {BEST_MODEL_PATH}, skipped test evaluation.")

    # 特征重要性（在验证集上）
    names, imps, base_acc = permutation_feature_importance(model, val_loader, DEVICE, num_features=17)
    imp_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plot_feature_importance(names, imps, base_acc, imp_path)
    print(f"Saved feature importance to: {imp_path}")


if __name__ == "__main__":
    main()

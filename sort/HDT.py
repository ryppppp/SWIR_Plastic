import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"# 全局字体设置为 Times New Roman
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示问题

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report as cls_report, accuracy_score,roc_curve, auc


# ----------------------------
# 配置参数
# ----------------------------
TRAIN_DIR = "./dataset/Multispectral/train"
TEST_DIR = "./dataset/Multispectral/test"
PLOT_DIR = "./sort/results_hdt"
MODEL_DIR = "./sort/results_hdt"

BANDS = ['1140','1200','1245','1310']
LABEL_MAP = {
    1: "PA",
    2: "PET",
    3: "PMMA",
    4: "PP",
    5: "PS",
    6: "PVC",
    7: "PC",
    8: "PE"
}

os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_SEED = 42


# ============================================================
# 17 维特征增强
# ============================================================
def compute_spectral_features(v):
    b1, b2, b3, b4 = v.astype(np.float32)
    eps = 1e-8

    ratio1 = b2/(b1+eps)
    ratio2 = b3/(b2+eps)
    ratio3 = b4/(b3+eps)

    slope12 = b2-b1
    slope23 = b3-b2
    slope34 = b4-b3

    curv1 = b3 - 2*b2 + b1
    curv2 = b4 - 2*b3 + b2

    depth_mid  = b2 - (b1+b3)/2
    depth_mid2 = b3 - (b2+b4)/2

    a1 = (b2+b3)/2 - b1
    a2 = (b2+b3)/2 - b4

    u = np.ones(4, dtype=np.float32)
    denom = (np.linalg.norm(v) * np.linalg.norm(u) + eps)
    angle = np.arccos(np.clip(np.dot(v, u) / denom, -1+eps, 1-eps))

    return np.array([
        b1,b2,b3,b4,
        ratio1,ratio2,ratio3,
        slope12,slope23,slope34,
        curv1,curv2,
        depth_mid,depth_mid2,
        a1,a2,
        angle
    ], dtype=np.float32)


# ============================================================
# 加载 csv（train 集合并做增强）
# ============================================================
def load_train(csv_dir):
    csv_paths = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_raw, all_feat, all_y = [], [], []

    for p in csv_paths:
        df = pd.read_csv(p)
        base = os.path.basename(p)
        print(f"Reading {base}...")

        arr = df[BANDS].values.astype(np.float32)
        labels = df["label"].astype(int).values

        for i in range(len(arr)):
            raw = arr[i]
            y = labels[i]

            all_raw.append(raw)
            all_feat.append(compute_spectral_features(raw))
            all_y.append(y)

    print("Train total:", len(all_y))
    print("Label counts:", Counter(all_y))
    return np.array(all_raw), np.array(all_feat), np.array(all_y)


# ============================================================
# 测试集
# ============================================================
def load_csv_test(csv_dir):
    csv_paths = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_raw, all_feat, all_y = [], [], []

    for p in csv_paths:
        df = pd.read_csv(p)
        arr = df[BANDS].values.astype(np.float32)
        labels = df["label"].astype(int).values

        for i in range(len(arr)):
            all_raw.append(arr[i])
            all_feat.append(compute_spectral_features(arr[i]))
            all_y.append(labels[i])

    return np.array(all_raw), np.array(all_feat), np.array(all_y)


# ============================================================
# 绘制混淆矩阵
# ============================================================
def plot_confusion(cm, classes, save_path=None):
    cm = cm.astype('float')
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum==0] = 1
    cm = cm / row_sum

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.close(fig)


# ============================================================
# 分段式层级分类器训练
# ============================================================
def train_hierarchical(X_train, y_train):
    models = {}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    def train_rf(X, y, name):
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X, y)
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"[OK] Trained {name}")
        return clf

    # -----------------------------
    # Stage 1 — PP or PE vs others
    # -----------------------------
    y_stage1 = np.where(np.isin(y_train, [4,8]), 1, 0)  # PP=4, PE=8
    models["stage1"] = train_rf(Xs, y_stage1, "stage1_PP_PE_vs_others")

    # -----------------------------
    # Stage 2 — PVC vs others
    # -----------------------------
    y_stage2 = np.where(y_train == 6, 1, 0)  # PVC=6
    models["stage2"] = train_rf(Xs, y_stage2, "stage2_PVC_vs_others")

    # -----------------------------
    # Stage 3 — Aromatic vs Non-aromatic
    # -----------------------------
    aromatic = {1,3,5}  # PA, PMMA, PS
    y_stage3 = np.where(np.isin(y_train, list(aromatic)), 1, 0)
    models["stage3"] = train_rf(Xs, y_stage3, "stage3_aromatic_vs_nonaromatic")

    # -----------------------------
    # Final subtree classifiers
    # -----------------------------
    mask_pp_pe  = np.isin(y_train, [4,8])
    mask_pvc    = (y_train == 6)
    mask_aromatic = np.isin(y_train, list(aromatic))
    mask_nonaromatic = ~(mask_pp_pe | mask_pvc | mask_aromatic)

    # PP vs PE
    models["pp_pe_final"] = train_rf(Xs[mask_pp_pe], y_train[mask_pp_pe], "final_PP_vs_PE")
    # PVC
    models["pvc_final"] = train_rf(Xs[mask_pvc], y_train[mask_pvc], "final_PVC")
    # Aromatic subset
    models["aromatic_final"] = train_rf(Xs[mask_aromatic], y_train[mask_aromatic], "final_aromatic")
    # Non-aromatic subset
    models["nonaromatic_final"] = train_rf(Xs[mask_nonaromatic], y_train[mask_nonaromatic], "final_nonaromatic")

    return models, scaler

# ============================================================
# 推理
# ============================================================
def hierarchical_predict(models, scaler, X):
    Xs = scaler.transform(X)
    n = Xs.shape[0]
    pred = np.zeros(n, dtype=int)

    # Stage 1: PP/PE vs Others
    s1 = models["stage1"].predict(Xs)
    pp_pe_idx = np.where(s1 == 1)[0]
    non_idx   = np.where(s1 == 0)[0]

    if len(pp_pe_idx) > 0:
        pred[pp_pe_idx] = models["pp_pe_final"].predict(Xs[pp_pe_idx])

    # Stage 2: PVC vs Others
    if len(non_idx) > 0:
        s2 = models["stage2"].predict(Xs[non_idx])
        pvc_idx = non_idx[np.where(s2 == 1)[0]]
        rest_idx = non_idx[np.where(s2 == 0)[0]]

        if len(pvc_idx) > 0:
            pred[pvc_idx] = models["pvc_final"].predict(Xs[pvc_idx])

        # Stage 3: Aromatic vs Non-aromatic
        if len(rest_idx) > 0:
            s3 = models["stage3"].predict(Xs[rest_idx])
            ar_idx = rest_idx[np.where(s3 == 1)[0]]
            nonar_idx = rest_idx[np.where(s3 == 0)[0]]

            if len(ar_idx) > 0:
                pred[ar_idx] = models["aromatic_final"].predict(Xs[ar_idx])
            if len(nonar_idx) > 0:
                pred[nonar_idx] = models["nonaromatic_final"].predict(Xs[nonar_idx])

    return pred

# ============================================================
# 可视化函数
# ============================================================
def feature_names():
    return [
        "b1", "b2", "b3", "b4",
        "ratio2/1", "ratio3/2", "ratio4/3",
        "slope12", "slope23", "slope34",
        "curv1", "curv2",
        "depth_mid", "depth_mid2",
        "a1", "a2",
        "angle"
    ]

def plot_feature_importance(importance_dict, names, save_path):
    """
    importance_dict: {label: importances(np.array shape=(n_features,))}
    names: feature names
    """
    labels = list(importance_dict.keys())
    num_models = len(labels)
    num_features = len(names)

    x = np.arange(num_features)
    width = 0.8 / num_models  # 三组并排柱子

    plt.figure(figsize=(8, 4))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for i, lbl in enumerate(labels):
        imps = np.array(importance_dict[lbl])
        plt.bar(
            x + (i - (num_models - 1) / 2) * width,
            imps,
            width,
            label=lbl,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.4,
        )

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("RandomForest Feature Importances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_per_class_bar(metrics_dict, metric_name, save_path):
    """
    metrics_dict: {class_name: value}
    metric_name: 'Accuracy' / 'Precision' / 'Recall' / 'F1-score' 等
    """
    classes = list(metrics_dict.keys())
    values = [metrics_dict[c] for c in classes]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(classes))
    plt.bar(x, values, color="#4C78A8", edgecolor="black")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.ylim(0, 1.0)
    plt.title(f"{metric_name} per Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_multiclass_roc(y_true, y_score, classes, save_path):
    """
    y_true: 1D array of true labels (integers)
    y_score: 2D array of shape (n_samples, n_classes) -> 预测概率
    classes: list of label values (e.g., [1,2,3,...])
    """
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 6))
    # 每个类别
    for i, c in enumerate(classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1.2,
            label=f"{LABEL_MAP[c]} (AUC={roc_auc[i]:.2f})",
        )
    # micro
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        color="black",
        lw=1.5,
        linestyle="--",
        label=f"micro-average (AUC={roc_auc['micro']:.2f})",
    )

    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# CSV 多光谱伪彩图可视化 + 分类结果标注
def visualize_csv_multispectral(csv_file, models, scaler, label_map, save_path=None):

    df = pd.read_csv(csv_file)

    xs, ys = df["x"].values, df["y"].values
    bands = df[BANDS].values.astype(np.float32) 
    labels_true = df["label"].values.astype(int)

    # 获取图像尺寸
    H = int(ys.max() + 1)
    W = int(xs.max() + 1)

    image_cube = np.zeros((H, W, len(BANDS)), dtype=np.float32)
    label_img = np.zeros((H, W), dtype=int)

    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if y < H and x < W:  
            image_cube[y, x, :] = bands[i]
            label_img[y, x] = labels_true[i]

    # RGB伪彩色显示（b3->R, b2->G, b1->B）
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)
    rgb_img[..., 0] = image_cube[..., 2]
    rgb_img[..., 1] = image_cube[..., 1]
    rgb_img[..., 2] = image_cube[..., 0]
    rgb_img -= rgb_img.min()
    rgb_img /= (rgb_img.max()+1e-8)

    # 分类预测
    bands_flat = image_cube.reshape(-1, len(BANDS))  # (N, 4)
    feat_flat = np.array([compute_spectral_features(v) for v in bands_flat], dtype=np.float32)
    pred_flat = hierarchical_predict(models, scaler, feat_flat)
    pred_img = pred_flat.reshape(H, W)

    # 分类颜色映射
    num_classes = len(label_map)
    cmap = plt.get_cmap("tab10", num_classes)
    pred_rgb = cmap(pred_img % 10)[..., :3]

    # 可视化
    valid_mask = (label_img > 0)  # 背景为 0，不参与评估
    # 计算像素级准确率（仅在有真实标签的像素上）
    valid_mask = (label_img > 0)
    if valid_mask.sum() > 0:
        accuracy = (pred_img[valid_mask] == label_img[valid_mask]).mean()
    else:
        accuracy = np.nan

    # 每张图上标注类别名称字符串
    unique_labels = np.unique(label_img[valid_mask])
    class_names = ", ".join([label_map[int(c)] for c in unique_labels if c in label_map])

    # 正确/错误掩膜（仅在有标签的像素上）
    correct_mask = (pred_img == label_img) & valid_mask
    error_mask = (pred_img != label_img) & valid_mask

    # 可视化：4 行 1 列竖排
    fig, axes = plt.subplots(4, 1, figsize=(6, 16))

    # 1) 原始伪彩图
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Multispectral Pseudo-color\nClasses: {class_names}", fontsize=16)
    axes[0].axis("off")

    # 2) 纯分类结果
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Classification Result", fontsize=16)
    axes[1].axis("off")

    # 3) 叠加 + 准确率
    axes[2].imshow(rgb_img)
    axes[2].imshow(pred_rgb, alpha=0.5)
    axes[2].set_title(f"Overlay (Accuracy={accuracy:.2%})\nClasses: {class_names}", fontsize=16)
    axes[2].axis("off")

    # 4) 对错掩膜：绿色=预测正确，红色=预测错误
    mask_img = np.zeros((H, W, 3), dtype=np.float32)
    mask_img[correct_mask] = [0.0, 0.8, 0.0]  # 绿
    mask_img[error_mask] = [0.9, 0.0, 0.0]    # 红

    axes[3].imshow(mask_img)
    axes[3].set_title("Correct (green) vs Error (red)", fontsize=16)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved visualization for {os.path.basename(csv_file)}, accuracy={accuracy:.2%}")
    return accuracy


# ============================================================
# 主流程
# ============================================================
print("=== Loading ALL data (TRAIN_DIR + TEST_DIR) ===")
Xtrain_raw_a, Xtrain_feat_a, ytrain_a = load_train(TRAIN_DIR)
Xtrain_raw_b, Xtrain_feat_b, ytrain_b = load_train(TEST_DIR)

X_all_feat = np.concatenate([Xtrain_feat_a, Xtrain_feat_b], axis=0)
y_all = np.concatenate([ytrain_a, ytrain_b], axis=0)
print("All samples:", X_all_feat.shape[0])
print("Label distribution (all):", Counter(y_all.tolist()))

print("\n=== Bootstrap sampling (with replacement) on ALL data ===")
N = X_all_feat.shape[0]
rng = np.random.default_rng(RANDOM_SEED)

bootstrap_idx = rng.integers(0, N, size=N)
oob_mask = np.ones(N, dtype=bool)
oob_mask[bootstrap_idx] = False  # OOB = 未抽到

X_train_boot = X_all_feat[bootstrap_idx]
y_train_boot = y_all[bootstrap_idx]

X_oob = X_all_feat[oob_mask]
y_oob = y_all[oob_mask]

print("Bootstrap train size:", X_train_boot.shape[0])
print("OOB size:", X_oob.shape[0])
print("Train label distribution (bootstrap):", Counter(y_train_boot.tolist()))
print("OOB label distribution:", Counter(y_oob.tolist()))

# 再从 bootstrap 训练集中划分一部分做验证（20%）
VAL_RATIO = 0.2
X_val, X_test, y_val, y_test = train_test_split(
    X_oob, y_oob, test_size=(1 - VAL_RATIO),
    random_state=RANDOM_SEED,
    stratify=y_oob
)
print("Val/Test split sizes (from OOB):", X_val.shape[0], X_test.shape[0])
print("Val label distribution:", Counter(y_val.tolist()))
print("Test label distribution:", Counter(y_test.tolist()))

# 可视化训练 / OOB / Val / Test 的类别分布
train_counts = Counter(y_train_boot.tolist())
oob_counts   = Counter(y_oob.tolist())
val_counts   = Counter(y_val.tolist())
test_counts  = Counter(y_test.tolist())

ordered_labels = sorted(LABEL_MAP.keys())
class_names = [LABEL_MAP[i] for i in ordered_labels]

def _counter_to_list(counter_obj):
    return [counter_obj.get(i, 0) for i in ordered_labels]

train_vals = _counter_to_list(train_counts)
oob_vals   = _counter_to_list(oob_counts)
val_vals   = _counter_to_list(val_counts)
test_vals  = _counter_to_list(test_counts)

x = np.arange(len(ordered_labels))
width = 0.2

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# 1) 绝对数量
axes[0].bar(x - 1.5*width, train_vals, width, label="Train (bootstrap)")
axes[0].bar(x - 0.5*width, oob_vals,   width, label="OOB")
axes[0].bar(x + 0.5*width, val_vals,   width, label="Val")
axes[0].bar(x + 1.5*width, test_vals,  width, label="Test")
axes[0].set_ylabel("Sample Count")
axes[0].set_title("Class Distribution (Counts)")
axes[0].legend(fontsize=8)

# 2) 归一化比例
def _normalize(vals):
    s = sum(vals)
    return [v / s if s > 0 else 0 for v in vals]

train_ratios = _normalize(train_vals)
oob_ratios   = _normalize(oob_vals)
val_ratios   = _normalize(val_vals)
test_ratios  = _normalize(test_vals)

axes[1].bar(x - 1.5*width, train_ratios, width, label="Train (bootstrap)")
axes[1].bar(x - 0.5*width, oob_ratios,   width, label="OOB")
axes[1].bar(x + 0.5*width, val_ratios,   width, label="Val")
axes[1].bar(x + 1.5*width, test_ratios,  width, label="Test")
axes[1].set_ylabel("Proportion")
axes[1].set_title("Class Distribution (Proportions)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_names, rotation=45, ha="right")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "dataset_class_distribution.png"), dpi=160)
plt.close(fig)

print("Saved dataset class distribution plot.")


print("=== Training hierarchical classifier (train only) ===")
models, scaler = train_hierarchical(X_train_boot, y_train_boot)

# 验证 
print("\n=== Evaluating on validation set ===")
y_val_pred = hierarchical_predict(models, scaler, X_val)
print("Val accuracy:", accuracy_score(y_val, y_val_pred))
print(cls_report(
    y_val, y_val_pred,
    labels=sorted(LABEL_MAP.keys()),
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
    zero_division=0
))
cm_val = confusion_matrix(y_val, y_val_pred, labels=sorted(LABEL_MAP.keys()))
plot_confusion(cm_val, [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())], os.path.join(PLOT_DIR, "val_confusion.png"))
print("Saved validation confusion matrix.")

# 测试 
print("\n=== Evaluating on test set ===")
y_test_pred = hierarchical_predict(models, scaler, X_test)

print("Test accuracy:", accuracy_score(y_test, y_test_pred))

report_dict = cls_report(
    y_test,
    y_test_pred,
    labels=sorted(LABEL_MAP.keys()),
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
    zero_division=0,
    output_dict=True
)
print(pd.DataFrame(report_dict).T)

cm_test = confusion_matrix(y_test, y_test_pred, labels=sorted(LABEL_MAP.keys()))
plot_confusion(
    cm_test,
    [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
    os.path.join(PLOT_DIR, "test_confusion.png"),
)
print("Saved test confusion matrix.")

# 按类别绘制 Precision / Recall / F1 柱状图
per_class_precision = {}
per_class_recall = {}
per_class_f1 = {}

for cls_id in sorted(LABEL_MAP.keys()):
    name = LABEL_MAP[cls_id]
    if name in report_dict:
        per_class_precision[name] = report_dict[name]["precision"]
        per_class_recall[name] = report_dict[name]["recall"]
        per_class_f1[name] = report_dict[name]["f1-score"]

plot_per_class_bar(
    per_class_precision,
    "Precision",
    os.path.join(PLOT_DIR, "test_precision_per_class.png"),
)
plot_per_class_bar(
    per_class_recall,
    "Recall",
    os.path.join(PLOT_DIR, "test_recall_per_class.png"),
)
plot_per_class_bar(
    per_class_f1,
    "F1-score",
    os.path.join(PLOT_DIR, "test_f1_per_class.png"),
)
print("Saved per-class bar plots for precision/recall/F1.")

# 可视化特征重要性
fnames = feature_names()
importance_dict = {
    "PP vs PE": models["pp_pe_final"].feature_importances_,
    "Aromatic subset": models["aromatic_final"].feature_importances_,
    "Non-aromatic subset": models["nonaromatic_final"].feature_importances_,
}
plot_feature_importance(
    importance_dict,
    fnames,
    os.path.join(PLOT_DIR, "hdt_feature_importance_combined.png"),
)


csv_paths = glob.glob(os.path.join(TEST_DIR, "*.csv"))
accuracies = []
for csv_file in csv_paths:
    fname = os.path.basename(csv_file).replace(".csv","")
    save_path = os.path.join(PLOT_DIR, f"{fname}_pseudo_class.png")
    acc = visualize_csv_multispectral(csv_file, models, scaler, LABEL_MAP, save_path)
    accuracies.append(acc)
print(f"Average classification accuracy over {len(csv_paths)} files: {np.mean(accuracies):.2%}")

# 多类别 ROC 曲线 (one-vs-rest)
rf_for_roc = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
)
rf_for_roc.fit(X_train_boot, y_train_boot)
y_test_proba = rf_for_roc.predict_proba(X_test)

roc_save_path = os.path.join(PLOT_DIR, "test_multiclass_roc.png")
plot_multiclass_roc(
    y_test,
    y_test_proba,
    classes=sorted(LABEL_MAP.keys()),
    save_path=roc_save_path,
)
print("Saved multi-class ROC curves.")






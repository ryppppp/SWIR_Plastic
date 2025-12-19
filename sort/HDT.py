import os 
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------
# 配置参数
# ----------------------------
TRAIN_DIR = "./dataset/Multispectral/train"
TEST_DIR = "./dataset/Multispectral/test"
PLOT_DIR = "./sort/results_hdt"
MODEL_DIR = "./sort/results_hdt"

BANDS = ['1140','1200','1245','1310']
LABEL_MAP = {
    # 0: "ABS",
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

def plot_feature_importance(importances, names, save_path):
    idx = np.argsort(importances)[::-1]
    names_sorted = [names[i] for i in idx]
    imps_sorted = importances[idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(imps_sorted)), imps_sorted, color="#4C78A8")
    plt.xticks(range(len(imps_sorted)), names_sorted, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("RandomForest Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

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

print("=== Training hierarchical classifier (train only) ===")
models, scaler = train_hierarchical(X_train_boot, y_train_boot)

# ====== 验证 ======
print("\n=== Evaluating on validation set ===")
y_val_pred = hierarchical_predict(models, scaler, X_val)
print("Val accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(
    y_val, y_val_pred,
    labels=sorted(LABEL_MAP.keys()),
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
    zero_division=0
))
cm_val = confusion_matrix(y_val, y_val_pred, labels=sorted(LABEL_MAP.keys()))
plot_confusion(cm_val, [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())], os.path.join(PLOT_DIR, "val_confusion.png"))
print("Saved validation confusion matrix.")

# ====== 测试 ======
print("\n=== Evaluating on test set ===")
y_test_pred = hierarchical_predict(models, scaler, X_test)
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(
    y_test, y_test_pred,
    labels=sorted(LABEL_MAP.keys()),
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
    zero_division=0
))
cm_test = confusion_matrix(y_test, y_test_pred, labels=sorted(LABEL_MAP.keys()))
plot_confusion(cm_test, [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())], os.path.join(PLOT_DIR, "test_confusion.png"))
print("Saved test confusion matrix.")


# 可视化特征重要性
fnames = feature_names()

for name in ["pp_pe_final", "pvc_final", "aromatic_final", "nonaromatic_final"]:
    imp = models[name].feature_importances_
    plot_feature_importance(imp, fnames, os.path.join(PLOT_DIR, f"hdt_feature_importance_{name}.png"))




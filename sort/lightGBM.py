import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ----------------------------
# 配置参数
# ----------------------------
TRAIN_DIR = "./dataset/Multispectral/train"
TEST_DIR = "./dataset/Multispectral/test"
PLOTS_DIR = "./sort/results"
PATCH_SIZE = 5
BANDS = ['1140','1200','1245','1310']
LABEL_MAP = {
    0: "ABS",
    1: "PA",
    2: "PET",
    3: "PMMA",
    4: "PP",
    5: "PS",
    6: "PVC",
    7:"PC"
}
class_weights = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 0.2  # 最大类降权
}

RANDOM_SEED = 42
TEST_SPLIT = 0.8
USE_LIGHTGBM = True    # True=LightGBM, False=XGBoost

# ----------------------------
# 数据读取与预处理
# ----------------------------
def load_csvs_to_images(csv_dir, band_cols=BANDS):
    result = []
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if len(csv_paths) == 0:
        raise ValueError(f"No csv found in {csv_dir}")

    for p in csv_paths:
        df = pd.read_csv(p)
        xs = df['x'].astype(int).values
        ys = df['y'].astype(int).values
        H = ys.max() + 1
        W = xs.max() + 1

        img = np.zeros((H, W, len(band_cols)), dtype=np.float32)
        labels = np.full((H, W), fill_value=-1, dtype=np.int64)

        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            img[y, x, :] = np.array([row[c] for c in band_cols], dtype=np.float32)
            labels[y, x] = int(row['label'])

        result.append({'image': img, 'label': labels, 'path': p})
    return result


def snv(vec):
    m = vec.mean()
    s = vec.std()
    if s < 1e-6:
        s = 1e-6
    return (vec - m) / s


def compute_spectral_features(v):
    b1, b2, b3, b4 = v

    ratio1 = b2 / (b1 + 1e-6)
    ratio2 = b3 / (b2 + 1e-6)
    ratio3 = b4 / (b3 + 1e-6)

    slope12 = b2 - b1
    slope23 = b3 - b2
    slope34 = b4 - b3

    curv1 = b3 - 2*b2 + b1
    curv2 = b4 - 2*b3 + b2

    depth_mid = b2 - ((b1 + b3) / 2)
    depth_mid2 = b3 - ((b2 + b4) / 2)

    a1 = (b2 + b3) / 2 - b1
    a2 = (b2 + b3) / 2 - b4

    u = np.array([1,1,1,1], dtype=np.float32)
    angle = np.arccos(
        np.dot(v, u) / (np.linalg.norm(v)*np.linalg.norm(u) + 1e-6)
    )

    return np.array([
        b1,b2,b3,b4,
        ratio1,ratio2,ratio3,
        slope12,slope23,slope34,
        curv1,curv2,
        depth_mid,depth_mid2,
        a1,a2,
        angle
    ], dtype=np.float32)


def build_dataset_from_images(image_list, patch_size=PATCH_SIZE):
    half = patch_size // 2
    spectral_list = []
    label_list = []

    for item in image_list:
        img = item['image']
        lab = item['label']
        H,W,C = img.shape

        pad_img = np.pad(img, ((half,half),(half,half),(0,0)), mode="reflect")

        for y in range(H):
            for x in range(W):
                label = lab[y,x]
                if label < 0:
                    continue

                raw = img[y,x,:].astype(np.float32)
                snv_raw = snv(raw)
                feat = compute_spectral_features(snv_raw)

                spectral_list.append(feat)
                label_list.append(int(label))

    spectral_np = np.stack(spectral_list, axis=0)
    unique_labels = sorted(set(label_list))
    label_map = {lab:i for i,lab in enumerate(unique_labels)}
    labels_np = np.array([label_map[l] for l in label_list], dtype=np.int64)

    return spectral_np, labels_np


# ----------------------------
# 训练 LightGBM / XGBoost
# ----------------------------
def train_tree_model(X_train, y_train, X_val, y_val, num_classes, class_weight_dict):
    if USE_LIGHTGBM:
        print("Using LightGBM...")
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            objective="multiclass",
            num_class=num_classes,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            min_data_in_leaf=5,
            class_weight=class_weight_dict
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="multi_logloss"
        )
        evals_result = getattr(model, "evals_result_", {})
        return model, evals_result
    else:
        print("Using XGBoost...")
        model = XGBClassifier(
            objective="multi:softmax",
            num_class=num_classes,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=RANDOM_SEED
        )
        sample_weight = np.array([class_weight_dict[int(c)] for c in y_train], dtype=np.float32)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            sample_weight=sample_weight
        )
        evals_result = {}
        try:
            evals_result = model.evals_result()
        except Exception:
            pass
        return model, evals_result

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def plot_confusion_matrices(y_true, y_pred, class_names, out_prefix):
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # counts
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax1, cmap="Blues", colorbar=True)
    ax1.set_title("Confusion Matrix (counts)")
    plt.tight_layout()
    fig1.savefig(f"{out_prefix}_cm_counts.png", dpi=200)
    plt.close(fig1)

    # normalized
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(ax=ax2, cmap="Blues", colorbar=True)
    ax2.set_title("Confusion Matrix (normalized)")
    plt.tight_layout()
    fig2.savefig(f"{out_prefix}_cm_normalized.png", dpi=200)
    plt.close(fig2)

def plot_loss_and_acc(evals_result, model_name, out_prefix):
    if not evals_result:
        return

    # 损失曲线（logloss / multi_logloss）
    plt.figure(figsize=(7, 5))
    found_loss = False
    for ds, metrics in evals_result.items():
        key = None
        for k in ("multi_logloss", "mlogloss", "logloss"):
            if k in metrics:
                key = k
                break
        if key is None:
            continue
        vals = metrics[key]
        plt.plot(range(1, len(vals)+1), vals, label=f"{ds} ({key})")
        found_loss = True
    if found_loss:
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_loss.png", dpi=200)
    plt.close()

    # 准确率曲线（1 - merror / error）
    plt.figure(figsize=(7, 5))
    found_acc = False
    for ds, metrics in evals_result.items():
        key = None
        for k in ("merror", "error"):
            if k in metrics:
                key = k
                break
        if key is None:
            continue
        errs = metrics[key]
        accs = [1.0 - e for e in errs]
        plt.plot(range(1, len(accs)+1), accs, label=f"{ds} Accuracy")
        found_acc = True
    if found_acc:
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_acc.png", dpi=200)
    plt.close()

def plot_output_hist(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(y_pred, bins=len(np.unique(y_true)), alpha=0.7, edgecolor="k")
    plt.xlabel("Predicted class")
    plt.ylabel("Count")
    plt.title("Model Output Histogram (predicted labels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------
# 主流程
# ----------------------------
def main():
    ensure_dir(PLOTS_DIR)
    print("Loading BOTH batches (train + test)...")
    ### 把两批数据视为同一批
    images_A = load_csvs_to_images(TRAIN_DIR)
    images_B = load_csvs_to_images(TEST_DIR)
    all_images = images_A + images_B

    print("Building FULL dataset...")
    X_all, y_all = build_dataset_from_images(all_images)
    print("Full feature shape:", X_all.shape)
    print("Full label distribution:", Counter(y_all))


    # Bootstrap 自助法划分
    print("\n=== Performing Bootstrap sampling (with replacement) ===")
    N = X_all.shape[0]
    rng = np.random.default_rng(42)

    ### 训练集 → 有放回抽样 
    bootstrap_idx = rng.integers(0, N, size=N)     # 允许重复
    oob_mask = np.ones(N, dtype=bool)
    oob_mask[bootstrap_idx] = False                # 未被抽到者 = OOB 测试集

    X_train = X_all[bootstrap_idx]
    y_train = y_all[bootstrap_idx]

    X_test = X_all[oob_mask]
    y_test = y_all[oob_mask]

    print("Bootstrap train size:", len(X_train))
    print("Bootstrap OOB (test) size:", len(X_test))
    print("OOB rate:", len(X_test)/N)

    # ---------------------------
    # 标准化（仅用训练集 fit）
    # ---------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 无验证集 → 改用训练集划分一部分
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )
    X_train = X_train2
    y_train = y_train2

    # ---------------------------
    # 类别权重（自动）
    # ---------------------------
    num_classes = len(np.unique(y_train))
    counts = np.bincount(y_train, minlength=num_classes)
    counts[counts == 0] = 1
    class_weight_vals = (counts.sum()) / (counts * num_classes)
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weight_vals)}
    print("Class weights:", class_weight_dict)

    # ---------------------------
    # Train model
    # ---------------------------
    model, evals_result = train_tree_model(X_train, y_train, X_val, y_val, num_classes, class_weight_dict)

    # ---------------------------
    # Evaluate
    # ---------------------------
    print("\nEvaluating on TRAIN...")
    y_pred_train = model.predict(X_train)
    train_report = classification_report(y_train, y_pred_train)
    print(train_report)
    save_text(os.path.join(PLOTS_DIR, "bootstrap_train_report.txt"), train_report)

    print("\nEvaluating on OOB TEST...")
    y_pred_test = model.predict(X_test)
    test_acc = (y_pred_test == y_test).mean()
    print("OOB Test Accuracy:", test_acc)
    test_report = classification_report(y_test, y_pred_test)
    print(test_report)
    save_text(os.path.join(PLOTS_DIR, "bootstrap_test_report.txt"),
              f"OOB Test Accuracy: {test_acc:.4f}\n\n" + test_report)

    # 类名（此处 y 已经是 0..K-1 编码）
    class_names = [LABEL_MAP.get(i, str(i)) for i in range(num_classes)]
    model_tag = "lgbm" if USE_LIGHTGBM else "xgb"
    prefix = os.path.join(PLOTS_DIR, f"bootstrap_{model_tag}")

    # 混淆矩阵
    plot_confusion_matrices(y_test, y_pred_test, class_names, prefix)

    # 模型输出直方图（预测标签分布）
    plot_output_hist(y_test, y_pred_test, prefix + "_pred_hist.png")

    # 训练/验证损失和准确率曲线
    plot_loss_and_acc(evals_result, model_tag.upper(), os.path.join(PLOTS_DIR, f"{model_tag}"))

if __name__ == "__main__":
    main()

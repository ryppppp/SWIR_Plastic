import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# =======================
# 1. 读取数据（目录下多 CSV，每个 CSV 是一张图像）
# =======================
train_dir = "./dataset/Multispectual/train"   
csv_paths = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
if len(csv_paths) == 0:
    raise FileNotFoundError(f"No CSV found in {train_dir}")

dfs = []
for p in csv_paths:
    df_i = pd.read_csv(p)
    # 基本列校验
    required_cols = {"x","y","1140","1200","1245","1310","label"}
    missing = required_cols - set(df_i.columns)
    if missing:
        raise ValueError(f"CSV {p} missing columns: {missing}")
    dfs.append(df_i)

df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱

# =======================
# 新：统计每类在四个波段的数值分布，并保存结果与图像
# =======================
bands = ["1140", "1200", "1245", "1310"]  # 与训练保持一致的顺序

print("\n每类在各波段的描述性统计：")
group_stats = df.groupby("label")[bands].describe().transpose()
print(group_stats)

os.makedirs("./sort/plots", exist_ok=True)
group_stats.to_csv("per_class_band_stats.csv", encoding="utf-8")

for band in bands:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y=band, data=df, order=sorted(df["label"].unique()))
    plt.xlabel("Label")
    plt.ylabel(f"{band} value")
    plt.title(f"Boxplot of {band} by label")
    plt.tight_layout()
    plt.savefig(os.path.join("./sort/plots", f"box_{band}.png"))
    plt.close()

for band in bands:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="label", y=band, data=df, order=sorted(df["label"].unique()), inner="quartile")
    plt.xlabel("Label")
    plt.ylabel(f"{band} value")
    plt.title(f"Violinplot of {band} by label")
    plt.tight_layout()
    plt.savefig(os.path.join("./sort/plots", f"violin_{band}.png"))
    plt.close()

# =======================
# 2. 特征与标签
# =======================
X = df[bands].values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print("类别映射：", dict(zip(class_names, le.transform(class_names))))

# =======================
# 3. 划分训练/测试集
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# =======================
# 4. 标准化
# =======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# 5. SVM + 网格搜索优化参数
# =======================
param_grid = {
    "C": [0.1, 1, 5, 10, 50],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "kernel": ["rbf"]
}

svm = SVC()
grid = GridSearchCV(
    svm, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print("最佳参数：", grid.best_params_)
print("训练集准确率（交叉验证均值）：", grid.best_score_)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)

# =======================
# 6. 分类报告
# =======================
print("\n分类性能报告：")
print(classification_report(y_test, y_pred, target_names=class_names))

# =======================
# 7. 绘制混淆矩阵
# =======================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (SVM)")
plt.tight_layout()
plt.savefig(os.path.join("./sort/plots", "svm_confusion_matrix.png"))
plt.close()

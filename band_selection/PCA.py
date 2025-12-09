import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r'D:\NJU\HW\Final\filter\standard\slice.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=185)

# ================= 线性PCA分析 =================
linear_pca = PCA(n_components=0.95)
X_train_linear = linear_pca.fit_transform(X_train)
X_test_linear = linear_pca.transform(X_test)

# 线性PCA累计方差
explained_var_linear = np.cumsum(linear_pca.explained_variance_ratio_)

# ================= 核PCA分析（RBF核）=================
kernel_pca = KernelPCA(n_components=10, kernel='rbf', gamma=0.01, fit_inverse_transform=True)
X_train_kernel = kernel_pca.fit_transform(X_train)
X_test_kernel = kernel_pca.transform(X_test)

# 计算核PCA伪方差（需手动计算）
# 注意：核PCA没有直接方差解释，此处计算特征值平方的标准化
eigenvalues = np.sort(np.linalg.svd(X_train_kernel, compute_uv=False))[::-1]
explained_var_kernel = np.cumsum(eigenvalues**2 / np.sum(eigenvalues**2))

# ================= 可视化对比 =================
plt.figure(figsize=(12, 6))

# 累计方差对比
plt.plot(explained_var_linear, 'b-o', label='Linear PCA')
plt.plot(explained_var_kernel[:len(explained_var_linear)], 'r-s', label='Kernel PCA (RBF)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA vs Kernel PCA: Variance Explained')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# ================= 修正后的可视化代码 =================

plt.figure(figsize=(15, 6))

# 线性PCA权重分布（波段级别）
plt.subplot(1, 2, 1)
plt.hist(np.abs(linear_pca.components_[0]), bins=30, color='blue', alpha=0.7)
plt.xlabel('Linear PCA Weight Value')
plt.ylabel('Band Count')
plt.title('Linear PCA: Component 1 Weight Distribution')
plt.grid(True)

# 核PCA得分分布（样本级别）
plt.subplot(1, 2, 2)
plt.hist(X_train_kernel[:, 0], bins=30, color='red', alpha=0.7)
plt.xlabel('Kernel PCA Score Value')
plt.ylabel('Sample Count')
plt.title('Kernel PCA: Component 1 Score Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# ================= 波段重要性对比 =================
def calculate_band_importance(pca_model, n_bands=42, is_kernel=False):
    """计算波段重要性（适配线性与核PCA）"""
    if is_kernel:
        # 核PCA需通过逆变换近似
        inv_components = pca_model.inverse_transform(np.eye(pca_model.n_components))
        importance = np.sum(np.abs(inv_components), axis=0)
    else:
        importance = np.sum(np.abs(pca_model.components_) *
                          pca_model.explained_variance_ratio_[:, np.newaxis], axis=0)
    return importance / np.max(importance)

# 计算重要性
linear_importance = calculate_band_importance(linear_pca)
kernel_importance = calculate_band_importance(kernel_pca, is_kernel=True)

# 可视化波段重要性对比
plt.figure(figsize=(15, 6))

# 线性PCA波段重要性
plt.subplot(1, 2, 1)
sorted_linear = np.argsort(linear_importance)[::-1]
plt.bar(range(42), linear_importance[sorted_linear])
plt.xticks(range(42), sorted_linear, rotation=90)
plt.xlabel('Band Index')
plt.ylabel('Importance')
plt.title('Linear PCA: Band Importance Ranking')

# 核PCA波段重要性
plt.subplot(1, 2, 2)
sorted_kernel = np.argsort(kernel_importance)[::-1]
plt.bar(range(42), kernel_importance[sorted_kernel])
plt.xticks(range(42), sorted_kernel, rotation=90)
plt.xlabel('Band Index')
plt.ylabel('Importance')
plt.title('Kernel PCA: Band Importance Ranking')

plt.tight_layout()
plt.show()

# ================= 分类性能对比 =================
def evaluate_model(X_train_pca, X_test_pca, y_train, y_test):
    """评估SVM分类性能"""
    svm = SVC(kernel='rbf', C=100, gamma='scale')
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    return accuracy_score(y_test, y_pred)

# 分类准确率对比
linear_acc = evaluate_model(X_train_linear, X_test_linear, y_train, y_test)
kernel_acc = evaluate_model(X_train_kernel, X_test_kernel, y_train, y_test)

print(f"\nClassification Accuracy:\n"
      f"  Linear PCA: {linear_acc*100:.2f}%\n"
      f"  Kernel PCA: {kernel_acc*100:.2f}%")

# ================= 筛选关键波段对比 =================
# 保留重要性前6的波段
top6_linear = sorted_linear[:6]
top6_kernel = sorted_kernel[:6]

print("\nTop 6 Bands Selected:")
print(f"  Linear PCA: {np.sort(top6_linear)}")
print(f"  Kernel PCA: {np.sort(top6_kernel)}")
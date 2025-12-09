import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem

# 加载数据
data = pd.read_csv(r'D:\NJU\HW\Final\filter\standard\slice.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据标准化（列标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ================= 增强版多模型RFE =================
def enhanced_rfe(X, y, n_select=6):
    models = {
        'Logistic': LogisticRegression(max_iter=2000),
        'SVM-Linear': SVC(kernel='linear'),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'PLS': PLSRegression(n_components=3)
    }

    results = {}
    n_features = X.shape[1]  # 原始特征数
    for name, model in models.items():
        try:
            estimator = clone(model)
            selector = RFE(estimator, n_features_to_select=n_select, step=1)
            selector.fit(X, y)
            support = selector.support_

            # 提取重要性并扩展到原始特征数
            if name == 'PLS':
                importance = np.abs(selector.estimator_.x_weights_[:, 0])
            elif isinstance(selector.estimator_, Pipeline):
                final_step = selector.estimator_.named_steps[list(selector.estimator_.named_steps.keys())[-1]]
                if hasattr(final_step, 'feature_importances_'):
                    importance = final_step.feature_importances_
                elif hasattr(final_step, 'coef_'):
                    coefs = final_step.coef_
                    importance = np.abs(coefs).mean(axis=0) if len(coefs.shape) > 1 else np.abs(coefs)
                else:
                    importance = np.zeros(n_select)  # 默认处理
            else:
                if hasattr(selector.estimator_, 'feature_importances_'):
                    importance = selector.estimator_.feature_importances_
                elif hasattr(selector.estimator_, 'coef_'):
                    coefs = selector.estimator_.coef_
                    importance = np.abs(coefs).mean(axis=0) if len(coefs.shape) > 1 else np.abs(coefs)
                else:
                    importance = np.zeros(n_select)  # 默认处理

            # 将重要性映射到原始特征位置
            importance_full = np.zeros(n_features)
            importance_full[support] = importance

            # 归一化处理
            importance_full = (importance_full - importance_full.min()) / (importance_full.max() - importance_full.min() + 1e-8)
            results[name] = {
                'selected': np.where(support)[0],
                'importance': importance_full,
                'ranking': selector.ranking_
            }
        except Exception as e:
            print(f"[Error] {name} failed: {str(e)}")

    return results

# ================= Kruskal-Wallis检验 =================
def kruskal_feature_ranking(X, y):
    """ Kruskal-Wallis特征排名 """
    n_bands = X.shape[1]
    h_values = np.zeros(n_bands)
    for band in range(n_bands):
        groups = [X[y == cls, band] for cls in np.unique(y)]
        h, _ = stats.kruskal(*groups)
        h_values[band] = h
    return h_values / h_values.max()


# ================= 可视化分析 =================
def visualize_results(results, kruskal_scores, n_select=6):
    plt.figure(figsize=(15, 10))

    # 1. 模型重要性对比
    plt.subplot(2, 2, 1)
    band_idx = np.arange(X.shape[1])
    for name, res in results.items():
        plt.plot(res['importance'], label=name)
    plt.plot(kruskal_scores, 'k--', label='Kruskal-Wallis')
    plt.xlabel('Band Index')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance Comparison')
    plt.legend()
    plt.grid(True)

    # 2. 波段选择分布
    plt.subplot(2, 2, 2)
    for i, (name, res) in enumerate(results.items()):
        plt.scatter(res['selected'], [i] * len(res['selected']), label=name, s=50)
    plt.scatter(np.argsort(kruskal_scores)[-n_select:], [len(results)] * n_select,
                marker='x', label='Kruskal')
    plt.yticks(range(len(results) + 1), list(results.keys()) + ['Kruskal'])
    plt.xlabel('Band Index')
    plt.title('Selected Bands Distribution')
    plt.grid(True)

    # 3. 特征排名热力图
    plt.subplot(2, 2, 3)
    rankings = np.array([res['ranking'] for res in results.values()])
    plt.imshow(rankings, aspect='auto', cmap='viridis_r')
    plt.colorbar(label='Feature Ranking')
    plt.xticks(band_idx)
    plt.yticks(range(len(results)), results.keys())
    plt.xlabel('Band Index')
    plt.title('RFE Feature Rankings')

    # 4. 方法一致性分析
    plt.subplot(2, 2, 4)
    all_selected = [res['selected'] for res in results.values()] + [np.argsort(kruskal_scores)[-n_select:]]
    labels = list(results.keys()) + ['Kruskal']

    # 计算Jaccard相似度
    jac_sim = np.zeros((len(all_selected), len(all_selected)))
    for i in range(len(all_selected)):
        for j in range(len(all_selected)):
            set_i = set(all_selected[i])
            set_j = set(all_selected[j])
            jac_sim[i, j] = len(set_i & set_j) / len(set_i | set_j) if len(set_i | set_j) > 0 else 0

    plt.imshow(jac_sim, vmin=0, vmax=1, cmap='YlGnBu')
    plt.colorbar(label='Jaccard Similarity')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title('Method Consistency Analysis')

    plt.tight_layout()
    plt.show()


# ================= 执行主程序 =================
if __name__ == "__main__":
    # 运行多模型RFE
    rfe_results = enhanced_rfe(X_scaled, y, 6)

    # 计算Kruskal-Wallis得分
    kruskal_scores = kruskal_feature_ranking(X_scaled, y)

    # 可视化结果
    visualize_results(rfe_results, kruskal_scores)

    # 输出最佳波段交集
    all_bands = [res['selected'] for res in rfe_results.values()] + [np.argsort(kruskal_scores)[-6:]]
    print(all_bands)
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# ================= 数据加载与预处理 =================
data = pd.read_csv(r'D:\NJU\HW\Final\filter\standard\slice.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 注意：此处应为列标准化


# ================= 多模型CARS算法 =================
def enhanced_cars(X, y, target_bands=6, n_iter=50, decay_start=0.7, models=None):
    """ 支持多回归模型的增强版CARS算法 """
    # 修改模型初始化部分
    models = {
        'PLS': PLSRegression(n_components=3),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),  # 增大alpha和迭代次数
        'ElasticNet': ElasticNet(l1_ratio=0.5, alpha=0.1, max_iter=10000),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5)
    }

    n_samples, n_bands = X.shape
    results = {}

    for model_name, model in models.items():
        remaining_bands = np.arange(n_bands)
        rmscv_values = []
        band_history = []

        decay_factor = decay_start  # 动态衰减因子

        for i in range(n_iter):
            current_bands = len(remaining_bands)
            if current_bands <= target_bands:
                break

            # 动态调整选择比例
            select_ratio = max(target_bands / current_bands * 1.5, 0.2)
            num_keep = int(current_bands * select_ratio * (decay_factor ** i))
            num_keep = max(num_keep, target_bands + 2)  # 留出缓冲

            # 蒙特卡罗采样
            sample_idx = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=False)
            X_sub = X[sample_idx][:, remaining_bands]
            y_sub = y[sample_idx]

            # 模型训练与特征权重计算
            if model_name == 'PLS':
                model.n_components = min(3, X_sub.shape[1] - 1)
                model.fit(X_sub, y_sub)
                coef = np.abs(model.coef_.ravel())
            elif model_name in ['Lasso', 'ElasticNet']:
                model.fit(X_sub, y_sub)
                coef = np.abs(model.coef_)
            else:  # RF
                model.fit(X_sub, y_sub)
                coef = model.feature_importances_

            # 波段筛选
            top_bands = np.argsort(coef)[-num_keep:]
            selected_bands = remaining_bands[top_bands]
            remaining_bands = np.sort(selected_bands)

            # 交叉验证
            kf = KFold(n_splits=5)
            rmse_list = []
            for train_idx, test_idx in kf.split(X[:, remaining_bands]):
                X_train = X[train_idx][:, remaining_bands]
                X_test = X[test_idx][:, remaining_bands]
                y_train = y[train_idx]
                y_test = y[test_idx]

                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
                rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))

            rmscv = np.mean(rmse_list)
            rmscv_values.append(rmscv)
            band_history.append(remaining_bands.copy())

            # 自适应调整衰减因子
            decay_factor = min(decay_factor * 1.05, 0.95)

        optimal_idx = np.argmin(rmscv_values)
        results[model_name] = {
            'bands': band_history[optimal_idx][:target_bands],
            'rmscv': rmscv_values[optimal_idx]
        }

    return results


# ================= 执行与可视化 =================
results = enhanced_cars(X_scaled, y, target_bands=8)

# 可视化模型对比
plt.figure(figsize=(12, 6))

# 1. RMSECV对比
plt.subplot(1, 2, 1)
model_names = list(results.keys())
rmscvs = [results[name]['rmscv'] for name in model_names]
plt.bar(model_names, rmscvs, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('RMSECV')
plt.title('Model Performance Comparison')

# 2. 波段选择对比
plt.subplot(1, 2, 2)
for i, (name, res) in enumerate(results.items()):
    plt.scatter(res['bands'], [i] * len(res['bands']), label=name, s=100)
plt.yticks(range(len(results)), model_names)
plt.xlabel('Band Index')
plt.title('Selected Bands Comparison')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 输出最优结果
best_model = min(results, key=lambda x: results[x]['rmscv'])
print(f"Best Model: {best_model}")
print(f"Selected Bands: {results[best_model]['bands']}")
print(f"RMSECV: {results[best_model]['rmscv']:.4f}")
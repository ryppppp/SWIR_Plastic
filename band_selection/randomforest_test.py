import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import generic_filter


# 修改generate_sample_data函数以支持种子参数
def generate_sample_data(size=1000, seed=None):
    np.random.seed(seed)
    data = pd.DataFrame({
        'x': np.random.randint(0, 100, size),
        'y': np.random.randint(0, 100, size),
        'feat1': np.random.normal(0, 1, size),
        'feat2': np.random.normal(1, 2, size),
        'label': np.random.choice([0, 1], size, p=[0.7, 0.3])
    })
    return data


def add_spatial_features(df, window_size=3):
    """添加邻域统计特征（修复索引错误最终版）"""
    # 创建连续的整数坐标索引
    y_unique = sorted(df['y'].unique())
    x_unique = sorted(df['x'].unique())
    y_mapping = {v: i for i, v in enumerate(y_unique)}
    x_mapping = {v: i for i, v in enumerate(x_unique)}

    # 生成完整网格模板
    grid_template = np.full((len(y_unique), len(x_unique)), np.nan)

    # 填充网格值
    for idx, row in df.iterrows():
        y_idx = y_mapping[row['y']]
        x_idx = x_mapping[row['x']]
        grid_template[y_idx, x_idx] = row['label']

    # 处理缺失值（填充为0）
    grid_filled = np.nan_to_num(grid_template, nan=0)

    # 定义邻域统计函数
    def neighborhood_stats(arr):
        center = arr[len(arr) // 2]
        neighbors = arr[arr != center]
        return np.nanmean(neighbors) if len(neighbors) > 0 else 0

    # 计算邻域特征
    neighborhood_mean = generic_filter(grid_filled, neighborhood_stats, size=window_size)

    # 反向映射到原始坐标
    reverse_y_mapping = {i: v for v, i in y_mapping.items()}
    reverse_x_mapping = {i: v for v, i in x_mapping.items()}

    # 创建坐标到特征的映射字典
    coord_feat_map = {}
    for y_idx in range(grid_filled.shape[0]):
        for x_idx in range(grid_filled.shape[1]):
            orig_y = reverse_y_mapping[y_idx]
            orig_x = reverse_x_mapping[x_idx]
            coord_feat_map[(orig_y, orig_x)] = neighborhood_mean[y_idx, x_idx]

    # 应用特征映射
    df['spatial_feat'] = df.apply(
        lambda row: coord_feat_map.get((row['y'], row['x']), 0),
        axis=1
    )

    return df

# 模型融合类
class SpatialEnsemble:
    def __init__(self):
        self.spatial_model = RandomForestClassifier(n_estimators=100)
        self.classifier = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # 第一阶段：空间特征学习
        spatial_features = X[['x', 'y', 'spatial_feat']]
        self.spatial_model.fit(spatial_features, y)

        # 第二阶段：联合特征训练
        spatial_proba = self.spatial_model.predict_proba(spatial_features)
        combined_features = np.hstack([X.drop(['x', 'y', 'spatial_feat'], axis=1), spatial_proba])
        combined_features = self.scaler.fit_transform(combined_features)
        self.classifier.fit(combined_features, y)

    def predict(self, X):
        # 生成空间特征概率
        spatial_features = X[['x', 'y', 'spatial_feat']]
        spatial_proba = self.spatial_model.predict_proba(spatial_features)

        # 组合特征
        combined_features = np.hstack([X.drop(['x', 'y', 'spatial_feat'], axis=1), spatial_proba])
        combined_features = self.scaler.transform(combined_features)

        return self.classifier.predict(combined_features)


    # 5. 后处理（可选：空间平滑）
def spatial_smoothing(predictions, coords, radius=2):
        smoothed = predictions.copy()
        for i in range(len(coords)):
            x, y = coords[i]
            neighbors = predictions[
                (coords[:, 0] >= x - radius) & (coords[:, 0] <= x + radius) &
                (coords[:, 1] >= y - radius) & (coords[:, 1] <= y + radius)
                ]
            smoothed[i] = np.argmax(np.bincount(neighbors))
        return smoothed



# 主流程
if __name__ == "__main__":
    # 1. 数据准备
    data = generate_sample_data()
    data = add_spatial_features(data)

    # 2. 划分数据集
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 训练模型
    model = SpatialEnsemble()
    model.fit(X_train, y_train)

    # 4. 评估
    y_pred = model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")



    coords = X_test[['x', 'y']].values
    y_smoothed = spatial_smoothing(y_pred, coords)
    print(f"平滑后准确率: {accuracy_score(y_test, y_smoothed):.4f}")


    # 在文件末尾添加以下代码
def main_execution(seed=None):
        """封装主流程为可调用函数"""
        # 1. 数据准备
        data = generate_sample_data(size=1000, seed=seed)
        data = add_spatial_features(data)

        # 2. 划分数据集
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        # 3. 训练模型
        model = SpatialEnsemble()
        model.fit(X_train, y_train)

        # 4. 评估原始准确率
        y_pred = model.predict(X_test)
        base_acc = accuracy_score(y_test, y_pred)

        # 5. 计算平滑后准确率
        coords = X_test[['x', 'y']].values
        y_smoothed = spatial_smoothing(y_pred, coords)
        smoothed_acc = accuracy_score(y_test, y_smoothed)

        return base_acc, smoothed_acc
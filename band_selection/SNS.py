from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# 修改 KMeans 部分，添加以下代码
if len(sys.argv) > 1:
    random_state = int(sys.argv[1])
else:
    random_state = 367  # 默认值

# 加载数据
data = pd.read_csv(r'D:\NJU\HW\Final\filter\standard\slice.csv')

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_scaled=np.array(X)

# 1. 计算波段间相关系数矩阵
corr_matrix = np.corrcoef(X_scaled, rowvar=False)

# 2. 聚类波段（假设分为10个簇）
kmeans = KMeans(n_clusters=6, random_state=random_state)
cluster_labels = kmeans.fit_predict(corr_matrix)

# 3. 从每个簇中选择最中心波段
selected_bands = []
for cluster_id in range(6):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    # 计算簇内波段与簇中心的距离
    cluster_center = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(corr_matrix[cluster_indices] - cluster_center, axis=1)
    # 选择距离最近的波段作为代表
    selected_bands.append(cluster_indices[np.argmin(distances)])

selected_bands = np.array(selected_bands)
# 4. 输出选中波段
print("聚类选中的波段索引:", selected_bands)

# 5. 验证性能
X_selected = X_scaled[:, selected_bands]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("聚类筛选后的分类准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

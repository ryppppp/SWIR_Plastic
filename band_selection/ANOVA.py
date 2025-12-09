from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
data = pd.read_csv(r'D:\NJU\HW\Final\filter\standard\slice.csv')
filterlist = ['FBH910-10.xlsx','FBH920-10.xlsx','FBH930-10.xlsx','FBH940-10.xlsx','FBH950-10.xlsx','FBH960-10.xlsx','FBH970-10.xlsx','FBH980-10.xlsx',
              'FBH990-10.xlsx','FBH1000-10.xlsx','FBH1050-10.xlsx','FBH1070-10.xlsx','FBH1100-10.xlsx','FBH1150-10.xlsx','FBH1200-10.xlsx','FBH1250-10.xlsx',
              'FBH1300-12.xlsx','FBH1310-12.xlsx','FBH1320-12.xlsx','FBH1330-12.xlsx','FBH1340-12.xlsx','FBH1350-12.xlsx','FBH1400-12.xlsx','FBH1450-12.xlsx',
              'FBH1480-12.xlsx','FBH1490-12.xlsx','FBH1500-12.xlsx','FBH1510-12.xlsx','FBH1520-12.xlsx','FBH1530-12.xlsx','FBH1540-12.xlsx','FBH1550-12.xlsx',
              'FBH1560-12.xlsx','FBH1570-12.xlsx','FBH1580-12.xlsx','FBH1590-12.xlsx','FBH1600-12.xlsx','FBH1610-12.xlsx','FBH1620-12.xlsx','FBH1650-12.xlsx',
              'FLH1030-10.xlsx','FLH1064-10.xlsx']

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

#数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 计算每个波段的ANOVA F值
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
f_scores = selector.scores_

# 3. 按F值排序选择Top波段
sorted_bands = np.argsort(f_scores)[::-1]  # 降序排列
selected_bands = sorted_bands[:10]  # 取Top 10波段
print(selected_bands)
for j in selected_bands:
    print(filterlist[j])
# 4. 可视化F值分布
plt.figure(figsize=(12, 4))
plt.bar(range(len(f_scores)), f_scores, alpha=0.7)
plt.xlabel('Bands')
plt.ylabel('ANOVA F values')
plt.title('ability of each band to classify')
plt.axhline(y=np.median(f_scores), color='r', linestyle='--', label='median')
plt.legend()
plt.show()

# 5. 使用选中的波段训练SVM
X_selected = X[:, selected_bands]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Top 10波段分类准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

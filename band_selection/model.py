import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
import numpy as np

selected_bands =[13, 14, 17, 38]

# 加载数据
data = pd.read_csv(r'D:\NJU\HW\Final\filter\slice.csv')

X=data.iloc[:,selected_bands].values
y=data.iloc[:,-1].values

#数据标准化
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

X_selected = X_scaled  # 标准化后的高光谱数据（N个样本×M个波段）

#训练svm模型
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3)
svm = SVC(C=500,kernel='poly',gamma='scale',degree=3,coef0=1.0)
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print("准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

#保存模型
import pickle

model=svm

with open("pc_robust.pkl","wb") as f:
    # noinspection PyTypeChecker
    pickle.dump({
        'model':model,
        'scaler':scaler,
    },f)

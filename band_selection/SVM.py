import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
selected_bands = [13, 14, 17, 38]  # 从42维中筛选的4个波段
scalers = {
    'none': None,  # 无标准化
    'minmax': MinMaxScaler(),  # 按列MinMax
    'standard': StandardScaler(),  # 按列Z-score
    'robust': RobustScaler()  # 按列鲁棒标准化
}

# 扩展的网格搜索参数
param_grid = {
    'C': [0.1, 1, 10, 100, 500],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3],
    'coef0': [0.0, 1.0]
}

# 结果记录字典
results = {
    'scaler': [],
    'best_params': [],
    'test_acc': []
}


# 列标准化函数
def column_scaling(X, scaler_name):
    """按列标准化或返回原始数据"""
    if scaler_name == 'none':
        return X
    scaler = scalers[scaler_name]
    return scaler.fit_transform(X)


# 增强可视化函数
def plot_results(scaler_name, test_acc, cm, best_params,classes):
    plt.figure(figsize=(15, 6))

    # 混淆矩阵
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title = f"Confusion Matrix ({scaler_name})\n"
    title += f"Best Params: {best_params}\n"
    title += f"Test Acc: {test_acc:.4f}"
    plt.title(title)

    plt.tight_layout()
    plt.show()


# 准确率对比函数
def plot_acc_comparison():
    plt.figure(figsize=(10, 6))
    plt.bar(results['scaler'], results['test_acc'], color=['skyblue', 'salmon', 'lightgreen', 'gold'])
    plt.ylim(0, 1.0)
    plt.xlabel('Scaler Method')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Comparison Across Scaling Methods')
    for i, acc in enumerate(results['test_acc']):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center')
    plt.show()


# 主训练流程
def train_model(slice_path):
    # 加载数据
    data = pd.read_csv(slice_path)
    X = data.iloc[:, :-1].values[:, selected_bands]  # 直接筛选波段
    y = data.iloc[:, -1].values

    # 获取所有实际类别标签（假设标签为1-11）
    classes = np.unique(y).astype(int)  # 确保标签为整数
    # 遍历所有标准化方法
    for scaler_name in scalers.keys():
        # 按列标准化
        if scaler_name == 'none':
            X_scaled = X
        else:
            X_scaled = column_scaling(X, scaler_name)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)

        # 网格搜索
        svc = SVC()
        clf = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)

        # 记录结果
        best_params = clf.best_params_
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results['scaler'].append(scaler_name)
        results['best_params'].append(best_params)
        results['test_acc'].append(test_acc)

        # 输出并可视化当前结果
        print(f"\n=== {scaler_name.upper()} Scaler ===")
        print("Best params:", best_params)
        print(f"Test Accuracy: {test_acc:.4f}")
        plot_results(scaler_name, test_acc, cm, best_params,classes)

    # 最终准确率对比
    plot_acc_comparison()


# 执行训练
train_model(r'D:\NJU\HW\Final\filter\slice.csv')
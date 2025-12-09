import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from randomforest_test import main_execution


def run_experiments(num_trials=50):
    """执行多次实验并记录结果"""
    accuracy_diffs = []
    base_accuracies = []
    smoothed_accuracies = []

    for seed in tqdm(range(num_trials), desc="Running Experiments"):
        base_acc, smoothed_acc = main_execution(seed=seed)
        accuracy_diff = smoothed_acc - base_acc

        accuracy_diffs.append(accuracy_diff)
        base_accuracies.append(base_acc)
        smoothed_accuracies.append(smoothed_acc)

    # 可视化结果
    plt.figure(figsize=(12, 6))

    # 准确率提升曲线
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_diffs, 'o--', color='darkorange', alpha=0.7)
    plt.axhline(np.mean(accuracy_diffs), color='red', linestyle='--')
    plt.title('Accuracy Improvement After Smoothing')
    plt.xlabel('Trial')
    plt.ylabel('Improvement')
    plt.grid(True)

    # 提升分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(accuracy_diffs, bins=15, color='skyblue', edgecolor='black')
    plt.title('Improvement Distribution')
    plt.xlabel('Improvement')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"\n[统计摘要]")
    print(f"平均提升: {np.mean(accuracy_diffs):.4f}")
    print(f"最大提升: {np.max(accuracy_diffs):.4f}")
    print(f"最小提升: {np.min(accuracy_diffs):.4f}")
    print(f"正提升比例: {sum(np.array(accuracy_diffs) > 0) / len(accuracy_diffs):.1%}")


if __name__ == "__main__":
    run_experiments(num_trials=50)
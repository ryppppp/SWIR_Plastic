import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False# 加载模型和预处理对象

with open("pc_robust.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']  # 获取已训练好的标准化器

#file_list = ['ABS_1_result.csv', 'ABS_2_result.csv', 'abs_2nd_result.csv', 'pa_2nd_result.csv','PC_1_result.csv', 'PC_2_result.csv', 'pc_2nd_result.csv', 'PE_1_result.csv','PE_2_result.csv', 'pe_2nd_result.csv', 'PET_1_result.csv', 'PET_2_result.csv',
  #  'pet_2nd_result.csv', 'pmma_2nd_result.csv', 'PMMA_result.csv', 'PP_1_result.csv','PP_2_result.csv', 'pp_2nd_result.csv', 'PS_1_result.csv', 'PS_2_result.csv','ps_2nd_result.csv', 'pu_2nd_result.csv', 'PVC_1_result.csv', 'pvc_2nd_result.csv',
 #   'pvc_transparent_2nd_result.csv', 'sx_2nd_result.csv', 'SX_result.csv']

file_list = ['abs_f_result.csv','pa_f_result.csv','pc_f_result.csv','pe_f_result.csv','pet_f_result.csv','pmma_f_result.csv','pp_f_result.csv','ps_f_result.csv','pvc_f_result.csv']

selected_bands = [13, 14, 17, 38]

# 初始化数据结构
accuracy_results = {}
short_labels = {}

# 循环处理文件
for filename in file_list:
    # 读取数据
    new_data = pd.read_csv(rf"D:\NJU\HW\Final_test\filter\{filename}")
    new_data_X = new_data.iloc[:, selected_bands].values
    new_data_y = new_data.iloc[:, -1].values

    # 数据标准化（使用已训练的scaler）
    new_data_X_scaled = scaler.transform(new_data_X)
    #new_data_X_scaled = new_data_X

    # 预测并计算准确率
    new_y_pred = model.predict(new_data_X_scaled)
    accuracy = accuracy_score(new_data_y, new_y_pred) * 100

    # 提取简化标签（取第一个下划线前的部分，不区分大小写）
    label = filename.split('_')[0].upper()  # 统一转为大写
    if label not in accuracy_results:
        accuracy_results[label] = []
        short_labels[label] = filename.split('_')[0]  # 保留原始大小写的显示标签
    accuracy_results[label].append(accuracy)


# 计算平均准确率用于可视化
labels = []
avg_accuracies = []
for label, acc_list in accuracy_results.items():
    labels.append(short_labels[label])  # 使用原始大小写的标签
    avg_accuracies.append(np.mean(acc_list))

# 绘制柱状图
plt.figure(figsize=(15, 6))

# 使用tab10色系生成颜色（10种循环颜色）
colors = plt.cm.tab10(np.arange(len(labels)) % 10)

# 绘制柱状图（设置zorder=3让柱子显示在网格上方）
bars = plt.bar(labels, avg_accuracies,
               color=colors,          # 使用丰富颜色
               edgecolor='black',
               linewidth=0.8,
               zorder=3)             # 控制图层顺序

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.1f}%',
             ha='center', va='bottom',
             fontsize=10,
             color='black',          # 确保文字颜色对比
             zorder=4)              # 文字在柱子之上

# 图表装饰
plt.xlabel('材料类型', fontsize=12)
plt.ylabel('平均准确率 (%)', fontsize=12)
plt.title('robust:不同材料分类准确率对比（按类别分组）', fontsize=14)
plt.xticks(rotation=45, ha='right')

# 优化网格设置（使用浅灰色，降低透明度，zorder=2在柱子下方）
plt.grid(axis='y',
         linestyle='--',
         alpha=0.4,                 # 降低透明度
         color='#666666',           # 灰色网格线
         zorder=2)                  # 网格在柱子下方

plt.ylim(0, 110)
plt.tight_layout()

# 添加图例示例（按需添加）
# handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(labels))]
# plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
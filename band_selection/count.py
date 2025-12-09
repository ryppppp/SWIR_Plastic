import re
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

count = defaultdict(int)

# 读取日志文件并统计波段出现次数
with open("output.log", "r", encoding="utf-16-le") as f:
    for line in f:
        if "聚类选中的波段索引:" in line:
            bands = re.findall(r"\d+", line.split(": ")[1])
            for band in bands:
                count[int(band)] += 1

# 打印统计结果
print("波段索引出现次数统计:")
for band in sorted(count):
    print(f"波段 {band}: {count[band]} 次")

# 可视化柱形图
if count:
    bands = list(sorted(count.keys()))
    frequencies = [count[band] for band in bands]

    plt.figure(figsize=(12, 6))
    plt.bar(bands, frequencies, color='skyblue', edgecolor='black')
    plt.xlabel('波段索引', fontsize=12)
    plt.ylabel('出现次数', fontsize=12)
    plt.title('波段索引出现次数统计对比', fontsize=14)
    plt.xticks(bands, rotation=45)  # 显示所有波段索引刻度
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱顶显示数值
    for i, freq in enumerate(frequencies):
        plt.text(bands[i], freq, str(freq), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("未找到任何波段索引数据，无法生成图表。")
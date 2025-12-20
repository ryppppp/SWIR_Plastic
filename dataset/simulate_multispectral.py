import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# 配置：原始波段信息
# ---------------------------
num_bands = 224
wl_min = 900    
wl_max = 1700    
wavelengths = np.linspace(wl_min, wl_max, num_bands)

# ---------------------------
# 定义 4 个滤光片（高斯响应）
# ---------------------------
filters = {
    "1140": {"center": 1140, "half_bw": 10, "peak": 0.9},
    "1200": {"center": 1200, "half_bw": 25, "peak": 0.8},
    "1245": {"center": 1245, "half_bw": 13, "peak": 0.9},
    "1310": {"center": 1310, "half_bw": 25, "peak": 0.8},
}

def gaussian_filter(wl, center, half_bw, peak):
    """生成滤光片的高斯透过率曲线"""
    sigma = half_bw / 2.355  # half bandwidth = FWHM, FWHM = 2.355σ
    return peak * np.exp(-0.5 * ((wl - center) / sigma)**2)

# ---------------------------
# 主处理流程
# ---------------------------
def simulate_multispectral(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    band_columns = [f"band_{i}" for i in range(num_bands)]
    spectra = df[band_columns].values

    # 计算每个滤光片的响应
    filter_responses = {}
    for f_name, params in filters.items():
        resp = gaussian_filter(
            wavelengths,
            params["center"],
            params["half_bw"],
            params["peak"]
        )
        filter_responses[f_name] = resp

    # ---------------------------
    # 对每个像素点做积分
    # multispectral = Σ spectrum(λ) * filter(λ)
    # ---------------------------
    output = {}
    for f_name, resp in filter_responses.items():
        output[f_name] = np.sum(spectra * resp, axis=1)

    # ---------------------------
    # 构造输出CSV
    # ---------------------------
    out_df = pd.DataFrame({
        "x": df["x"],
        "y": df["y"],
        "1140": output["1140"],
        "1200": output["1200"],
        "1245": output["1245"],
        "1310": output["1310"],
        "label": df["label"]
    })

    out_df.to_csv(output_csv, index=False)
    print("生成完成 →", output_csv)

def batch_simulate_multispectral(input_dir, output_dir=None):
    """
    input_dir: 存放原始高光谱 csv 的文件夹路径
    output_dir: 输出结果存放的文件夹路径；如果为 None，则输出到 input_dir
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历文件夹下所有 csv 文件
    for csv_path in input_dir.glob("*.csv"):
        output_csv = output_dir / f"{csv_path.stem}_4bands.csv"
        print(f"处理文件: {csv_path} → {output_csv}")
        simulate_multispectral(str(csv_path), str(output_csv))


if __name__ == "__main__":
    input_dir = "./dataset/HSI_2"
    output_dir = "./dataset/MSI_2"
    batch_simulate_multispectral(input_dir, output_dir)

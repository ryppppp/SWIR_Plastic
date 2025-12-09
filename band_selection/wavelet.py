import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import pandas as pd

os.chdir(r'D:\NJU\3rd')
file_list=['abs_1_3rd_result.csv','pa_1_3rd_result.csv','pa_2_3rd_result.csv','pc_1_3rd_result.csv','pe_1_3rd_result.csv','pe_2_3rd_result.csv','pet_1_3rd_result.csv','pmma_1_3rd_result.csv','pp_1_3rd_result.csv',
           'ps_1_3rd_result.csv','pu_1_3rd_result.csv','pvc_1_3rd_result.csv','pvc_2_3rd_result.csv']

# Savitzky-Golay滤波器
def savgol_filtering(data, window_size, order):
    return savgol_filter(data, window_size, order)

#读取数据并应用噪声降低技术
for j in range(0,len(file_list)):
    df = pd.DataFrame(pd.read_csv(file_list[j]))
    x_data = np.arange(0, 224, 1)
    filedata = np.array(df)
    mainframe = pd.DataFrame()

    for i in range(0, filedata.shape[0]):
        y_data = filedata[i:i + 1, 2:-1].flatten()
        sg_data = savgol_filtering(y_data, 15, 3)  # 窗口大小15，多项式阶数3的Savitzky-Golay滤波
        sg_temp = pd.DataFrame(sg_data).transpose()
        #print(sg_temp)
        mainframe = pd.concat([mainframe, sg_temp], axis=0)
    print(mainframe)
    mainframe.to_csv(rf'D:\NJU\3rd\noise\{file_list[j]}', index=False)

    mainplot=np.array(mainframe)
    num_samples = filedata.shape[0]
    # 绘制MSC校正后的数据
    ax = plt.subplots()
    for i in range(num_samples):
        ax[1].plot(mainplot[i, :], label=f'Sample {i + 1}' if i == 0 else None)
    ax[1].set_title('MSC Corrected Spectra')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Wavelength Index')
    ax[1].legend()
    plt.show()


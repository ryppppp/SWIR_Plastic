import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



os.chdir(r'D:\NJU\3rd')
#file_list=['abs_2nd_result.csv','pa_2nd_result.csv','pc_2nd_result.csv','pe_2nd_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','pp_2nd_result.csv','ps_2nd_result.csv','pu_2nd_result.csv','pvc_2nd_result.csv','pvc_transparent_2nd_result.csv','sx_2nd_result.csv']
#file_list=['ABS_1_result.csv','ABS_2_result.csv','PC_1_result.csv','PC_2_result.csv','PE_1_result.csv','PE_2_result.csv','PET_1_result.csv','PET_2_result.csv','PMMA_result.csv','PP_1_result.csv','PP_2_result.csv','PS_1_result.csv','PS_2_result.csv','PVC_1_result.csv','SX_result.csv','antiflaming_ABS_result.csv','antiflaming_PS_result.csv','PA_result.csv']
file_list=['abs_1_3rd_result.csv','pa_1_3rd_result.csv','pa_2_3rd_result.csv','pc_1_3rd_result.csv','pe_1_3rd_result.csv','pe_2_3rd_result.csv','pet_1_3rd_result.csv','pmma_1_3rd_result.csv','pp_1_3rd_result.csv','ps_1_3rd_result.csv','pu_1_3rd_result.csv','pvc_1_3rd_result.csv','pvc_2_3rd_result.csv']
# 进行多元散射校正（MSC）
def do_msc(input_data):
    # 计算平均光谱作为参考
    ref_spectrum = np.mean(input_data, axis=0)
    corrected_data = np.zeros_like(input_data)

    for i in range(input_data.shape[0]):
        # 对每个样本进行最小二乘线性回归
        fit = np.polyfit(ref_spectrum, input_data[i, :], 1)
        # 应用校正
        corrected_data[i, :] = (input_data[i, :] - fit[1]) / fit[0]

    return corrected_data, ref_spectrum


#剔除异常值
def decrease_abnormal(input_data):
    i = 0
    while i < input_data.shape[0]:
        for m in range(input_data.shape[1]):
            if input_data[i, m] < 0 or input_data[i, m] > 1:
                input_data = np.delete(input_data, i, 0)
                i -= 1
                break
        i += 1
    return input_data

for j in range(0,len(file_list)):
    # 导入数据
    df = pd.DataFrame(pd.read_csv(file_list[j]))
    #data = np.array(df.iloc[:,10:-11])
    data = np.array(df.iloc[:, 12:-12])
    num_samples = data.shape[0]

    data = decrease_abnormal(data)

    # 应用MSC
    corrected_spectra, mean_spectrum = do_msc(data)

    corrected_spectra = decrease_abnormal(corrected_spectra)

    # 绘制原始和校正后的光谱数据
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 绘制原始数据
    for i in range(data.shape[0]):
        ax[0].plot(data[i, :], label=f'Sample {i + 1}' if i == 0 else None)
    ax[0].set_title(f'Original Spectra:{file_list[j]}')
    ax[0].set_ylabel('Intensity')
    ax[0].legend()

    # 绘制MSC校正后的数据
    for i in range(corrected_spectra.shape[0]):
        ax[1].plot(corrected_spectra[i, :], label=f'Sample {i + 1}' if i == 0 else None)
    ax[1].set_title('MSC Corrected Spectra')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Wavelength Index')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    print('success')
    #导出
    #msc=pd.DataFrame(corrected_spectra)
    #msc.to_csv(rf'D:\NJU\3rd\MSC\{file_list[j]}', index=False)
    #print(msc)

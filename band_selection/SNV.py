import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_list = ['ABS_1_result.csv', 'ABS_2_result.csv','abs_2nd_result.csv', 'pa_2nd_result.csv','PC_1_result.csv','PC_2_result.csv','pc_2nd_result.csv','PE_1_result.csv','PE_2_result.csv','pe_2nd_result.csv','PET_1_result.csv','PET_2_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','PMMA_result.csv',
         'PP_1_result.csv','PP_2_result.csv','pp_2nd_result.csv','PS_1_result.csv','PS_2_result.csv','ps_2nd_result.csv','pu_2nd_result.csv','PVC_1_result.csv','pvc_2nd_result.csv','pvc_transparent_2nd_result.csv','SX_result.csv','sx_2nd_result.csv']

def snv_correction(X):
    # SNV标准化
    X_snv = np.zeros_like(X)
    for i in range(X.shape[0]):
        # 计算每个样本的均值和标准差
        sample_mean = np.mean(X[i, :])
        sample_std = np.std(X[i, :])

        # 处理标准差为零的情况（所有波段值相同）
        if sample_std == 0:
            X_snv[i, :] = 0
        else:
            X_snv[i, :] = (X[i, :] - sample_mean) / sample_std

    return X_snv


for j in range(0,len(file_list)):
    # 加载数据
    data = pd.read_csv(rf"D:\NJU\HW\Final\filter\{file_list[j]}")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1]

    X_snv = snv_correction(X)


    ax = plt.subplots()
    # 绘制归一化后的数据
    for i in range(X_snv.shape[0]):
        ax[1].plot(X_snv[i, :], label=f'Sample {i + 1}' if i == 0 else None)
    ax[1].set_title(f'Scaled Data:{file_list[j]}')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Wavelength Index')
    ax[1].legend()
    plt.show()

    X_scaled = pd.DataFrame(X_snv)
    data_scaled = pd.concat([X_scaled, y], axis=1)
    data_scaled.to_csv(rf"D:\NJU\HW\Final\filter\standard\{file_list[j]}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#加载新数据
#file_list=['abs_2nd_result.csv','pc_2nd_result.csv','pe_2nd_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','pp_2nd_result.csv','ps_2nd_result.csv','pvc_2nd_result.csv','sx_2nd_result.csv']
#file_list=['ABS_1_result.csv','PA_result.csv','ABS_2_result.csv','PC_1_result.csv','PC_2_result.csv','PE_1_result.csv','PE_2_result.csv','PET_1_result.csv','PET_2_result.csv','PMMA_result.csv','PP_1_result.csv','PP_2_result.csv','PS_1_result.csv','PS_2_result.csv','PVC_1_result.csv','SX_result.csv']
#file_list=['abs_1_3rd_result.csv','pa_1_3rd_result.csv','pa_2_3rd_result.csv','pc_1_3rd_result.csv','pe_1_3rd_result.csv','pe_2_3rd_result.csv','pet_1_3rd_result.csv','pmma_1_3rd_result.csv','pp_1_3rd_result.csv','ps_1_3rd_result.csv','pvc_1_3rd_result.csv','pvc_2_3rd_result.csv']
#file_list=['abs.csv','pa.csv','pc.csv','pe.csv','pet.csv','pet.csv','pmma.csv','pp.csv','ps.csv','pvc.csv','sx.csv','pu.csv']
#file_list = ['ABS_1_result.csv', 'ABS_2_result.csv','abs_2nd_result.csv', 'pa_2nd_result.csv','PC_1_result.csv','PC_2_result.csv','pc_2nd_result.csv','PE_1_result.csv','PE_2_result.csv','pe_2nd_result.csv','PET_1_result.csv','PET_2_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','PMMA_result.csv',
  #           'PP_1_result.csv','PP_2_result.csv','pp_2nd_result.csv','PS_1_result.csv','PS_2_result.csv','ps_2nd_result.csv','pu_2nd_result.csv','PVC_1_result.csv','pvc_2nd_result.csv','pvc_transparent_2nd_result.csv','SX_result.csv','sx_2nd_result.csv']
file_list = ['abs_f_result.csv','pa_f_result.csv','pc_f_result.csv','pe_f_result.csv','pet_f_result.csv','pmma_f_result.csv','pp_f_result.csv','ps_f_result.csv','pvc_f_result.csv']

def linear_normalization(data):
    # 计算最大值和最小值
    min_val = np.min(data)
    max_val = np.max(data)

    # 线性归一化
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data

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
    # 加载数据
    data = pd.read_csv(rf"D:\NJU\HW\Final_test\{file_list[j]}")

    X = data.iloc[:, 2:-1]
    y = data.iloc[:, -1]
    X = np.array(X)

    #X = decrease_abnormal(X)
    X_scaled = linear_normalization(X)

    ax = plt.subplots()
    # 绘制归一化后的数据
    for i in range(X_scaled.shape[0]):
        ax[1].plot(X_scaled[i, :], label=f'Sample {i + 1}' if i == 0 else None)
    ax[1].set_title(f'Scaled Data:{file_list[j]}')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Wavelength Index')
    ax[1].legend()
    plt.show()
    #data_scaled = pd.concat([X_scaled,y], axis=1)
    #data_scaled.to_csv(rf"D:\NJU\Standard_databse\scaled\{file_list[j]}")
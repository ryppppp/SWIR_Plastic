import numpy as np
import pandas as pd
import os

os.chdir(r'D:\NJU\HW\Final\filter\standard')
file_list = ['ABS_1_result.csv', 'ABS_2_result.csv','abs_2nd_result.csv', 'pa_2nd_result.csv','PC_1_result.csv','PC_2_result.csv','pc_2nd_result.csv','PE_1_result.csv','PE_2_result.csv','pe_2nd_result.csv','PET_1_result.csv','PET_2_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','PMMA_result.csv',
             'PP_1_result.csv','PP_2_result.csv','pp_2nd_result.csv','PS_1_result.csv','PS_2_result.csv','ps_2nd_result.csv','pu_2nd_result.csv','PVC_1_result.csv','pvc_2nd_result.csv','pvc_transparent_2nd_result.csv','sx_2nd_result.csv','SX_result.csv']

# 读取CSV文件并进行数据裁剪
main_dataframe = pd.DataFrame()
for i in range(0,len(file_list)-1):
    df = pd.DataFrame(pd.read_csv(file_list[i]))
    # 转换为NumPy数组
    data = np.array(df)

    #数据裁剪
    indices = np.random.choice(data.shape[0], 1000, replace=False)
    data_slice=data[indices,:]
    data_slice=pd.DataFrame(data_slice)
    main_dataframe = pd.concat([main_dataframe, data_slice], axis=0)

main_dataframe.to_csv('slice.csv', index=False)

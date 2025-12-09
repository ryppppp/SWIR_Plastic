import pandas as pd
import numpy as np

#file_list=['abs_2nd_result.csv','pc_2nd_result.csv','pe_2nd_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','pp_2nd_result.csv','ps_2nd_result.csv','pvc_2nd_result.csv','sx_2nd_result.csv']
#file_list=['ABS_1_result.csv','ABS_2_result.csv','PC_1_result.csv','PC_2_result.csv','PE_1_result.csv','PE_2_result.csv','PET_1_result.csv','PET_2_result.csv','PMMA_result.csv','PP_1_result.csv','PP_2_result.csv','PS_1_result.csv','PS_2_result.csv','PVC_1_result.csv','SX_result.csv']
#file_list=['abs_1_3rd_result.csv','pa_1_3rd_result.csv','pa_2_3rd_result.csv','pc_1_3rd_result.csv','pe_1_3rd_result.csv','pe_2_3rd_result.csv','pet_1_3rd_result.csv','pmma_1_3rd_result.csv','pp_1_3rd_result.csv','ps_1_3rd_result.csv','pvc_1_3rd_result.csv','pvc_2_3rd_result.csv']
#file_list=['abs.csv','pa.csv','pc.csv','pe.csv','pet.csv','pet.csv','pmma.csv','pp.csv','ps.csv','pvc.csv','sx.csv','pu.csv']
file_list = ['PA_result.csv']
for i in range(0,len(file_list)):
    new_data = pd.read_csv(rf"D:\NJU\nev_CSV_data\{file_list[i]}")
    new_data_X = new_data.iloc[:, 2:-1]
    x = pd.DataFrame(np.array(new_data_X))
    new_data_y = new_data.iloc[:, -1]
    y = pd.DataFrame(np.array(new_data_y))
    final = pd.concat([x,y], axis=1)
    final.to_csv(rf"D:\NJU\nev_CSV_data\Original\{file_list[i]}",index=False)
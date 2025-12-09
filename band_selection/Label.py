import pandas as pd

#file_list=['abs_2nd_result.csv','pa_2nd_result.csv','pc_2nd_result.csv','pe_2nd_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','pp_2nd_result.csv','ps_2nd_result.csv','pvc_2nd_result.csv','sx_2nd_result.csv']
#file_list=['ABS_1_result.csv','ABS_2_result.csv','PC_1_result.csv','PC_2_result.csv','PE_1_result.csv','PE_2_result.csv','PET_1_result.csv','PET_2_result.csv','PMMA_result.csv','PP_1_result.csv','PP_2_result.csv','PS_1_result.csv','PS_2_result.csv','PVC_1_result.csv','SX_result.csv','antiflaming_ABS_result.csv','antiflaming_PS_result.csv','PA_result.csv']
file_list=['abs_1_3rd_result.csv','pa_1_3rd_result.csv','pa_2_3rd_result.csv','pc_1_3rd_result.csv','pe_1_3rd_result.csv','pe_2_3rd_result.csv','pet_1_3rd_result.csv','pmma_1_3rd_result.csv','pp_1_3rd_result.csv','ps_1_3rd_result.csv','pu_1_3rd_result.csv','pvc_1_3rd_result.csv','pvc_2_3rd_result.csv']

for j in range(11,13):
    data = pd.DataFrame(pd.read_csv(rf"D:\NJU\3rd\MSC\{file_list[j]}"))
    data.insert(data.shape[1], '203', 9)
    data.to_csv(rf"D:\NJU\3rd\Label\{file_list[j]}", index=False)

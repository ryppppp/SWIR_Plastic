import pandas as pd
import numpy as np

#datalist = ['ABS_1_result.csv', 'ABS_2_result.csv','abs_2nd_result.csv', 'pa_2nd_result.csv','PC_1_result.csv','PC_2_result.csv','pc_2nd_result.csv','PE_1_result.csv','PE_2_result.csv','pe_2nd_result.csv','PET_1_result.csv','PET_2_result.csv','pet_2nd_result.csv','pmma_2nd_result.csv','PMMA_result.csv',
 #            'PP_1_result.csv','PP_2_result.csv','pp_2nd_result.csv','PS_1_result.csv','PS_2_result.csv','ps_2nd_result.csv','pu_2nd_result.csv','PVC_1_result.csv','pvc_2nd_result.csv','pvc_transparent_2nd_result.csv','SX_result.csv','sx_2nd_result.csv']
datalist = ['abs_f_result.csv','pa_f_result.csv','pc_f_result.csv','pe_f_result.csv','pet_f_result.csv','pmma_f_result.csv','pp_f_result.csv','ps_f_result.csv','pvc_f_result.csv']

filterlist = ['FBH910-10.xlsx','FBH920-10.xlsx','FBH930-10.xlsx','FBH940-10.xlsx','FBH950-10.xlsx','FBH960-10.xlsx','FBH970-10.xlsx','FBH980-10.xlsx',
              'FBH990-10.xlsx','FBH1000-10.xlsx','FBH1050-10.xlsx','FBH1070-10.xlsx','FBH1100-10.xlsx','FBH1150-10.xlsx','FBH1200-10.xlsx','FBH1250-10.xlsx',
              'FBH1300-12.xlsx','FBH1310-12.xlsx','FBH1320-12.xlsx','FBH1330-12.xlsx','FBH1340-12.xlsx','FBH1350-12.xlsx','FBH1400-12.xlsx','FBH1450-12.xlsx',
              'FBH1480-12.xlsx','FBH1490-12.xlsx','FBH1500-12.xlsx','FBH1510-12.xlsx','FBH1520-12.xlsx','FBH1530-12.xlsx','FBH1540-12.xlsx','FBH1550-12.xlsx',
              'FBH1560-12.xlsx','FBH1570-12.xlsx','FBH1580-12.xlsx','FBH1590-12.xlsx','FBH1600-12.xlsx','FBH1610-12.xlsx','FBH1620-12.xlsx','FBH1650-12.xlsx',
              'FLH1030-10.xlsx','FLH1064-10.xlsx']

#插值函数
def insert_spectra(spectral_data):
    # 生成原始波段的中心波长（224个点）
    original_wl = 900 + (1700 - 900) / 224 * np.arange(224)

    # 生成目标整数波长（900到1700共801个点）
    target_wl = np.arange(900, 1701)  # 1700是闭合区间终点

    # 执行线性插值
    #interpolated = np.interp(target_wl, original_wl, spectral_data)
    interpolated = np.array([np.interp(target_wl, original_wl, sample) for sample in spectral_data])

    return interpolated

for i in range(0,len(datalist)):
    # 加载数据
    data = pd.read_csv(rf"D:\NJU\HW\Final_test\{datalist[i]}")
    X = data.iloc[:, 2:-1]
    y = data.iloc[:, -1]

    #插值
    datax= np.array(X)
    interp_data = insert_spectra(datax)

    new_data = pd.DataFrame()
    #滤波
    for j in range(0,len(filterlist)):
        filter_data =  pd.read_excel(rf"D:\NJU\HW\Filter\{filterlist[j]}",usecols=[2,4])
        filter_data = filter_data[(filter_data['Wavelength (nm)']<=1700) & (filter_data['Wavelength (nm)']>=900)&(filter_data['Wavelength (nm)']%1==0)]
        wavelength_array =np.array(filter_data['Wavelength (nm)'],dtype=int)
        transmission_array = np.array(filter_data['% Transmission'])

        sum = np.zeros(interp_data.shape[0])
        m=0
        for k in wavelength_array - 900:
            temp= interp_data[:,k]*transmission_array[m] / 100
            sum += temp
            m += 1
        sum = pd.DataFrame(sum)
        new_data = pd.concat([new_data, sum], axis=1)
    new_data = pd.concat([new_data, y], axis=1)
    new_data.to_csv(rf"D:\NJU\HW\Final_test\filter\{datalist[i]}",index=False)


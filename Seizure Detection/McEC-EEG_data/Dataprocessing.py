import numpy as np
import pandas as pd

from mne.io import RawArray, read_raw_edf
import mne

from sklearn.preprocessing import StandardScaler

from scipy import signal

import warnings
warnings.filterwarnings("ignore")     #忽略警告消息
import matplotlib.pyplot as plt
import random



path1 = 'D:\\JetBrains\\PyCharm 2023.2.4\\pyprogram\\pyproject\\JDdata\\new_data\\2_1.edf'
raw = read_raw_edf(path1)
# 加载数据到内存中
raw.load_data()
# #
# raw1 = raw.copy().crop(tmin=1, tmax=14399)
# raw1= raw1.to_data_frame()
# useless1_C11,raw1, useless1_C12= np.split(raw1, (1,22,), axis=1)
# print(raw1.shape)
# print(raw1)
np.random.seed(1)
# 添加滤波器
raw = raw.filter(0.53, 40., fir_design='firwin')
#
# raw3 = raw.copy().crop(tmin=6940, tmax=70100)
raw3 = raw.copy().crop(tmin=6965, tmax=7065)
raw3= raw3.to_data_frame()
useless1_C11,raw3, useless1_C12= np.split(raw3, (1,22,), axis=1)
print(raw3.shape)
print(raw3)



#1_Fp1
data1=raw3.iloc[0:256*100, 0]
print(data1.shape)
#print(data)
arr= data1.to_numpy()
df_split = np.array_split(arr, 100)

# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)
new_raw=pd.DataFrame(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data1= pd.DataFrame(x_data)
data1.to_csv('Singledata\\1_Fp1.csv')


#2_Fp2
data2=raw3.iloc[0:256*100, 1]
print(data2.shape)
#print(data)
arr= data2.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data

data2= pd.DataFrame(x_data)
data2.to_csv('Singledata\\2_Fp2.csv')

#3_F3
data3=raw3.iloc[0:256*100, 2]
print(data3.shape)
#print(data)
arr= data3.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data3= pd.DataFrame(x_data)
data3.to_csv('Singledata\\3_F3.csv')

#4_F4
data4=raw3.iloc[0:256*100, 3]
print(data4.shape)
#print(data)
arr= data4.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data4= pd.DataFrame(x_data)
data4.to_csv('Singledata\\4_F4.csv')

#5_C3
data5=raw3.iloc[0:256*100, 4]
print(data5.shape)
#print(data)
arr= data5.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data5= pd.DataFrame(x_data)
data5.to_csv('Singledata\\5_C3.csv')

#6_C4
data6=raw3.iloc[0:256*100, 5]
print(data6.shape)
#print(data)
arr= data6.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data6= pd.DataFrame(x_data)
data6.to_csv('Singledata\\6_C4.csv')

#7_P3
data7=raw3.iloc[0:256*100, 6]
print(data7.shape)
#print(data)
arr= data7.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data7= pd.DataFrame(x_data)
data7.to_csv('Singledata\\7_P3.csv')


#8_P4
data8=raw3.iloc[0:256*100, 7]
print(data8.shape)
#print(data)
arr= data8.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data8= pd.DataFrame(x_data)
data8.to_csv('Singledata\\8_P4.csv')

#9_O1
data9=raw3.iloc[0:256*100, 8]
print(data9.shape)
#print(data)
arr= data9.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data9= pd.DataFrame(x_data)
data9.to_csv('Singledata\\9_O1.csv')

#10_O2
data10=raw3.iloc[0:256*100, 9]
print(data10.shape)
#print(data)
arr= data10.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data10= pd.DataFrame(x_data)
data10.to_csv('Singledata\\10_O2.csv')

#11_F7
data11=raw3.iloc[0:256*100, 10]
print(data11.shape)
#print(data)
arr= data11.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data11= pd.DataFrame(x_data)
data11.to_csv('Singledata\\11_F7.csv')

#12_F8
data12=raw3.iloc[0:256*100, 11]
print(data12.shape)
#print(data)
arr= data12.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data12= pd.DataFrame(x_data)
data12.to_csv('Singledata\\12_F8.csv')

#13_T3
data13=raw3.iloc[0:256*100, 12]
print(data13.shape)
#print(data)
arr= data13.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data13= pd.DataFrame(x_data)
data13.to_csv('Singledata\\13_T3.csv')

#14_T4
data14=raw3.iloc[0:256*100, 13]
print(data14.shape)
#print(data)
arr= data14.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data14= pd.DataFrame(x_data)
data14.to_csv('Singledata\\14_T4.csv')

#15_T5
data15=raw3.iloc[0:256*100, 14]
print(data15.shape)
#print(data)
arr= data15.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data15= pd.DataFrame(x_data)
data15.to_csv('Singledata\\15_T5.csv')

#16_T6
data16=raw3.iloc[0:256*100, 15]
print(data16.shape)
#print(data)
arr= data16.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data16= pd.DataFrame(x_data)
data16.to_csv('Singledata\\16_T6.csv')

#17_A1
data17=raw3.iloc[0:256*100, 16]
print(data17.shape)
#print(data)
arr= data17.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data17= pd.DataFrame(x_data)
data17.to_csv('Singledata\\17_A1.csv')

#18_A2
data18=raw3.iloc[0:256*100, 17]
print(data18.shape)
#print(data)
arr= data18.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data18= pd.DataFrame(x_data)
data18.to_csv('Singledata\\18_A2.csv')

#19_Fz
data19=raw3.iloc[0:256*100, 18]
print(data19.shape)
#print(data)
arr= data19.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data19= pd.DataFrame(x_data)
data19.to_csv('Singledata\\19_Fz.csv')

#20_Cz
data20=raw3.iloc[0:256*100, 19]
print(data20.shape)
#print(data)
arr= data20.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data20= pd.DataFrame(x_data)
data20.to_csv('Singledata\\20_Cz.csv')

#21_Pz
data21=raw3.iloc[0:256*100, 20]
print(data21.shape)
#print(data)
arr= data21.to_numpy()
df_split = np.array_split(arr, 100)
# #改变采样率
#
new_raw = np.apply_along_axis(lambda x: signal.resample(x, 178), axis=1, arr=df_split)
print(new_raw.shape)
print(new_raw)

# # #归一化
scaler = StandardScaler()
normalized_array_x_data = scaler.fit_transform(new_raw)
normalized_array_x_data= pd.DataFrame(normalized_array_x_data)
normalized_df_x_data = pd.DataFrame(normalized_array_x_data, columns=normalized_array_x_data.columns)
print(normalized_array_x_data)
x_data=normalized_array_x_data
data21= pd.DataFrame(x_data)
data21.to_csv('Singledata\\21_Pz.csv')

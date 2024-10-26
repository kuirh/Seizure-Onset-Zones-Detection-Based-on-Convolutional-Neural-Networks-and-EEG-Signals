import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.saving.saving_api import load_model

from mne.io import RawArray, read_raw_edf
import mne

from sklearn.preprocessing import StandardScaler

from scipy import signal

import warnings


warnings.filterwarnings("ignore")     #忽略警告消息

model = load_model('D:\\JetBrains\\PyCharm 2023.2.4\\pyprogram\\pyproject\\SOZ\\Best_model_fold_7_val_recall_0.9974.h5')
# 1_Fp1
data_1_Fp1 ='Singledata\\1_Fp1.csv'
x_test_1_Fp1= pd.read_csv(data_1_Fp1, header=0, index_col=0)
y_pred = model.predict(x_test_1_Fp1)
cou=0
# print(y_pred)
for i in range(len(x_test_1_Fp1)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_1_Fp1= pd.DataFrame(y_pred)
data_1_Fp1.to_csv('Resultdata\\r1_Fp1.csv')

#2_Fp2
data_2_Fp2 ='Singledata\\2_Fp2.csv'
x_test_2_Fp2= pd.read_csv(data_2_Fp2, header=0, index_col=0)
y_pred = model.predict(x_test_2_Fp2)
cou=0
print(y_pred)
for i in range(len(x_test_2_Fp2)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
# print(y_pred)
print(cou)
data_2_Fp2= pd.DataFrame(y_pred)
data_2_Fp2.to_csv('Resultdata\\r2_Fp2.csv')

# 3_F3
data_3_F3='Singledata\\3_F3.csv'
x_test_3_F3= pd.read_csv(data_3_F3, header=0, index_col=0)
y_pred = model.predict(x_test_3_F3)
cou=0
# print(y_pred)
for i in range(len(x_test_3_F3)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_3_F3= pd.DataFrame(y_pred)
data_3_F3.to_csv('Resultdata\\r3_F3.csv')

# 4_F4
data_4_F4='Singledata\\4_F4.csv'
x_test_4_F4= pd.read_csv(data_4_F4, header=0, index_col=0)
y_pred = model.predict(x_test_4_F4)
cou=0
# print(y_pred)
for i in range(len(x_test_4_F4)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_4_F4= pd.DataFrame(y_pred)
data_4_F4.to_csv('Resultdata\\r4_F4.csv')

# 5_C3
data_5_C3='Singledata\\5_C3.csv'
x_test_5_C3= pd.read_csv(data_5_C3, header=0, index_col=0)
y_pred = model.predict(x_test_5_C3)
cou=0
# print(y_pred)
for i in range(len(x_test_5_C3)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_5_C3= pd.DataFrame(y_pred)
data_5_C3.to_csv('Resultdata\\r5_C3.csv')

# 6_C4
data_6_C4='Singledata\\6_C4.csv'
x_test_6_C4= pd.read_csv(data_6_C4, header=0, index_col=0)
y_pred = model.predict(x_test_6_C4)
cou=0
# print(y_pred)
for i in range(len(x_test_6_C4)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_6_C4= pd.DataFrame(y_pred)
data_6_C4.to_csv('Resultdata\\r6_C4.csv')

# 7_P3
data_7_P3='Singledata\\7_P3.csv'
x_test_7_P3= pd.read_csv(data_7_P3, header=0, index_col=0)
y_pred = model.predict(x_test_7_P3)
cou=0
# print(y_pred)
for i in range(len(x_test_7_P3)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_7_P3= pd.DataFrame(y_pred)
data_7_P3.to_csv('Resultdata\\r7_P3.csv')

# 8_P4
data_8_P4='Singledata\\8_P4.csv'
x_test_8_P4= pd.read_csv(data_8_P4, header=0, index_col=0)
y_pred = model.predict(x_test_8_P4)
cou=0
# print(y_pred)
for i in range(len(x_test_8_P4)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_8_P4= pd.DataFrame(y_pred)
data_8_P4.to_csv('Resultdata\\r8_P4.csv')

#9_O1
data_9_O1='Singledata\\9_O1.csv'
x_test_9_O1= pd.read_csv(data_9_O1, header=0, index_col=0)
y_pred = model.predict(x_test_9_O1)
cou=0
# print(y_pred)
for i in range(len(x_test_9_O1)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_9_O1= pd.DataFrame(y_pred)
data_9_O1.to_csv('Resultdata\\r9_O1.csv')

#10_O2
data_10_O2='Singledata\\10_O2.csv'
x_test_10_O2= pd.read_csv(data_10_O2, header=0, index_col=0)
y_pred = model.predict(x_test_10_O2)
cou=0
# print(y_pred)
for i in range(len(x_test_10_O2)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_10_O2= pd.DataFrame(y_pred)
data_10_O2.to_csv('Resultdata\\r10_O2.csv')

#11_F7
data_11_F7='Singledata\\11_F7.csv'
x_test_11_F7= pd.read_csv(data_11_F7, header=0, index_col=0)
y_pred = model.predict(x_test_11_F7)
cou=0
# print(y_pred)
for i in range(len(x_test_11_F7)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_11_F7= pd.DataFrame(y_pred)
data_11_F7.to_csv('Resultdata\\r11_F7.csv')

#12_F8
data_12_F8='Singledata\\12_F8.csv'
x_test_12_F8= pd.read_csv(data_12_F8, header=0, index_col=0)
y_pred = model.predict(x_test_12_F8)
cou=0
# print(y_pred)
for i in range(len(x_test_12_F8)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_12_F8= pd.DataFrame(y_pred)
data_12_F8.to_csv('Resultdata\\r12_F8.csv')

#13_T3
data_13_T3='Singledata\\13_T3.csv'
x_test_13_T3= pd.read_csv(data_13_T3, header=0, index_col=0)
y_pred = model.predict(x_test_13_T3)
cou=0
# print(y_pred)
for i in range(len(x_test_13_T3)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_13_T3= pd.DataFrame(y_pred)
data_13_T3.to_csv('Resultdata\\r13_T3.csv')

#14_T4
data_14_T4='Singledata\\14_T4.csv'
x_test_14_T4= pd.read_csv(data_14_T4, header=0, index_col=0)
y_pred = model.predict(x_test_14_T4)
cou=0
# print(y_pred)
for i in range(len(x_test_14_T4)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_14_T4= pd.DataFrame(y_pred)
data_14_T4.to_csv('Resultdata\\r14_T4.csv')

#15_T5
data_15_T5='Singledata\\15_T5.csv'
x_test_15_T5= pd.read_csv(data_15_T5, header=0, index_col=0)
y_pred = model.predict(x_test_15_T5)
cou=0
# print(y_pred)
for i in range(len(x_test_15_T5)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_15_T5= pd.DataFrame(y_pred)
data_15_T5.to_csv('Resultdata\\r15_T5.csv')

#16_T6
data_16_T6='Singledata\\16_T6.csv'
x_test_16_T6= pd.read_csv(data_16_T6, header=0, index_col=0)
y_pred = model.predict(x_test_16_T6)
cou=0
# print(y_pred)
for i in range(len(x_test_16_T6)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_16_T6= pd.DataFrame(y_pred)
data_16_T6.to_csv('Resultdata\\r16_T6.csv')

#17_A1
data_17_A1='Singledata\\17_A1.csv'
x_test_17_A1= pd.read_csv(data_17_A1, header=0, index_col=0)
y_pred = model.predict(x_test_17_A1)
cou=0
# print(y_pred)
for i in range(len(x_test_17_A1)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_17_A1= pd.DataFrame(y_pred)
data_17_A1.to_csv('Resultdata\\r17_A1.csv')

#18_A2
data_18_A2='Singledata\\18_A2.csv'
x_test_18_A2= pd.read_csv(data_18_A2, header=0, index_col=0)
y_pred = model.predict(x_test_18_A2)
cou=0
# print(y_pred)
for i in range(len(x_test_18_A2)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_18_A2= pd.DataFrame(y_pred)
data_18_A2.to_csv('Resultdata\\r18_A2.csv')

#19_Fz
data_19_Fz='Singledata\\19_Fz.csv'
x_test_19_Fz= pd.read_csv(data_19_Fz, header=0, index_col=0)
y_pred = model.predict(x_test_19_Fz)
cou=0
# print(y_pred)
for i in range(len(x_test_19_Fz)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_19_Fz= pd.DataFrame(y_pred)
data_19_Fz.to_csv('Resultdata\\r19_Fz.csv')

#20_Cz
data_20_Cz='Singledata\\20_Cz.csv'
x_test_20_Cz= pd.read_csv(data_20_Cz, header=0, index_col=0)
y_pred = model.predict(x_test_20_Cz)
cou=0
# print(y_pred)
for i in range(len(x_test_20_Cz)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_20_Cz= pd.DataFrame(y_pred)
data_20_Cz.to_csv('Resultdata\\r20_Cz.csv')

#21_Pz
data_21_Pz='Singledata\\21_Pz.csv'
x_test_21_Pz= pd.read_csv(data_21_Pz, header=0, index_col=0)
y_pred = model.predict(x_test_21_Pz)
cou=0
# print(y_pred)
for i in range(len(x_test_21_Pz)):
    if y_pred[i]<0.5:
        y_pred[i]=0
    else:
        y_pred[i]=1
        cou+=1
print(y_pred)
print(cou)
data_21_Pz= pd.DataFrame(y_pred)
data_21_Pz.to_csv('Resultdata\\r21_Pz.csv')

data = np.concatenate((data_1_Fp1,data_2_Fp2,data_3_F3,data_4_F4,data_5_C3,data_6_C4,
                       data_7_P3,data_8_P4,data_9_O1,data_10_O2,data_11_F7,data_12_F8,
                       data_13_T3,data_14_T4,data_15_T5,data_16_T6,data_17_A1,data_18_A2,
                       data_19_Fz,data_20_Cz,data_21_Pz),
                      axis=0)

data = pd.DataFrame(data)
arr = data.to_numpy()
df_split = np.array_split(arr, 21)
print(df_split)
list_shape = np.array(df_split).shape
print(list_shape)
data_array = np.array(df_split)
data_2d = data_array.reshape(21, -1)
print(data_2d.shape)
data = pd.DataFrame(data_2d)
data=data.T
print(data.shape)
data.to_csv('Resultdata\\1_Resultdata.csv')












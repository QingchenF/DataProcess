import numpy as np
import pandas as pd

file = pd.read_csv('./ABCD_CBCL_Label_z.csv')

General = file['Int']
#mean value
General_mean = np.mean(General)
print('mean = ',General_mean)
#variance
General_var = np.var(General)
print('var = ',General_var)
#STD
General_std = np.std(General)
print('General_std',General_std)

list_1 = []
for i in General:
    list_1.append((i - General_mean)/General_std)
print(list_1)

fw = open('./Int_Label_z.csv',mode='w')
for j in list_1:
    a = '%.4f' % j
    fw.write(a)
    fw.write('\n')
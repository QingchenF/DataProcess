import glob
import nibabel as nib
import numpy as np
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
#data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/ABCD_FC_10min/*.nii"),reverse=True)#读取数据
#label_files_all = pd.read_csv("/Users/fan/Documents/Data/ABCD_CBCL_L.csv")#读取标签文件
data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_nii/*.nii"),reverse=True)#读取数据
label_files_all = pd.read_csv("/Users/fan/Documents/Data/testt.csv")#读取标签文件
label = label_files_all['src_subject_id']#获取sub id那一列

print(data_files_all[0])
print(label[0])
files_data = []
for i in data_files_all:
    img_data = nib.load(i)
    img_data = img_data.get_data()
    files_data.append(img_data)

data = np.asarray(files_data)
data = np.around(data,decimals=4)
print(len(data),type(data),data.shape)

for j in range(len(data)):
    np.savetxt('/Users/fan/Documents/Data/test_nii/'+label[j]+'.txt',data[j])#np自带保存函数  #读取使用np.genfromtxt()
#for j in range(len(data)):
#    fw = open('/Users/fan/Documents/Data/test_nii/'+label[j]+'.txt', mode='w')
#    fw.write(str(data[j]))
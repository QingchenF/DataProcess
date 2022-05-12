import pandas as pd
import numpy as np

ddp = pd.read_csv('/Users/fan/Documents/Data/ABCD_dia/Substitute/Conduct_Disorder_present_childhood_onset:adolescent_onset.csv')
#获取ID
ddp_Id = ddp['subject_ID']

abcd_label = pd.read_csv('/Users/fan/Documents/Data/ABCD_dia/Substitute/abcd_baseline_z.csv')
#获取label文件的ID
abcd_Id = abcd_label['subject_id']
#获取label文件的"General_Z"，可以修改"列名字"不同的Z
abcd_ZG = abcd_label['Int_Z']
#创建个字典，装id和Z-value
abcd_Id_ZG = {}
#封装一个函数：遍历生成列表
def makebox(list):
    list_box = []
    for i in list:
        list_box.append(i)
    return list_box
ddp_Id_list = makebox(ddp_Id)
abcd_Id_list = makebox(abcd_Id)
abcd_ZG_list = makebox(abcd_ZG)
#将label中的id和Z-value添加到字典
for i,j in zip(abcd_Id_list,abcd_ZG_list):
    abcd_Id_ZG[i] = j
#遍历源文件，找到源文件中id在label文件中的key(z-value)。添加列表里
list_Zvalue = []
for m in ddp_Id_list:
    list_Zvalue.append(abcd_Id_ZG[m])

fw = open('./Conduct_Disorder_present_childhood_onset:adolescent_onset_ADHD_Z.csv',mode='w')
for k in list_Zvalue:
    fw.write(str(k))
    fw.write('\n')

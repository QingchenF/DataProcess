import pandas as pd

# data_1 = pd.read_csv('./factor_score_wx.csv')
#
# data_2 = pd.read_csv('./ABCD_CBCL_FQC.csv')
#较多的数据集
with open('./abcd_cbcl_baseline_label.csv',mode='r') as data_1:
    data_1 = data_1.readlines()
#较少的数据集
with open('./ABCD_Label.csv',mode='r')as data_2:
    data_2 = data_2.readlines()

list_f = []
for i in data_1:
    list_f.append(i)

list_2 = []
for i in data_2:
    list_2.append(i[0:16])

list_1 = []
for j in data_1:
    list_1.append(j[0:16])


list_index = []
for m in list_2:
    if m in list_1:
        list_index.append(list_1.index(m))

fw = open("./ABCD_CBCL_Label_age.csv",mode='w')

for i in list_index:
    #fw.write('\n')

    fw.write(list_f[i])








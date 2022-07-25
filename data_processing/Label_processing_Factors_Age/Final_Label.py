import pandas as pd

# data_1 = pd.read_csv('./factor_score_wx.csv')
#
# data_2 = pd.read_csv('./ABCD_CBCL_FQC.csv')

with open('./factor_score_wx.csv',mode='r') as data_1:
    data_1 = data_1.readlines()

with open('./ABCD_CBCL_FQC.csv',mode='r')as data_2:
    data_2 = data_2.readlines()
print(data_1,type(data_1))
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

fw = open("./ABCD_CBCL_Label.csv",mode='w')

for i in list_index:
    #fw.write('\n')

    fw.write(list_f[i])








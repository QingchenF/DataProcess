import pandas as pd
fw = open('./res.csv',mode='w')
fw_2 = open('./res_2_int.csv',mode='w')

with open('./data_all.csv') as f1:
    file_name = f1.readlines()

list_1 = []
for i in file_name:
    list_1.append(i[40:51])

print(list_1)

with open("./abcd_cbcl_baseline_label.csv") as f2:
    file_label = f2.readlines()
list_2 = []
for j in file_label:
    list_2.append(j)
#print(pd.DataFrame(list_2))
list_3 = []
for i in list_2:
    a = i.split()
    list_3.append(a[0].split(','))
list_4 = []
for i in list_3 :
    list_4.append(i[0])
list_5 = []
for i in list_1:
    if i in list_4:
        list_5.append(list_4.index(i))
for i in list_5 :
    #print(list_3[i][75])
    fw_2.write(list_3[i][77])
    fw_2.write('\n')


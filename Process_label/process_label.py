fw = open("/Users/fan/Desktop/label_file_abcd.csv",mode='w')
with open('./abcd_cbcl_baseline.csv') as file_abc:
    list_abcd = file_abc.readlines()
list_1 = []
for i in list_abcd:
    list_1.append(i[5:16])
print(list_1)

with open('./file_name.csv') as file_name:
    list_name = file_name.readlines()
list_2 = []
for i in list_name:
    list_2.append(i[8:19])
print(list_2)
print(len(list_2))

list_3 = []
for m in list_2:
    if m not in list_1:
        print(m)
       # list_3.append(list_1.index(m))
        list_3.append(m)
print(len(list_3),list_3)

#for n in list_3:
 #   fw.write(list_abcd[n])


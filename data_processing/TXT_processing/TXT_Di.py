
fw = open("/Users/fan/Desktop/unique_all.csv",mode='w')

with open("./id_res_all.csv",mode='r') as f1:
    list_max= f1.readlines()

with open("./id_res_pconn.csv", mode='r') as f2:
    list_min = f2.readlines()

for item in list_max:
    if item not in list_min:
        fw.write(item)

fw.close()
'''
fw = open("/Users/fan/Desktop/unique_1.csv",mode='w')
f = open('id_res.csv')
f2 = open('id_res_pconn.csv')
list1 = []
list2 = []
for i in f:
    list1.append(i)

for j in f2:
    list2.append(j)

for m in list1:
    if m not in list2:
        fw.write(m)
'''


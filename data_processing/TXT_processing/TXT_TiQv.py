
data_10min_fc_ID = open("/Users/fan/Desktop/id_res_all.csv",mode='w+')

f = open('data_10min_fc.txt')

#line = f.readline() # 以行的形式进行读取文件
'''
#print(line)
a = line.split()
b = a[3]
c = b[5:20]
#print(b)
#print(c)
'''
list1 = []
for line in f:
    #a = line.split()
    # b = a[3]
   #  c = b[5:20]
    list1.append(line)   # 将其添加在列表之中
print('源数据长度：',len(list1))
res = list(set(list1))
res.sort(key = list1.index)
print('处理后长度：',len(res))
f.close()

for i in res:
  data_10min_fc_ID.write(i)
  data_10min_fc_ID.write('\n')
  print(i)
def  a(b):
    if b == 0:
        return
        print('nono')
    else:
        print('yes')
a(1)
from sklearn.decomposition import PCA
PCA()
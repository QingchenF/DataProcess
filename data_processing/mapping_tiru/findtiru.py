import numpy as np

a = np.array([[1,2,3,6],[4,5,6,8],[7,8,9,2],[6,4,5,3]])  #源数据 矩阵
print('--原矩阵--\n',a)
m = a.shape[0]
r, c = np.triu_indices(m, 1)
matirx = a[r,c]  #拉平之后
res_1, res_2 = np.triu_indices_from(a,k=1) #生成的上三角，在源数据中的索引位置res_1是行，res_2是列（注意索引是从0开始）

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  #权重肯定也是一个拉平的矩阵 1 x *大小
p = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]  #计算的p值肯定也是1 x *大小
#下面找到p值大于0.05的索引
index_list = []
for i ,j in enumerate(p) :
    if j>4:
        index_list.append(i)
#遍历"p值大于0.05的索引"，找到对应值在源数据中的位置。因为p/w的索引位置（拉平后的矩阵）和res_1、res_2保持一致
feature = []
for i in index_list:
    list = []
    row = res_1[i]
    col = res_2[i]
    list.append(row+1)#加一是为了和脑区数值保持一致
    list.append(col+1)
    feature.append(list)
print(feature)




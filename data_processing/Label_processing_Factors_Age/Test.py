import pandas as pd

# data_1 = pd.read_csv('./factor_score_wx.csv')
# #
# data_2 = pd.read_csv('./ABCD_CBCL_FQC.csv')
# print(data_1.head())
# print(data_1[data_1['subjectkey']=='NDAR_INV003RTV85'])
# list_1 = []
# print('NDAR_INV003RTV85' in data_1['subjectkey'].values)
#
# fw = open("./test.csv",mode='w')
# list_res = []
# for i in data_2['subjectkey']:
#     #print(data_1[data_1['subjectkey'] == i])
#      list_res.append(pd.DataFrame(data_1[data_1['subjectkey'] == i]).values.flatten())
#
# df = pd.DataFrame(list_res,columns=['subjectkey','General','Ext','ADHD','Int'])
# #print(df.head())
# df.to_csv('./test2.csv')



with open('./ABCD_Label.csv',mode='r') as data_1:
    data_1 = data_1.readlines()

list_1 = []
for j in data_1:
    list_1.append(j[0:16])


with open('./ABCD_CBCL_Label_age.csv',mode='r')as data_2:
    data_2 = data_2.readlines()

list_2 = []
for i in data_2:
    list_2.append(i[0:16])



for i in list_1:
    if i not in list_2:
        print(i)



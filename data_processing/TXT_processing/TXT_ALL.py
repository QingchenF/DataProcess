# fw  = open("/Users/fan/Desktop/path.csv",mode='w')
#
# with open("./unique_1.csv",mode='r') as f1:
#     list1 = f1.readlines()
# print(len(list1))
#
# list_path = []
# with open("./data_10min_fc.txt",mode="r") as f2:
#     for line in f2:
#         #print(line)
#         if line.split()[-1][1:-1].endswith(".txt"):
#              #print(line.split()[-1][1:-1])
#              list_path.append(line.split()[-1][1:-1])
# print(list_path[0])
#
# for id in list1:
#    # print(id)
#     for path in list_path:
#         if id in path:
#             print("000")
#             fw.write(path)
#
# fw.close()
import struct
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingRegressor
import joblib
from sklearn.metrics import r2_score
Times = 10
dimention = 'General'
import pandas as pd

data = pd.read_csv("./Res.csv")


res = data.iloc[:,:]


#print(res.iloc[:,1:2],type(res))
data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_data/test_train/*.nii"))
label_files_all = pd.read_csv("/Users/fan/Documents/Data/test_data/test_train/test.csv")
label = label_files_all[dimention]

#Label
y_label = np.array(label)

#All data
files_data = []
for i in data_files_all:
    img_data = nib.load(i)
    img_data = img_data.get_data()
    files_data.append(img_data)
p = 0
for i in files_data:
    p = p + 1
    print('----p----',p)
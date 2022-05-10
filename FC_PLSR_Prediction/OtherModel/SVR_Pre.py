import numpy as np
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

import os
#Loading Data

data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/ABCD_FC_10min/*.nii"),reverse=True)
label_files_all = pd.read_csv("/Users/fan/Documents/Data/ABCD_CBCL_L.csv")

#data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_train/*.nii"))
#label_files_all = pd.read_csv("/Users/fan/Documents/Data/test_train/test.csv")
data_all = open("./data_all.csv",mode='w')

for p in data_files_all:
    data_all.write(p)
    data_all.write('\n')

label = label_files_all['Int']

X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=0)

f_xtrain = open("./Other_Model_Res/train_set",mode='w')
for i in X_train:
    f_xtrain.write(i)
    f_xtrain.write('\n')

f_ytrain = open("./Other_Model_Res/train_y",mode='w')
for j in y_train:
    f_ytrain.write(str(j))
    f_ytrain.write('\n')

f_xtest = open("./Other_Model_Res/test_set",mode='w')
for n in X_test:
    f_xtest.write(n)
    f_xtest.write('\n')

f_ytest = open("./Other_Model_Res/test_y",mode='w')
for m in y_test:
    f_ytest.write(str(m))
    f_ytest.write('\n')
Train_label = np.array(y_train)
Test_label = np.array(y_test)

#Define a function that takes the upper triangle
def upper_tri_indexing(matirx):
    m = matirx.shape[0]
    r,c = np.triu_indices(m,1)
    return matirx[r,c]
#Train
files = X_train[:]
files_data = []
for i in files:
    img_data = nib.load(i)
    img_data = img_data.get_data()
    img_data_reshape = upper_tri_indexing(img_data)
    files_data.append(img_data_reshape)

Train_data = np.asarray(files_data)

#Test Data
Test_files = X_test[:]
Test_list = []
for j in Test_files:
    test_data = nib.load(j)
    test_data = test_data.get_data()
    test_data_reshape = upper_tri_indexing(test_data)
    Test_list.append(test_data_reshape)
Test_data = np.asarray(Test_list)

#Train_data Train_label  Test_data Test_label
#Model
predict_model = SVR()
predict_model.fit(Train_data,Train_label)

Predict_Score = predict_model.predict(Test_data)

Corr = np.corrcoef(Predict_Score.T,Test_label)
MAE_inv =  np.mean(np.abs(Predict_Score - Test_label))
print('Prediction Result\n',Predict_Score)
print('Correlation\n',Corr)
print('MAE:',MAE_inv)

fw = open("./Other_Model_Res/Predict_Score_Int.csv",mode='w')
for l in Predict_Score:
    fw.write(str(l))
    fw.write('\n')


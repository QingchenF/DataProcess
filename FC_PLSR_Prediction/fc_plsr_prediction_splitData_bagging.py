import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  BaggingRegressor

#Loading Data
data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/ABCD_FC_10min/*.nii"),reverse=True)
label_files_all = pd.read_csv("/Users/fan/Documents/Data/ABCD_CBCL_L.csv")
label = label_files_all['Int']

X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=1)

f_xtrain = open("./Note_Res_bagging/train_set_b.csv",mode='w')
for i in X_train:
    f_xtrain.write(i)
    f_xtrain.write('\n')

f_ytrain = open("./Note_Res_bagging/train_y_b.csv",mode='w')
for j in y_train:
    f_ytrain.write(str(j))
    f_ytrain.write('\n')

f_xtest = open("./Note_Res_bagging/test_set_b.csv",mode='w')
for n in X_test:
    f_xtest.write(n)
    f_xtest.write('\n')

f_ytest = open("./Note_Res_bagging/test_y_b.csv",mode='w')
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

#Model
#plsr = PLSRegression()

#bagging,基分类器PLS
bagging = BaggingRegressor(base_estimator=PLSRegression())

#网格交叉验证
param_grid = {'n_estimators':[1,2,3,4,5,6,7,8,9,10]}
predict_model = GridSearchCV(bagging,param_grid,cv=5)
predict_model.fit(Train_data,Train_label)

print("----best_estimator-----",predict_model.best_estimator_)
Predict_Score = predict_model.predict(Test_data)
Predict_Score_new = np.transpose(Predict_Score)
Corr = np.corrcoef(Predict_Score_new,Test_label)

#plsr.fit(Train_data,Train_label)
#Predict_Score = plsr.predict(Test_data)
#Predict_Score = np.transpose(Predict_Score)
#print(Predict_Score,Predict_Score.shape)
#print(Test_label,Test_label.shape)
#Corr = np.corrcoef(Predict_Score,Test_label)

MAE_inv =  np.mean(np.abs(Predict_Score - Test_label))
print('Prediction Result\n',Predict_Score)
print('Correlation\n',Corr)
print('MAE:',MAE_inv)

fw = open("./Note_Res_bagging/Predict_Score_Bagging.csv",mode='w')
for l in Predict_Score:
    fw.write(str(l))
    fw.write('\n')


import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
import ToolBox.ToolBox as tb


#Loading Data
data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/ABCD_FC_10min/*.nii"),reverse=True)
label_files_all = pd.read_csv("/Users/fan/Documents/Data/ABCD_CBCL_L.csv")

#data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_train/*.nii"))
#label_files_all = pd.read_csv("/Users/fan/Documents/Data/test_train/test.csv")
tb.ToolboxCSV('data_all.csv',data_files_all)

print(label_files_all)
label = label_files_all['Conduct']

X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=0)


tb.ToolboxCSV('train_set_test.csv',X_train)
tb.ToolboxCSV('train_y_test.csv',y_train)
tb.ToolboxCSV('test_set_test.csv',X_test)
tb.ToolboxCSV('test_y_test.csv',y_test)


'''
f_xtrain = open("./Note_Res/train_set",mode='w')
for i in X_train:
    f_xtrain.write(i)
    f_xtrain.write('\n')

f_ytrain = open("./Note_Res/train_y",mode='w')
for j in y_train:
    f_ytrain.write(str(j))
    f_ytrain.write('\n')

f_xtest = open("./Note_Res/test_set",mode='w')
for n in X_test:
    f_xtest.write(n)
    f_xtest.write('\n')

f_ytest = open("./Note_Res/test_y",mode='w')
for m in y_test:
    f_ytest.write(str(m))
    f_ytest.write('\n')
'''
Train_label = np.array(y_train)
Test_label = np.array(y_test)

#Train
files = X_train[:]
files_data = []
for i in files:
    img_data = nib.load(i)
    img_data = img_data.get_data()
    img_data_reshape = tb.upper_tri_indexing(img_data)
    files_data.append(img_data_reshape)

Train_data = np.asarray(files_data)

#Test Data
Test_files = X_test[:]
Test_list = []
for j in Test_files:
    test_data = nib.load(j)
    test_data = test_data.get_data()
    test_data_reshape = tb.upper_tri_indexing(test_data)
    Test_list.append(test_data_reshape)
Test_data = np.asarray(Test_list)


#Model
plsr = PLSRegression()
#GridSearchCV
param_grid = {'n_components':[1,2,3,4,5,6,7,8,9,10]}
predict_model = GridSearchCV(plsr,param_grid,cv=5)
predict_model.fit(Train_data,Train_label)
Predict_Score = predict_model.predict(Test_data)
#save model param
joblib.dump(predict_model,"pls_model.pkl")

Corr = np.corrcoef(Predict_Score.T,Test_label)
MAE_inv = np.mean(np.abs(Predict_Score - Test_label))
print('Prediction Result\n',Predict_Score)
print('Correlation\n',Corr)
print('MAE:',MAE_inv)

fw = open("./Predict_Score_Conduct.csv",mode='w')
for l in Predict_Score:
    fw.write(str(l))
    fw.write('\n')



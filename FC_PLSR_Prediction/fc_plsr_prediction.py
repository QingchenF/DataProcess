import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
#Loading Data
Train_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_train/*.nii"))
label_all = pd.read_csv("/Users/fan/Documents/Data/test_train/test.csv")
Train_label = label_all['General']
a = np.array(Train_label)
print(a,type(a))
print(Train_label,type(Train_label))
Test_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_test/*.nii"))
Test_label_file = pd.read_csv("/Users/fan/Documents/Data/test_test/test_test.csv")
Test_label = Test_label_file['General']
Test_label = np.array(Test_label)
print(Test_label)
#Train Data
files = Train_files_all[:]
files_data = []
for i in files:
    img_data = nib.load(i)
    img_data = img_data.get_data()
    img_data_reshape = img_data[np.triu_indices(352)] #拉成一行
    files_data.append(img_data_reshape)

Train_data = np.asarray(files_data)
print(Train_data.shape) #(8,62128)

#Test Data
Test_files = Test_files_all[:]
Test_list = []
for j in Test_files:
    test_data = nib.load(j)
    test_data = test_data.get_data()
    test_data_reshape = test_data[np.triu_indices(352)]
    Test_list.append(test_data_reshape)
Test_data = np.asarray(Test_list)

#Model
plsr = PLSRegression()
plsr.fit(Train_data,Train_label)
Predict_Score = plsr.predict(Test_data)
#Predict_Score = np.transpose(Predict_Score)
print(Predict_Score,Predict_Score.shape,type(Predict_Score),Test_label,Test_label.shape)
Corr = np.corrcoef(Predict_Score.T,Test_label)
print('Test_label:',Test_label)

MAE_inv =  np.mean(np.abs(Predict_Score - Test_label))
print('Prediction Result\n',Predict_Score)
print('Correlation \n',Corr)
print('MAE:',MAE_inv)


fw = open("./Predict_Score.csv",mode='w')
for l in Predict_Score:
    print(str(l))
    fw.write(str(l))
    fw.write('\n')
    #res = list_res.append(l)
    #print(str(Predict_Score[l]))
    #fw.write(str(Predict_Score[l]))

#print('X_train',X_train)
#print('y_train',y_train)
#print('X_test',X_test)
#print('y_test',y_test)

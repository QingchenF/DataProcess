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
import ToolBox as tb
import xgboost as xgb
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Loading Data
data_files_all = sorted(glob.glob("/home/cuizaixu_lab/fanqingchen/DATA/ABCD_FC_10min/*.nii"),reverse=True)
label_files_all = pd.read_csv("/home/cuizaixu_lab/fanqingchen/DATA/ABCD_FC_10min/ABCD_CBCL_L.csv")
#data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_train/*.nii"))
#label_files_all = pd.read_csv("/Users/fan/Documents/Data/test_train/test.csv")
tb.ToolboxCSV_server('data_all.csv',data_files_all)

print(label_files_all)
label = label_files_all['General']

X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=0)

tb.ToolboxCSV_server('train_set_x.csv',X_train)
tb.ToolboxCSV_server('train_y_x.csv',y_train)
tb.ToolboxCSV_server('test_set_x.csv',X_test)
tb.ToolboxCSV_server('test_y_x.csv',y_test)

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
Hyper_param = {'max_depth':range(3,10,2)}
predict_model = GridSearchCV(estimator=xgb.XGBRegressor(booster='gbtree',learning_rate=0.1, n_estimators=160, verbosity=0, objective='reg:linear'),
                             param_grid=Hyper_param,
                             scoring='neg_mean_absolute_error',
                             #n_jobs=4,
                             verbose=1,
                             cv=5)
predict_model.fit(Train_data,Train_label)
Predict_Score = predict_model.predict(Test_data)

print("-best_estimator-",predict_model.best_estimator_,"-",
      "-best_params-",   predict_model.best_params_,"-",
      "-best_score-",    predict_model.best_score_
      )
#save model param
joblib.dump(predict_model, "./pls_model.pkl")

Corr = np.corrcoef(Predict_Score.T,Test_label)
MAE_inv = np.mean(np.abs(Predict_Score - Test_label))
print('Prediction Result\n',Predict_Score)
print('Correlation\n',Corr)
print('MAE:',MAE_inv)

tb.ToolboxCSV_server('Predict_Score_General.csv',Predict_Score)



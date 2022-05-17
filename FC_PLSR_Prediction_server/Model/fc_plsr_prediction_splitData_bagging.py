
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
import ToolBox as tb
import joblib
from datetime import datetime
try:
    #Loading Data
    data_files_all = sorted(glob.glob("/home/cuizaixu_lab/fanqingchen/DATA/ABCD_FC_10min/*.nii"),reverse=True)
    label_files_all = pd.read_csv("/home/cuizaixu_lab/fanqingchen/DATA/ABCD_FC_10min/ABCD_CBCL_L.csv")
    label = label_files_all['General']

    X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=1)

    tb.ToolboxCSV_server('train_set_b.csv',X_train)
    tb.ToolboxCSV_server('train_y_b.csv',y_train)
    tb.ToolboxCSV_server('test_set_b.csv',X_test)
    tb.ToolboxCSV_server('test_y_b.csv',y_test)

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
    #plsr = PLSRegression()

    #bagging,基分类器PLS
    bagging = BaggingRegressor(base_estimator=PLSRegression())

    #网格交叉验证
    cv_times = 5
    param_grid = {'n_estimators':[1,2,3,4,5,6,7,8,9,10]}
    predict_model = GridSearchCV(bagging,param_grid,verbose=6,cv=cv_times)
    predict_model.fit(Train_data,Train_label)

    print("-best_estimator-",predict_model.best_estimator_,"-",
          "-best_params-",   predict_model.best_params_,"-",
          "-best_score-",    predict_model.best_score_
          )
    joblib.dump(predict_model, "./pls_bagging_model_General.pkl")

    Predict_Score = predict_model.predict(Test_data)
    Predict_Score_new = np.transpose(Predict_Score)
    Corr = np.corrcoef(Predict_Score_new,Test_label)

    MAE_inv =  np.mean(np.abs(Predict_Score - Test_label))
    print('Prediction Result\n',Predict_Score)
    print('Correlation\n',Corr)
    print('MAE:',MAE_inv)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Method:PLSRegression_bagging\n')

    res = 'MAE: ' + str(MAE_inv) + '\
             \nR: ' + str(round(Corr[0, 1], 4)) + '\
             \nCV_times: ' + str(cv_times) + ' \
             \nMethod: PLSRegression-bagging'
    tb.send_result_Ding(res)
    tb.ToolboxCSV_server('Predict_Score_General_bagging.csv',Predict_Score)
except BaseException as e:
    print(e)
    tb.send_warning_Ding(e)
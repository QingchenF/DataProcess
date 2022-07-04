import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingRegressor
import FC_PLSR_Prediction.ToolBox.ToolBox as tb
import joblib
Times = 10
dimention = 'General'    #['General', 'Int', 'ADHD', 'Ext'] #[]
#Loading Data
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
    img_data_reshape = tb.upper_tri_indexing(img_data)
    files_data.append(img_data_reshape)
data_files_all = np.array(data_files_all)
x_data = np.asarray(files_data)

epoch = 0
outer_results = []
outer_results_mae = []
kf = KFold(n_splits=2, shuffle=True, random_state=22)
for train_index,test_index in kf.split(x_data):
    epoch = epoch + 1
    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    y_train, y_test = y_label[train_index], y_label[test_index]
    tb.ToolboxCSV('train_set_bagging_' + dimention + '_' + str(Times)+'_' +str(epoch)+ '.csv', X_train)
    tb.ToolboxCSV('train_label_bagging_' + dimention + '_' + str(Times)+'_' +str(epoch)+ '.csv', y_train)
    tb.ToolboxCSV('test_set_bagging_' + dimention + '_' + str(Times)+'_' +str(epoch)+ '.csv', X_test)
    tb.ToolboxCSV('test_label_bagging_' + dimention + '_' + str(Times)+'_' +str(epoch)+ '.csv', y_test)
    #Model
    #bagging,PLS
    bagging = BaggingRegressor(base_estimator=PLSRegression())

    #网格交叉验证
    cv_times = 2
    param_grid = {'n_estimators':[1,2,3,4,5,6,7,8,9,10]}
    predict_model = GridSearchCV(bagging, param_grid, verbose=6, cv=cv_times)
    predict_model.fit(X_train, y_train)
    best_model = predict_model.best_estimator_
    Predict_Score = best_model.predict(X_test)
    #modelweight = best_model.coef0
    #print('--modelweight--', modelweight)
    # joblib.dump(predict_model,
    #             "/home/cuizaixu_lab/fanqingchen/DATA/Res/model_weight/pls_bagging_model_" + dimention + '_' + str(
    #                 Times) +str(epoch)+ ".pkl")
    tb.ToolboxCSV('Predict_Score_bagging_' + dimention + '_' + str(Times) +'_'+ str(epoch)+'.csv', Predict_Score)

    Predict_Score_new = np.transpose(Predict_Score)
    Corr = np.corrcoef(Predict_Score_new, y_test)
    Corr = Corr[0,1]

    outer_results.append(Corr)
    MAE_inv = round(np.mean(np.abs(Predict_Score - y_test)), 4)
    outer_results_mae.append(MAE_inv)
    print('>Corr=%.3f, MAE=%.3f, est=%.3f, cfg=%s' % (Corr, MAE_inv, predict_model.best_score_, predict_model.best_params_))
print('Result: R=%.3f ,MAE=%.3f' % (np.mean(outer_results), np.mean(outer_results_mae)))


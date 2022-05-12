import joblib
import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold,train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  BaggingRegressor
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
#Loading Data
data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_train/*.nii"),reverse=True)
label_files_all = pd.read_csv("/Users/fan/Documents/Data/test_train/test.csv")
label = label_files_all['General']

X_train, X_test, y_train, y_test = train_test_split(data_files_all,label,test_size=0.2,random_state=0)


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
Hyper_param = {'max_depth':range(3,10,2)}
predict_model = GridSearchCV(estimator=xgb.XGBRegressor(booster='gbtree',learning_rate=0.1, n_estimators=160, verbosity=1,objective='reg:squarederror'),
                             param_grid=Hyper_param,
                             scoring='neg_mean_absolute_error',
                             verbose=1,
                             cv=5)
predict_model.fit(Train_data,Train_label)
Predict_Score = predict_model.predict(Test_data)

print("-best_estimator-",predict_model.best_estimator_,"-",
      "-best_params-",   predict_model.best_params_,"-",
      "-best_score-",    predict_model.best_score_
      )
print('Predict_Score:',Predict_Score)
'''
#原生态xgboost使用该参数，调sklearn不用设置这个params
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

dtrain = xgb.DMatrix(Train_data, Train_label)
print(dtrain,type(dtrain))
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(Test_data)
ans = model.predict(dtest)
print(ans)
'''
'''
kf = KFold(n_splits=2,shuffle=True,random_state=1)
for Train_data_index,Test_data_index in kf.split(Train_data,Train_label):
     predict_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
     predict_model.fit(Train_data[Train_data_index],Train_label[Train_data_index])
     Predict_Score = predict_model.predict(Test_data[Test_data_index])
     
'''
'''
predict_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
predict_model.fit(Train_data,Train_label)
Predict_Score = predict_model.predict(Test_data)
plot_importance(predict_model)
plt.show()
# 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，即是二进制方式,DT是模型名字
#pickle.dump(DT,open("pls_model.dat","wb"))   # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据

# 加载模型
#loaded_model = joblib.load("./pls_model.pkl")

# 使用模型,对测试数据集进行预测
#Predict_Score = loaded_model.predict(Test_data)

'''
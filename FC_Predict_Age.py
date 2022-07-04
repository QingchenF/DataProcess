import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate,cross_val_score
from sklearn import svm,preprocessing
import sklearn
import math
import openpyxl
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve,accuracy_score,r2_score,mean_absolute_error
import  joblib
import matplotlib
from sklearn.preprocessing import  MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,RidgeClassifier
matplotlib.__version__
import scipy as sp
import seaborn as sns
import glob
from scipy import stats
import random
import lmdb
import pickle
import pylab

def train_cpm(ipmat, pheno, regions, pthr):
    print('----ipmat-shape--------', ipmat.shape)  # (60516, 80)）
    print('----pheno-shape--------', pheno.shape)  # (80,)
    print('----pheno--------', pheno)
    print('----regions--------', regions)  # 246

    cc = [stats.pearsonr(pheno, im) for im in ipmat]  # im=(80,),8个人的第一个功能连接与8个age的向量做相关，得到246个值
    print('-------cc--', np.array(cc))  # (60156,2)#每一行是一个功能连接，第一列是person值，第二列是p值，p越小相关性越显著

    rmat = np.array([c[0] for c in cc])
    pmat = np.array([c[1] for c in cc])

    rmat = np.reshape(rmat, [regions, regions])  # 246x246，r值，皮尔森相关系数
    pmat = np.reshape(pmat, [regions, regions])  # 246x246，p值，双尾p值，p越小越相关性越显著

    posedges = (rmat > 0) & (pmat < pthr)  # 正相关特征[true,false]
    posedges = posedges.astype(int)  # 转为[1,0](246,246)
    print('------posedges-------', posedges.shape)

    negedges = (rmat < 0) & (pmat < pthr)  # 负相关特征
    negedges = negedges.astype(int)

    edges = (rmat != 0) & (pmat < pthr)  # 相关
    edges = edges.astype(int)

    pe = ipmat[posedges.flatten().astype(bool), :]  # 对原始特征矩阵ipmat进行特征筛选（60516，80），列都取，行只取posedges为true的行，说明p值小于阈值
    ne = ipmat[negedges.flatten().astype(bool), :]
    ed = ipmat[edges.flatten().astype(bool), :]  # (30014, 80)

    print('------ed---------', ed.shape)

    pe = pe.sum(axis=0) / 2  # 把特征加起来除2
    ne = ne.sum(axis=0) / 2
    ed = ed.sum(axis=0) / 2  # (80，)最后特征矩阵，行全都加起来
    print('------ed-add---------', ed.shape)
    return  posedges, negedges, edges,ed

feature = pd.read_csv('./fun_data.csv',index_col=0)
print('#######festure-shape##########',feature.shape)#(103, 60516)

target = pd.read_csv('./age.csv',index_col=0)# (103,)
print('######target-shape###########',np.array(target).flatten().shape)

# feature = preprocessing.scale(np.array(feature))
M = MinMaxScaler()
feature = M.fit_transform(np.array(feature))
print('-------st--',feature)
x_train, x_test, y_train, y_test = train_test_split(feature, np.array(target).flatten(), test_size=0.2,random_state=42)

x_train = np.array(x_train).transpose(1,0)
y_train = np.array(y_train)
posedges,negedges,edges,eded = train_cpm(x_train,y_train,246,0.05)
print('-------eded-----------',eded.shape)
print('------y_train------',y_train.shape)

x_test =np.array(x_test).transpose(1,0)
y_test = np.array(y_test)

pe=np.sum(x_test[posedges.flatten().astype(bool),:], axis=0)/2#上下相加起来，{20，}，一个用户功能连接矩阵形成一个值
ne=np.sum(x_test[negedges.flatten().astype(bool),:], axis=0)/2
ed=np.sum(x_test[edges.flatten().astype(bool),:], axis=0)/2

eded = eded.reshape(-1,1)
best_score = -100
best_c = {}
for n in [-5,-4,-3,-2,-1,0,1,2,3]:
    svr = svm.SVR(kernel='linear', C=math.pow(10,n))
    score = cross_val_score(svr, eded, y_train, cv=5)
    score = score.mean()
    # print(score)
    if score>best_score:
        best_score = score
        best_c = {"C":math.pow(10,n)}
print('----------best_c------',best_c)
svr = svm.SVR(kernel='linear', C=best_c['C'])

print('-----eded.shape------',eded.shape)
cv_results = cross_validate(svr, eded, y_train, cv=5, return_estimator=True,return_train_score=True,scoring='r2')
print('----------cv_results------',cv_results)

svr.fit(eded,y_train)

ed = ed.reshape(-1,1)
print('-------ed.shape-----',ed.shape)

train_pre = svr.predict(eded)
pre = svr.predict(ed)

score = []
score.append(pre)
score.append(y_test)
score_np = np.array(score).transpose(1,0)
pre_score = pd.DataFrame(score_np)
pre_score.to_csv('./pre_score.csv')
'''
print('#######pre_true_r############',scipy.stats.pearsonr(y_test,pre))
print('#######test_r2############',r2_score(y_test,pre))
print('#######test_mae############',mean_absolute_error(pre,y_test))
print('#######train_mae############',mean_absolute_error(y_train,train_pre))
print('#######train_r2############',r2_score(train_pre,y_train))
'''
# best_score = -100
# best_c = {}
# for c in range(1,5):
#     svr = svm.SVR(kernel='linear', C=c)
#     score = cross_val_score(svr, x_train, y_train, cv=10)
#     score = score.mean()
#     print(score)
#     if score>best_score:
#         best_score =score
#         best_c = {"C":c}
# print('----------c--',best_c)
# svr = svm.SVR(kernel='linear', C=best_c['C'])
# cv_results = cross_validate(svr, x_train, y_train, cv=10, return_estimator=True,return_train_score=True)
# cv=0
# print(cv_results)
# for model in cv_results['estimator']:
#     print(model)
#     cv=cv+1
#     print(model.coef_.shape)
#     coef = model.coef_
#     coef = coef.flatten()
#     print(coef.shape)
#     coef_list = coef.tolist()
#     df = pd.DataFrame(coef_list)
#     # file_name = 'c_l_78_'+str(cv)+'.csv'
#     # df.to_csv('./dataset/' + file_name)
#
# svr.fit(x_train,y_train)
#
# pre = svr.predict(x_test)
# print('###########pre##############',pre)
# print('###########y_true##############',y_test)
# print('#######train_score############',svr.score(x_train,y_train))
# print('#######test_score############',svr.score(x_test,y_test))
# print('#######test_mae############',mean_absolute_error(pre,y_test))
# print('#######test_r2############',r2_score(pre,y_test))





import pandas as pd
import plotly as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pyplt = py.offline.plot
fm = pd.read_csv('man.csv')
fm = fm['age']

list_m = []
for i in fm:
    a = float(i)
    list_m.append(a)


ff = pd.read_csv('female.csv')
ff = ff['age']
list_f = []
for j in ff:
    list_f.append(float(j))

ft = pd.read_csv('../abcd_cbcl_baseline_label.csv')
ft = ft['interview_age']
list_t = []
for j in ft:
    list_t.append(float(j))
# seaborn模块绘制分组的直方图和核密度图
#取出所有
Age_T = list_t
# 取出男性年龄
Age_Male = list_m
# 取出女性年龄
Age_Female = list_f
print(Age_Female)
plt.figure(figsize=(15,5))
plt.subplot(121)
#总体分布
#sns.distplot(Age_T, bins = 20, kde = False, hist_kws = {'color':'black'}, label = 'M&F')

# 绘制男女乘客年龄的直方图
sns.distplot(Age_Male, bins = 20, kde = False, hist_kws = {'color':'green'}, label = 'M')
# 绘制女性年龄的直方图
sns.distplot(Age_Female, bins = 20, kde = False, hist_kws = {'color':'black'}, label = 'F')
plt.title('Age Distribution')
# 显示图例
plt.legend()


plt.subplot(122)
# 绘制男女乘客年龄的核密度图
sns.distplot(Age_Male, hist = False, kde_kws = {'color':'red', 'linestyle':'-'},
             norm_hist = True, label = 'M')
# 绘制女性年龄的核密度图
sns.distplot(Age_Female, hist = False, kde_kws = {'color':'black', 'linestyle':'--'},
             norm_hist = True, label = 'F')
plt.title('Age Distribution')
# 显示图例
plt.legend()
# 显示图形
plt.show()
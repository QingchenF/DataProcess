import pandas as pd
import plotly as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pyplt = py.offline.plot

ft = pd.read_csv('./ABCD_CBCL_Label_z.csv')
ft = ft['Int_Z']
list_t = []
for j in ft:
    list_t.append(float(j))

#取出所有
value_T = list_t

plt.figure(figsize=(15,5))
plt.subplot(121)
#总体分布
sns.distplot(value_T, bins = 20, kde = False, hist_kws = {'color':'black'})

plt.xlabel('Int_Z Value')
plt.ylabel('Count')

plt.title('Int_Z Distribution')

plt.subplot(122)
# 显示图例
plt.legend()
# 显示图形
plt.show()
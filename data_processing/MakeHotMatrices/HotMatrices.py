import scipy.io as scio
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

m1 = scio.loadmat('./sub_01_BOLD_wm_corr_2.mat')
m1_np = m1['corr']



fig, ax = plt.subplots(figsize = (38,38))

#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(m1_np,2)),
                annot=False, vmax=1, vmin=0, xticklabels= False, yticklabels= False, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
#            square=True, cmap="YlGnBu")

i = 'abc'
plt.savefig('/Users/fan/Desktop/Hotmap/'+ i +'.png')
plt.show()
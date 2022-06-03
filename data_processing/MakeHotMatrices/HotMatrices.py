import scipy.io as scio
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(threshold=np.sys.maxsize)


m1 = scio.loadmat('./sub_01_BOLD_wm_corr_2.mat')
m1_np = m1['corr']
m1_tril = np.tril(m1_np)
print(m1_tril)

m2 = scio.loadmat('./sub_01_seeg_corr_1_4Hz.mat')
m2_np = m2['corr']
m2_triu = np.triu(m2_np)
print(m2_triu)
#fig, ax = plt.subplots(figsize = (38,38))

#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(m1_tril,2)),
                annot=False,xticklabels= False, yticklabels= False,linewidths = 0.05, linecolor= 'black',  cmap="Blues")
#square=True,
#vmax=1, vmin=0,#设置展示最大值，最小值
#plt.figure(dpi=300,figsize=(120,100))
plt.savefig('sub_01_BOLD_wm_corr_2.png')

plt.show()

sns.heatmap(pd.DataFrame(np.round(m2_triu,2)),
                annot=False,xticklabels= False, yticklabels= False,  cmap="Blues")
#plt.figure(dpi=300,figsize=(120,100))

plt.savefig('sub_01_seeg_corr_1_4Hz.png')

plt.show()


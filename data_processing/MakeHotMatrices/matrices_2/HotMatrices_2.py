import scipy.io as scio
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(threshold=np.sys.maxsize)


m1 = scio.loadmat('DSI_BOLD_corr_matrix_0720_withouttransformed.mat')
print(m1)
m1_np = m1['DSI_wm_corr']
m1_tril = np.tril(m1_np) #下三角

m2_np = m1['BOLD_wm_corr']
m2_triu = np.triu(m2_np) #上三角
print(m2_triu)
#fig, ax = plt.subplots(figsize = (38,38))
'''
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
#square=True,
#vmax=1, vmin=0,#设置展示最大值，最小值
#plt.figure(dpi=300,figsize=(120,100))
'''
ax = sns.heatmap(pd.DataFrame(np.round(m1_tril,2)),
                annot=False, xticklabels=False, yticklabels=False, vmin=-1, vmax=1, cmap="Blues",
                cbar_kws = {'format': '%.1f'}
                )

cbar_1 = ax.collections[0].colorbar
cbar_1.ax.tick_params(labelsize=14, left=False, right=False)
plt.savefig('DSI_wm_corr_0720.png', dpi=300)

plt.show()

bx = sns.heatmap(pd.DataFrame(np.round(m2_triu,2)),
                annot=False, xticklabels=False, yticklabels=False, cmap="Blues",
                 cbar_kws={'format': '%.1f', 'ticks': [-0.2, 0.4, 1.0]}
                 )
cbar_2 = bx.collections[0].colorbar
cbar_2.ax.tick_params(labelsize=24, left=False, right=False)

plt.savefig('BOLD_wm_corr_0720.png', dpi=300)

plt.show()

mm = m1_tril + m2_triu

cx = sns.heatmap(pd.DataFrame(np.round(mm,2)),
                annot=False, xticklabels=False,yticklabels=False, cmap="Blues",
                cbar_kws ={'format': '%.1f','ticks': [-1.0, 0.0, 1.0]}
                )

cbar_3 = cx.collections[0].colorbar
cbar_3.ax.tick_params(labelsize=26, left=False, right=False)
plt.savefig('add_0720.png', dpi=300)

plt.show()




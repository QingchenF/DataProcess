import scipy.io as scio
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(threshold=np.sys.maxsize)


m1 = scio.loadmat('DSI_SEEG_corr_matrix_0720_withouttransformed.mat')
print(m1)
m1_np = m1['DSI_wm_corr']
m1_tril = np.tril(m1_np) #下三角

m2_np = m1['SEEG_wm_corr']
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
                annot=False, xticklabels=False, yticklabels=False, vmin=-0.4, cmap="Blues",
                cbar_kws = {'format': '%.1f'}
                )

cbar_1 = ax.collections[0].colorbar
cbar_1.ax.tick_params(labelsize=14,left=False,right=False)
plt.savefig('DSI_wm_corr_0720.png',dpi=300)

plt.show()
#以corr的形状生成一个全为0的矩阵
mask = np.zeros_like(m2_np)
#将mask的对角线及以上设置为True#这部分就是对应要被遮掉的部分
mask[np.tril_indices_from(mask)] = True
bx = sns.heatmap(pd.DataFrame(np.round(m2_triu,2)),
                annot=False, xticklabels=False, yticklabels=False,mask=mask, vmax=0.8, cmap="Blues",
                 cbar_kws={'format': '%.1f', 'ticks': [-0.4,  0.2, 0.8]}
                 )
cbar_2 = bx.collections[0].colorbar
cbar_2.ax.tick_params(labelsize=26, left=False, right=False)

plt.savefig('SEEG_corr_0720_c.png',dpi=300)

plt.show()

mm = m1_tril + m2_triu

cx = sns.heatmap(pd.DataFrame(np.round(mm, 2)),
                annot=False, xticklabels=False, vmax=0.8, yticklabels=False, cmap="Blues",
                cbar_kws={'format': '%.1f', 'ticks': [-0.4, 0.0, 0.4, 0.8]}
                )

cbar_3 = cx.collections[0].colorbar
cbar_3.ax.tick_params(labelsize=20, left=False, right=False)
plt.savefig('add_3_2.png',dpi=300)

plt.show()




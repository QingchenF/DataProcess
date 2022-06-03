
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
from PIL import Image


m1 = scio.loadmat('./sub_01_BOLD_wm_corr_2.mat')
m1_np = m1['corr']
#以corr的形状生成一个全为0的矩阵
mask = np.zeros_like(m1_np)
#将mask的对角线及以上设置为True#这部分就是对应要被遮掉的部分
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(m1_np, xticklabels= False,yticklabels= False,mask=mask,annot=False,cmap="Blues")
    plt.figure(dpi=300, figsize=(120, 100))
    #plt.savefig('a.png')
    #plt.show()

m2 = scio.loadmat('./sub_01_seeg_corr_1_4Hz.mat')
m2_np = m2['corr']
#以corr的形状生成一个全为0的矩阵
mask = np.zeros_like(m2_np)
#将mask的对角线及以上设置为True#这部分就是对应要被遮掉的部分
mask[np.tril_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(m2_np, xticklabels= False,yticklabels= False,mask=mask,annot=False,cmap="Blues")
    plt.figure(dpi=300, figsize=(120, 100))
    #plt.savefig('b.png')
    #plt.show()
'''
#合并到一起
img_a = Image.open('./a.png')
img_a = img_a.convert('RGBA')
img_b = Image.open('./b.png')
img_b = img_b.convert('RGBA')
img = Image.blend(img_a,img_b,0.4)
#img.show()
'''
'''

#旋转
im = Image.open('./Figure_1.png')
im.rotate(180,expand=True).save('./1-1.png')
'''
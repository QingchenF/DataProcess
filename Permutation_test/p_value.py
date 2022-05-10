import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    list=np.empty(nmc)
    for j in range(999):
        np.random.shuffle(zs)
        list[j]=np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return list
xs = np.array([24,43,58,67,61,44,67,49,59,52,62,50])
ys = np.array([42,43,65,26,33,41,19,54,42,20,17,60,37,42,55,28])
list_a = exact_mc_perm_test(xs, ys, 999)
print(list_a)
statistic, pvalue = stats.mstats.ttest_ind(xs,ys)
print('P:',pvalue)
sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(list_a,color="r",bins=30,kde=True) #kde=true，显示拟合曲线
plt.title('Permutation Test')
plt.xlabel('difference')
plt.ylabel('distribution')
plt.show()
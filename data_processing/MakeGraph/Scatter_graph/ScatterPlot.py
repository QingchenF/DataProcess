import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
x = pd.read_csv("/Users/fan/Documents/Data/Res/ABCD_CBCL_Res/Int/test_y")

y = pd.read_csv("/Users/fan/Documents/Data/Res/ABCD_CBCL_Res/Int/Predict_Score_Int.csv")

y = np.array(y)
x = np.array(x)

#plt.scatter(x,y,alpha=0.6)
plt.plot(x,y,'o',ls='--')

plt.text(1,1,"r = 0.03",size = 10,bbox = dict(alpha = 0.2))
plt.xlabel('True',)
plt.ylabel('Predict')
plt.show()

'''
# 图例
plt.legend()
#x轴范围
plt.xlim()
#x轴范围
plt.xticks(np.arange(2000,2011))
'''
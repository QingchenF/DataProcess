import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 读入数据
Res = pd.read_csv(r'./res_h.csv')
# 绘制水平交错条形图

sns.barplot(x = 'Diagnosis', # 指定x轴数据
            y = 'Factor Score(Z)', # 指定y轴数据
            hue = 'Factor', # 指定分组数据
            data = Res, # 指定绘图数据集
            palette = 'Greys_r', # 指定的不同颜色
            # 指定误差棒的颜色
            errwidth=10, # 指定误差棒的线宽
            saturation = 1, # 指定颜色的透明度，这里设置为无透明度
            capsize = 1# 指定误差棒两端线条的宽度
           )

# 添加图形标题
# 显示图形
plt.show()
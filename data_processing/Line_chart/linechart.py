import scipy.io as scio
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = scio.loadmat('./fig2a_bold_seeg_paris_wave1.mat')

data_bold_1_small = pd.DataFrame(data['bold_1_small'],index=['bold_1_small']).T #添加行名：index=['bold_1_small']
data_bold_2_big = pd.DataFrame(data['bold_2_big'],index=['bold_2_big']).T
data_bold_x = pd.DataFrame(data['bold_x'],index=['bold_x']).T
dataframe = [data_bold_1_small,data_bold_2_big,data_bold_x]
data_res = pd.concat(dataframe)
print(data_res)
sns.lineplot(data=data_res,x='bold_x',y='bold_1_small', linewidth=3.0, palette="tab10")
sns.lineplot(data=data_res,x='bold_x',y='bold_2_big', linewidth=3.0)
#修改x轴、y轴描述
plt.xlabel('Time(Sec)', fontdict={'color': 'black',
                             'fontstyle': 'normal',
                             'family': 'Times New Roman',
                             'weight': 'normal',
                             'size': 15})
plt.ylabel('BOLD signal(a.u.)', fontdict={'color': 'black',
                          'fontstyle': 'normal',
                          'family': 'Times New Roman',
                          'weight': 'normal',
                          'size': 15})

plt.show()

data_seeg_1_small = pd.DataFrame(data['seeg_1_small'])
data_seeg_2_big = pd.DataFrame(data['seeg_2_big'])
data_seeg_x = pd.DataFrame(data['seeg_x'])



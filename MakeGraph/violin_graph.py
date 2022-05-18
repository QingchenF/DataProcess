import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
tips = pd.read_csv(r'Histogram/man.csv')
# 绘制分组小提琴图
sns.violinplot(y = 250, # 指定y轴的数据
               x = 130, # 指定x轴的数据
               hue = "sex", # 指定分组变量
               data = tips, # 指定绘图的数据集
               order = ['Thur','Fri','Sat','Sun'], # 指定x轴刻度标签的顺序
               scale = 'count', # 以男女客户数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
              )
# 添加图形标题
plt.title('每天不同性别客户的消费额情况')
# 设置图例
plt.legend(loc = 'upper center', ncol = 2)
# 显示图形
plt.show()
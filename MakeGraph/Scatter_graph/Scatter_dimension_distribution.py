import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#iris = sns.load_dataset('name',data_home='/Users/fan/PycharmProjects/data_processing/MakeGraph/Scatter_graph/Predict_Score_Int.csv',cache=True)
iris = pd.read_csv('./Predict_Score_Bagging_Int.csv')
print(iris)
'''
# seaborn模块绘制分组散点图
sns.lmplot(x = 'pre', # 指定x轴变量
           y = 'y', # 指定y轴变量
           hue = 'Species', # 指定分组变量
           data = iris, # 指定绘图数据集
           legend_out = False, # 将图例呈现在图框内
           truncate=True # 根据实际的数据范围，对拟合线作截断操作
          )
          '''
sns.lmplot(x='y',y='Predict',data=iris)
# 修改x轴和y轴标签
plt.xlabel('True')
plt.ylabel('predict')
plt.text(8.7,2.9, "r = 0.09\nMAE = 1.22",size = 10,bbox = dict(alpha = 0.2))
# 添加标题
plt.title('Int')
# 显示图形
plt.show()
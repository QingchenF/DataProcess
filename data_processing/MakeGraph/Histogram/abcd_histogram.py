import pandas as pd
import plotly as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fm = pd.read_csv('ABCD_Label.csv')
res = fm['Age']

sns.distplot(res, bins=20, kde=False, hist_kws={'color': 'black'}, label='M')
plt.show()


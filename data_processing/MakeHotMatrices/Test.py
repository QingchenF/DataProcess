import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
def upper_tri_indexing(matirx):
    m = matirx.shape[0]
    r,c = np.triu_indices(m,1)
    return matirx[r,c]
m1 = scio.loadmat('./sub_01_BOLD_wm_corr_2.mat')
m1_np = m1['corr']
m1_triu = np.triu(m1_np)
m1_tril = np.tril(m1_np)

fig, ax = plt.subplots(figsize = (38,38))

list_1 = ['PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',   'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r',
 'Wistia', 'Wistia_r',
 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
 'flag', 'flag_r',
 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
  'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
  'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',
 'hot', 'hot_r', 'hsv', 'hsv_r',
 'icefire', 'icefire_r', 'inferno', 'inferno_r',
 'jet', 'jet_r',
 'magma, magma_r', 'mako, mako_r',
 'nipy_spectral', 'nipy_spectral_r',
 'ocean', 'ocean_r',
 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r',
 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',
 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
  'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
 'viridis', 'viridis_r', 'vlag', 'vlag_r',
 'winter', 'winter_r','Accent', 'Accent_r',
 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 
 'CMRmap', 'CMRmap_r', 
 'Dark2', 'Dark2_r', 
 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r']


#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
list_2 = ['magma', 'magma_r', 'mako', 'mako_r',
 'nipy_spectral', 'nipy_spectral_r',
 'ocean', 'ocean_r',
 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r',
 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',
 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
  'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
 'viridis', 'viridis_r', 'vlag', 'vlag_r',
 'winter', 'winter_r','Accent', 'Accent_r',
 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
 'CMRmap', 'CMRmap_r',
 'Dark2', 'Dark2_r',
 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r']

list_3 = ['Accent', 'Accent_r',
 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
 'CMRmap', 'CMRmap_r',
 'Dark2', 'Dark2_r',
 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r']

for i in list_3:
  sns.heatmap(pd.DataFrame(np.round(m1_tril,2)),
                 annot=False, vmax=1, vmin=0, xticklabels= False, yticklabels= False, square=True, cmap=i)
  plt.savefig('/Users/fan/Desktop/Hotmap2/' + i + '.png')

  plt.show()

# sns.heatmap(pd.DataFrame(np.round(m1_tril,2)),
#                  annot=False, vmax=1, vmin=0, xticklabels= False, yticklabels= False, square=True, cmap="RdBu_r")
# plt.show()

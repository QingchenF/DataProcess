import scipy.io as scio

def find_martrix_min_value(data_matrix):
    '''
    功能：找到矩阵最小值
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    print('data_matrix min:', min(new_data))


def find_martrix_max_value(data_matrix):
    '''
    功能：找到矩阵最大值
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    print('data_matrix max:', max(new_data))

m1 = scio.loadmat('./DSI_BOLD_corr_matrix_0605.mat')
m1_np = m1['BOLD_wm_corr']

find_martrix_min_value(m1_np)
find_martrix_max_value(m1_np)

m2 = scio.loadmat('./DSI_SEEG_corr_matrix_0605.mat')
m2_np = m2['SEEG_wm_corr']

find_martrix_min_value(m2_np)
find_martrix_max_value(m2_np)
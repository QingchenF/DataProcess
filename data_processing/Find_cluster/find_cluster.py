import scipy
import joblib
import glob
import nibabel as nib
import numpy as np
# data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/test_data/test_test/*.nii"),reverse=True)
# data = '/Users/fan/Documents/Data/test_data/test_test/sub-NDARINV90VTTWND_ses-baselineYear1Arm1_task-rest_bold_atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries_thresh-fd0p2mm_censor-10min_conndata-network_connectivity.pconn.nii'
# img_data = nib.load(data)
# img_data = img_data.get_data()
# np.savetxt('./img_data', img_data)
# #print(img_data,type(img_data))
# def upper_tri_indexing(matirx):
#     m = matirx.shape[0]
#     r,c = np.triu_indices(m,1)
#     return matirx[r,c]
#
# n = np.array([[1,2,3,4],[5,1,7,8],[2,4,1,6],[3,4,6,1]])
# #print('--n--\n',n)
# res_n = upper_tri_indexing(n)
# print('--res_n--\n', res_n)
# files_data = []
# for i in data_files_all:
#     img_data = nib.load(i)
#     img_data = img_data.get_data()
#     img_data_reshape = upper_tri_indexing(img_data)
#     files_data.append(img_data_reshape)
# print(files_data)
res = joblib.load('./pls_bagging_model_General_8.pkl')
print(res.estimator.base_estimator.coef_)
#res.base_estimator


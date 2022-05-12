import numpy as np
import scipy.io as io
from nilearn import plotting
from nilearn import datasets
from nilearn.image import load_img
from matplotlib import pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

fMRIData = load_img(r'/Users/fan/Desktop/sub-NDARINV3XMEM8L9_ses-baselineYear1Arm1_task-rest_bold_atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries_thresh-fd0p2mm_censor-10min_conndata-network_connectivity.pconn.nii')

# atlas

atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
labels = atlas.labels

# Extracting times series
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)
time_series = masker.fit_transform(fMRIData)

# correlation
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
print(correlation_matrix)

# Plot the correlation matrix
plt.figure()
np.fill_diagonal(correlation_matrix, 0)
print(correlation_matrix)
plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r")
plt.colorbar()
plt.show()

# save correlation result
io.savemat('correlation_matrix.mat', {'correlation_matrix': correlation_matrix})

# plot connectome with 99% edge strength threshold in the connectivity
coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
plotting.plot_connectome(correlation_matrix, coordinates,
                         edge_threshold="99%",
                         title='AAL 116')
plotting.show()

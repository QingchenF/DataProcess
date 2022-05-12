import nibabel as nib
import numpy as np

data = 'sub-NDARINV0FBX6KWX_ses-baselineYear1Arm1_task-rest_bold_atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii'

data_show = nib.load(data)
data_arr = np.array(data)

print(data_show,data_show.shape,data_arr,data_arr.shape)





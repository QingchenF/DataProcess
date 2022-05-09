# encoding: utf-8
import numpy as np

#generate a file
def ToolboxCSV(filename = 'filename.csv',list=[]):
    path = "./Note_Res/"
    file = open(path+filename,mode='w')
    for tra in list :
        if(isinstance(tra,str)):
          file.write(tra)
          file.write('\n')
        else:
            file.write(str(tra))
            file.write('\n')
#generate a experiment file
def ToolboxCSV_srtver(filename = 'filename.csv',list=[]):
    path = "/GPFS/cuizaixu_lab_permanent/fanqingchen/Res/Note_Res/"
    file = open(path+filename,mode='w')
    for tra in list :
        if(isinstance(tra,str)):
          file.write(tra)
          file.write('\n')
        else:
            file.write(str(tra))
            file.write('\n')

#Define a function that takes the upper triangle.Working with Symmetric Matrices
def upper_tri_indexing(matirx):
    m = matirx.shape[0]
    r,c = np.triu_indices(m,1)
    return matirx[r,c]
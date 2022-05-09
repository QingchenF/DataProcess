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
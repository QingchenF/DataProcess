import os

#设定文件路径
path="/Users/fan/Documents/Data/ABCD_FC_10min_2"

#获取该目录下所有文件，存入列表中
fileList=os.listdir(path)

n = 0000

for i in fileList:


    # 设置旧文件名（就是路径+文件名）
    oldname = path+'/'+i

    # 设置新文件名
    newname = path + os.sep + str(n + 1)+'_'+i[0:15]+'.nii'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名

    #print(oldname, '======>', newname)

    n += 1

import glob

data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/ABCD_FC_10min/*.nii"))
print(data_files_all)
data_all = open("./data_all.csv",mode='w')

for p in data_files_all:
    data_all.write(p)
    data_all.write('\n')
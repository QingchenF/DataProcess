fw  = open("/Users/fan/Desktop/path.csv",mode='w')

with open("./unique_1.csv",mode='r') as f1:
    list1 = f1.readlines()
print(len(list1))

list_path = []
with open("./data_10min_fc.txt",mode="r") as f2:
    for line in f2:
        #print(line)
        if line.split()[-1][1:-1].endswith(".txt"):
             #print(line.split()[-1][1:-1])
             list_path.append(line.split()[-1][1:-1])
print(list_path[0])

for id in list1:
   # print(id)
    for path in list_path:
        if id in path:
            print("000")
            fw.write(path)

fw.close()
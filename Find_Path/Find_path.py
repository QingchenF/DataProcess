list_r = list()
set_cf = set()
fw = open('./Re_Data_path',mode='w')
with open('ss_s',mode='r',encoding='utf-8') as s,open('tt_t',mode='r') as t:
    list_s = s.readlines()
    print(len(list_s))
    list_t = t.readlines()
    print(len(list_t))
    for l in list_t:
        if list_t.count(l)>1:
           # print(l)#重复内容
            set_cf.add(l)

    print("-----",len(set_cf))
    for line in list_t:
        if line not in list_s:
            list_r.append(line)
print(len(list_r))

for j in set_cf:
    fw.write(j)


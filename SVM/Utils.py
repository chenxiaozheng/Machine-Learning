from numpy import *


def Load_data(filename):
    featrue = []
    label = []
    sample_func = None
    sample_func = lambda x: x - 1

    f = open(filename)        
    for line in f.readlines():
        arr = line.strip().split('\t')
        featrue.append([float(arr[0]), float(arr[1])])
        if float(arr[2]) < 0.5:
            label.append([float(sample_func(float(arr[2])))])
        else:
            label.append([float(arr[2])])
    
    return featrue, label

#2 在样本集中采取随机选择的方法选取第二个不等于第一个alphai的
#优化向量alphaj
# def selectJrand(i, m):



#3 约束范围L<=alphaj<=H内的更新后的alphaj值  
# def clipAlpha(Aj, H, L):
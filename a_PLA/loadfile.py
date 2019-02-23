# homework 1

#Load the dataset

import numpy as np

def loadfile(file):
    X = [] # 特征，形状=（样本，特征）
    Y = [] # 标签，形状=（示例，）
    for line in open(file).readlines():
        items = line.strip().split('\t') # 功能和标签按标签拆分
        y = items[1].strip()
        y = float(y) # str to float
        Y.append(y)
        x = items[0].strip().split(' ')
        x = list(map(float, x)) # str to float
        X.append(x)
    X = np.array(X) # list to array
    Y = np.array(Y) # list to array
    return X, Y
from numpy import * 
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

def LoadData(filename):
    dataFeature = []
    dataLabel = []

    file = open(filename)
    # print(file)
    # print(file.readlines())
    for line in file.readlines():
        line_arr = line.strip().split()
        # print(line_arr)
        dataFeature.append([1.0, float(line_arr[0]), float(line_arr[1])]) 
        dataLabel.append(float(line_arr[2]))

    return dataFeature, dataLabel

def sigmod(w):
    return 1.0/(1 + exp(-w))

def GradientDescent(DataFeature, dataLabel, n = 500, learn_rate = 0.001):

    data = mat(DataFeature)
    label =  mat(dataLabel).transpose()
    # print(label)
    m,n = shape(data)
    print(m,n)
    Weight = ones((n, 1))

    for i in range(n):
        y_pred = sigmod(dot(DataFeature, Weight))
        Weight =+ dot(data.transpose(), learn_rate * (label - y_pred))
        # print(dot(DataFeature, Weight))  
    return Weight

def plot_fit(data, labelMat, weights):
    dataArr = array(data)
    print(dataArr)
    n = shape(dataArr)[0]
    print(n)
    x_cord1 = []; y_cord1 = []
    x_cord2 = []; y_cord2 = []
    for i in range(n):  
        if int(labelMat[i]) == 1:  
            x_cord1.append(dataArr[i,1]); y_cord1.append(dataArr[i,2])  
        else: x_cord2.append(dataArr[i,1]); y_cord2.append(dataArr[i,2])  
    
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    ax.scatter(x_cord1, y_cord1, s = 30, c = 'red', marker='s')  
    ax.scatter(x_cord2, y_cord2, s = 30, c = 'green')  
    
    x = arange(-3.0, 3.0, 0.1)  
    y = ((-weights[0]- weights[1] * x)/weights[2]).transpose()
    ax.plot(x, y)  
    plt.xlabel('X1');  
    plt.ylabel('X2');  
    plt.show()  
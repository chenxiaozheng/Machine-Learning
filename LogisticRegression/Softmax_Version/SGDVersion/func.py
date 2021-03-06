from numpy import * 
import matplotlib
import matplotlib.pyplot as plt

def LoadData(filename):
    dataFeature = []
    dataLabel = []

    file = open(filename)
    # print(file)
    # print(file.readlines())
    for line in file.readlines():
        line_arr = line.split().strip()
        # print(line_arr)
        dataFeature.append([1.0, float(line_arr[0]), float(line_arr[1])]) 
        dataLabel.append(float(line_arr[2]))

    return dataFeature, dataLabel


#softmax
def SoftMax(Molecular, Denominator):
    return exp(Molecular) / sum(Denominator)

# #随机梯度下降
# def SGdescent(data, label, n_iter = 150):
#     dataNp = mat(data)
#     labelNp = mat(label).transpose()
#     m, n = dataNp.shape
#     weight = ones((n))
#     # print(weight)
#     alpa = 0.01
#     # print("m = ", len(range(m)))
#     for i in range(m):
#         error = labelNp[i] - sigmod(dot(dataNp[i], weight.transpose()))
#         weight = weight + alpa * dot(error, dataNp[i])
#     return weight


# #改进的随机梯度下降
# def advanceSGdescent(data, label, n_inter = 150):
#     dataNp = mat(data)
#     labelNp = mat(label).transpose()
#     m,n = dataNp.shape
#     weight = ones((n))
#     alpa = 0.01

#     for i in range(n_inter):
#         for j in range(m):
#             randidx = int(random.uniform(0, len(range(m))))
#             alpha = (4 / (1.0 + i + j))+ 0.01 
#             error = labelNp[randidx] - sigmod(dot(dataNp[randidx], weight.transpose()))
#             weight = weight + alpha * dot(error, dataNp[randidx])
#     return weight



# #做图
# def plot_fit(data, labelMat, weights):
#     dataArr = array(data)
#     # print(dataArr)
#     n = shape(dataArr)[0]
#     # print(n)
#     x_cord1 = []; y_cord1 = []
#     x_cord2 = []; y_cord2 = []
#     for i in range(n):  
#         if int(labelMat[i]) == 1:  
#             x_cord1.append(dataArr[i,1]); y_cord1.append(dataArr[i,2])  
#         else: x_cord2.append(dataArr[i,1]); y_cord2.append(dataArr[i,2])  
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)  
#     ax.scatter(x_cord1, y_cord1, s = 30, c = 'red', marker='s')  
#     ax.scatter(x_cord2, y_cord2, s = 30, c = 'green')  
    
#     x = arange(-3.0, 3.0, 0.1)  

#     # W0*X0 + W1*X1 + W2*X2 = 0 and X0 = 1  >>  X2 = (-W0 - W1*X1)/W2
#     x2 = ((-weights[0]- weights[1] * x)/weights[2]).transpose()
#     ax.plot(x, x2)  
#     plt.xlabel('X1');  
#     plt.ylabel('X2');  
#     plt.show()  
#     return

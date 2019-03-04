import numpy as np
from func import *


if __name__ == "__main__":
    dataFeature, dataLabel = LoadData("../Data/wine.data")
    # print(dataFeature)
    # print(dataLabel)
    # print(sigmod(dataFeature))
    weights1 = SGdescent(dataFeature, dataLabel)          # 随机梯度下降
    weights2 = advanceSGdescent(dataFeature, dataLabel)   # 改进的随机梯度下降，
    # print(weights)
    weights = weights1.reshape(3, 1)
    weights = weights2.reshape(3, 1)
    print(weights2)
    plot_fit(dataFeature, dataLabel, weights)


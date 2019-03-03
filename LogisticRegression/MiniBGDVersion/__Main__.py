import numpy as np
from func import *


if __name__ == "__main__":
    dataFeature, dataLabel = LoadData("../testSet.txt")
    # print(dataFeature)
    # print(dataLabel)
    # print(sigmod(dataFeature))
    weights = MiniBatchGradientDescent(dataFeature, dataLabel, 500, 0.001)
    print(weights)
    plot_fit(dataFeature, dataLabel, weights)


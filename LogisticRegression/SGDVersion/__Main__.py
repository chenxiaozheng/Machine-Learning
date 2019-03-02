import numpy as np
from func import *


if __name__ == "__main__":
    dataFeature, dataLabel = LoadData("../testSet.txt")
    # print(dataFeature)
    # print(dataLabel)
    # print(sigmod(dataFeature))
    weights = SGdescent(dataFeature, dataLabel)
    print(weights)
    weights = weights.reshape(3, 1)
    print(weights)
    plot_fit(dataFeature, dataLabel, weights)


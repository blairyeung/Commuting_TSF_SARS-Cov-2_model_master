import numpy as np
import math


def moving_average(a, n=7) :
    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def normalize(vec):
    return vec / np.sum(vec)


def norm(x, mean=0, scale=1):

    size = len(x)

    matrix = np.zeros(shape=(size, ))

    # Raw Gaussian Values
    for i in range(size):
        matrix[i] = getGaussian(x[i], scale, mean)

    return matrix


def getGaussian(distance, sigma, mean):
    power = - (math.pow(distance - mean, 2)) / (sigma * sigma)
    val = (1 / (2 * math.pi * sigma * sigma)) * math.pow(math.e, power)
    return val

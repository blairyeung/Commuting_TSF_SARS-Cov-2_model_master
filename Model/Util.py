import numpy as np
import math


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

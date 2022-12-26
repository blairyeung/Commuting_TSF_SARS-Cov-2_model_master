import math

import matplotlib.pyplot as plt

import Parameters
import numpy as np
import cv2

def blur(lst):
    """
        Perform difference of Gaussian ()
        :param lst:
        :return:
    """
    raw = np.zeros(shape=(100, ))
    print(lst)
    raw[5:12] = lst[0]
    raw[12:18] = lst[1]
    raw[18:30] = lst[2]
    raw[30:40] = lst[3]
    raw[40:50] = lst[4]
    raw[50:60] = lst[5]
    raw[60:70] = lst[6]
    raw[70:80] = lst[7]
    raw[80:] = lst[8]

    # raw = raw * Parameters.RAW_AGE_DISTRIBUTION / Parameters.ONT_POPULATOIN

    blurred = np.reshape(np.array(cv2.GaussianBlur(raw.reshape(100, 1), (9, 9), 0)), newshape=(100,))


    rslt = np.zeros(shape=(16, ))

    for i in range(15):
        rslt[i] = np.sum(blurred[5 * i:5 * i + 5]) / 5
    rslt[15] = np.sum(blurred[75:]) / 25
    return rslt


def age_dog_algo(lst):
    if sum(lst) == 0:
        return np.zeros(shape=(16, ))
    else:
        return blur(lst)

if __name__ == '__main__':
    test = np.ones(shape=(9, ))
    rslt = age_dog_algo(test)
    plt.plot(rslt)
    plt.show()
import numpy as np
import Parameters
from scipy.signal import convolve2d
from Model import Model

ratio = np.ones(shape=(16,), dtype=float)
ratio = np.reshape(ratio, newshape=(16, 1))
kernel = np.reshape(Parameters.EXP2ACT_CONVOLUTION_KERNEL, newshape=(12, 1))
kernel = np.matmul(ratio, np.transpose(kernel))
print(kernel)
print(kernel.shape)


rslt = np.sum(np.multiply(np.ones(shape=(16, 12)),
                                  kernel), axis=1)

print(rslt)

"""
for i in range(100):
    pass
    m = Model()
    m.run_one_cycle()

"""

"""
date = 98
test = np.ones(shape=(100, 16))
kernel = Parameters.EXP2ACT_CONVOLUTION_KERNEL
print(kernel.shape[0])
kernel = kernel.reshape((kernel.shape[0], 1))

conv = test[date-12:date]
print(kernel)
conved = np.multiply(conv, kernel)
# conved = convolve2d(conv, kernel)[:10]
print(conved)
conv_rslt = np.sum(conved, axis=0)

ttt = np.sum(np.multiply(conv, kernel), axis=0)
print(conv_rslt)
print()
"""
import numpy as np

ratio = np.random.rand(16,)
ratio = np.ones(shape=(16))
data = np.random.rand(925)

raito = np.transpose(ratio)

rslt = np.matmul(data, raito)

print(rslt.shape)
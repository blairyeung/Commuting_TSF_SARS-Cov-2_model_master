import cv2
import numpy as np
import matplotlib.pyplot as plt

susceptibility_raw = np.zeros(shape=(100, ))

susceptibility_raw[0:10] = 0.4
susceptibility_raw[10:20] = 0.38
susceptibility_raw[20:30] = 0.79
susceptibility_raw[30:40] = 0.86
susceptibility_raw[40:50] = 0.8
susceptibility_raw[50:60] = 0.82
susceptibility_raw[60:70] = 0.88
susceptibility_raw[70:100] = 0.74

susceptibility = np.reshape(np.array(cv2.GaussianBlur(susceptibility_raw.reshape(100, 1), (25, 25), 0)),
                            newshape=(100,))

age_strat = np.zeros(shape=(16, ))
for i in range(15):
    age_strat[i] = np.sum(susceptibility[5*i:5 * (i+1)]) / 5

age_strat[15] = np.sum(susceptibility[74:]) / 25
print((age_strat))


plt.plot(age_strat)
np.savetxt('susp_by_age.csv', age_strat, delimiter=',')
plt.show()




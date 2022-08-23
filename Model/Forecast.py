import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os

for i in [2020, 2021, 2022]:
    pass
path = os.getcwd()[:-5] + 'Model Dependencies/Mobility_data/2020_CA_Region_Mobility_Report.csv'
print(path)
data = pd.read_csv(path)
print(data)
mobility = data.ix[:, 'retail_and_recreation_percent_change_from_baseline'].tolist()
# plt.plot(mobility)
# plt.show()
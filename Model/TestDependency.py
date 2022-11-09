import math
import numpy as np
import Dependency as Dependency
from ModelData import ModelData
import Parameters


d = Dependency.Dependency()
# print(d.commute_matrix.shape)
# print(d.date_to_cases_by_phu.keys())
print(d.date_to_vaccines_by_age.shape)
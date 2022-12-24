import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Dependency as Dependency


def augment(lst):
    augment = np.zeros(shape=(100,))
    augment[0:12] = np.array([lst[0]] * 12)
    augment[12:20] = np.array([lst[1]] * 8)
    augment[20:30] = np.array([lst[2]] * 10)
    augment[30:40] = np.array([lst[3]] * 10)
    augment[40:50] = np.array([lst[4]] * 10)
    augment[50:60] = np.array([lst[5]] * 10)
    augment[60:70] = np.array([lst[6]] * 10)
    augment[70:80] = np.array([lst[7]] * 10)
    augment[80:] = np.array([lst[8]] * 20)
    return augment


d = Dependency.Dependency()
cases_read_path = d.get_dependency_path() + 'Canada_hospitalization_inference.csv'
df = pd.read_csv(cases_read_path)
print(df)
cases = np.array(list(df.cases))
hospitalization = np.array(list(df.hospitalization))
icu = np.array(list(df.ICU))
deaths = np.array(list(df.death))

# fit them into the arrays

cases_augment = augment(cases)
hospitalized_augment = augment(hospitalization)
icu_augment = augment(icu)
deaths_augment = augment(deaths)


# cases_augment = np.reshape(np.array(cv2.GaussianBlur(cases_augment.reshape(100, 1), (25, 25), 5)), newshape=(100,))
cases_augment = np.reshape(np.array(cv2.GaussianBlur(cases_augment.reshape(100, 1), (25, 25), 0)), newshape=(100,))
hospitalized_augment = np.reshape(np.array(cv2.GaussianBlur(hospitalized_augment.reshape(100, 1), (25, 25), 0)),
                                  newshape=(100,))
icu_augment = np.reshape(np.array(cv2.GaussianBlur(icu_augment.reshape(100, 1), (25, 25), 0)), newshape=(100,))
deaths_augment = np.reshape(np.array(cv2.GaussianBlur(deaths_augment.reshape(100, 1), (25, 25), 0)), newshape=(100,))


hosp_ratio = hospitalized_augment / cases_augment
icu_ratio = icu_augment / cases_augment
cfr = deaths_augment / cases_augment

# print(cases_augment)

plt.plot(list(range(0, 100)), icu_ratio)
plt.show()

infereced_df = pd.DataFrame({'hospitalization': hosp_ratio,
                             'icu': icu_ratio,
                            'cfr': cfr})

print(hosp_ratio)

age_strat = np.zeros(shape=(16, ))
for i in range(15):
    age_strat[i] = np.sum(hosp_ratio[5*i:5 * (i+1)]) / 5

age_strat[15] = np.sum(hosp_ratio[74:]) / 25
print((age_strat))



# infereced_df.to_csv(d.get_dependency_path() + 'Inferenced_age_specific_data.csv')

# TODO: Find the WEIGHTED AVERAGE and inference the conditional probabilities.
"""
    P(Hosp|Case), P(ICU|Hosp), P(Decease|ICU), P(Decease| not ICU and Hosp), all age-specific.
"""

pass

# tot_cases = np.array([])
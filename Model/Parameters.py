"""
    Version
"""
import math
import os

import pandas as pd
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import norm
from Util import normalize
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2

ver = '1.0'


def log_fit(x, a=0, b=0, c=0):
    return a * np.exp(-b * x) + c

def get_immunity_kernel(dose=0, length=2000):
    if dose == 0:
        return INFECTION_IMMUNITY
    elif dose == 1:
        return 0.8 * TWO_DOSE_EFFICACY
    elif dose == 2:
        return TWO_DOSE_EFFICACY
    elif dose == 3:
        return THREE_DOSE_EFFICACY
    else:
        return np.ones(shape=(2000, 16))
    return np.ones(shape=(2000, 16))


DEPENDENCY_PATH = os.getcwd()[:-5] + 'Model Dependencies/'

# TWO_DOSE_EFFICACY = pd.read_csv(DEPENDENCY_PATH + 'two_dose.csv', delimiter=',').to_numpy().T[1:17].T
# THREE_DOSE_EFFICACY = pd.read_csv(DEPENDENCY_PATH + 'three_dose.csv', delimiter=',').to_numpy().T[1:17].T

TWO_DOSE_EFFICACY = np.multiply(log_fit(np.linspace(0, 1999, 2000), a=62.715, b=0.004, c=10).reshape(2000, 1),
                                np.ones(shape=(16, 1)).T) / 100

THREE_DOSE_EFFICACY = np.multiply(log_fit(np.linspace(0, 1999, 2000), a=93.327, b=0.003, c=0).reshape(2000, 1),
                                np.ones(shape=(16, 1)).T) / 100

# INFECTION_IMMUNITY = np.multiply(log_fit(np.linspace(0, 1999, 2000), a=100, b=0.002, c=0).reshape(2000, 1),
#                                 np.ones(shape=(16, 1)).T) / 100

<<<<<<< Updated upstream
INFECTION_IMMUNITY = np.ones(shape=TWO_DOSE_EFFICACY.shape) - \
                     (np.ones(shape=TWO_DOSE_EFFICACY.shape) - TWO_DOSE_EFFICACY) * 0.154
=======
TWO_DOSE_EFFICACY_RMV = np.concatenate([TWO_DOSE_EFFICACY[60:],
                                        np.zeros(shape=(60, 16))], axis=0)

INFECTION_IMMUNITY = np.ones(shape=THREE_DOSE_EFFICACY.shape) - \
                     (np.ones(shape=THREE_DOSE_EFFICACY.shape) - THREE_DOSE_EFFICACY) * 0.5

# INFECTION_IMMUNITY = np.multiply(log_fit(np.linspace(0, 1999, 2000), a=100, b=0.001, c=0).reshape(2000, 1),
#                                 np.ones(shape=(16, 1)).T) / 100


>>>>>>> Stashed changes
INFECTIOUSNESS = 0.08

"""
    Matrix constants
"""
MATRIX_SIZE = 16
MATRIX_CATEGORIES = ['urban', 'rural']
MATRIX_CONTACT_TYPE = ['home', 'school', 'work', 'others']
MATRIX_COUNTRY_ISO = 'CA'
MATRIX_PRESETS = {'day': np.array([0.9, 1.0, 0.2, 0.8]),
                  'night': (np.ones(shape=(4,)) - np.array([0.9, 1.0, 0.2, 0.8]))
                  }

"""
    Geo constants
"""

province_name = 'Ontario'
NO_PHU = 34
NO_COUNTY = 528
NO_DISTRICT = 49

"""
    Epidemiological constants
"""

OUTBREAK_FIRST_DAY = datetime.datetime(2020, 1, 15)

"""
    Ontario age-specific parameters Done!
"""


ONT_VACCINE_DISTRIBUTION = np.array([0.00060537, 0.00458526, 0.01328221, 0.02727299, 0.04668258, 0.0514049, 0.05772092,
                                     0.05664295, 0.05666874, 0.05662532, 0.06529888, 0.07479382, 0.08567756, 0.07610751,
                                     0.06813767, 0.25849333])

ONT_POPULATOIN = 15109416

ONT_AGE_BAND_POPULATION = np.array([714654, 769062, 803899, 849806, 1057366, 1125877, 1103728, 1042232, 953628,
                                    920118, 952509, 1037514, 1004860, 855432, 702717, 1216014])


ICU_HOSP = 0.1540844459589232

ONT_CASE_DISTRIBUTION = np.array([0.05249542, 0.05111049, 0.04896265, 0.05790512, 0.07973513, 0.09132002, 0.09107228,
                                  0.0873147, 0.08164537, 0.07587614, 0.06908633, 0.06024627, 0.04856209, 0.03894042,
                                  0.03145489, 0.03427268])

ONT_HOSP_DISTRIBUTION = np.array([0.01850005, 0.01709116, 0.01374572, 0.01524114, 0.02426419, 0.03302322, 0.04057495,
                                  0.04515149, 0.04847928, 0.05612047, 0.06963836, 0.08406513, 0.10022135, 0.11472814,
                                  0.13049681, 0.18865855])

ONT_ICU_DISTRIBUTION = np.array([0.01006172, 0.00936271, 0.00781271, 0.00928596, 0.01545035, 0.02279748, 0.03156158,
                                 0.04115525, 0.05372844, 0.07139009, 0.09591567, 0.11829946, 0.13837513, 0.14563542,
                                 0.13922661, 0.08994143])

ONT_DEATH_DISTRIBUTION = np.array([0.0007869, 0.00074751, 0.00071538, 0.00116392, 0.00228272, 0.00373057, 0.00572339,
                                   0.00843267, 0.01284209, 0.02093249, 0.03467635, 0.05417828, 0.08282102, 0.11956677,
                                   0.17747129, 0.47392865])

ONT_HOSP_RATIO = np.array([0.01303065, 0.01235898, 0.01038077, 0.0097061, 0.01121739, 0.01336316, 0.01648061,
                           0.01913183, 0.0219753, 0.02739945, 0.0373693, 0.05187129, 0.07685526, 0.10960614,
                           0.15406819, 0.21108015])

ONT_ICU_RATIO = np.array([0.00122118, 0.00116669, 0.00101664, 0.00101756, 0.0012296, 0.00158911, 0.0022096,
                          0.00300761, 0.00420072, 0.00601005, 0.00887262, 0.01257768, 0.01827112, 0.023918,
                          0.02824015, 0.01775055])

ONT_CFR = np.array([9.87293737e-05, 9.63117654e-05, 9.62190186e-05, 1.30947921e-04, 1.87404371e-04,
                    2.68738282e-04, 4.14312243e-04, 6.37480832e-04, 1.03892535e-03, 1.82436340e-03,
                    3.32161024e-03, 5.97837815e-03, 1.13594029e-02, 2.04377866e-02, 3.74666196e-02,
                    9.35947229e-02])

# susceptibility
SUSC_RATIO = np.array([3.991327254652800027e-01, 3.948075309394725174e-01, 4.022340898538860898e-01,
                       4.963382577801381812e-01, 6.774025858086478724e-01, 7.930450307450174208e-01,
                       8.374194527357667894e-01, 8.402912936299908875e-01, 8.176304599317949506e-01,
                       8.081257676036830429e-01, 8.168620238480317486e-01, 8.359357113555672125e-01,
                       8.574139000421464596e-01, 8.381473039806752734e-01, 7.793588002934677128e-01,
                       7.715186068117150242e-01
                       ])

CLINICAL_RATIO = np.array([0.2865309, 0.26753682, 0.23495708, 0.23019938, 0.25561971, 0.28454104,
                           0.31587461, 0.34737841, 0.38386853, 0.42265894, 0.47025061, 0.52584302,
                           0.59291004, 0.64122657, 0.67295434, 0.71677774])

SUBCLINICAL_RATIO = np.ones(shape=(16,), dtype=float) - CLINICAL_RATIO

REVERSE_CLINICAL_BY_AGE = np.ones(shape=CLINICAL_RATIO.shape) / CLINICAL_RATIO

<<<<<<< Updated upstream
=======
OMICRON_CLINICAL_RATIO = CLINICAL_RATIO * 0.8

OMICRON_SUBCLINICAL_RATIO = np.ones(shape=(16,), dtype=float) - OMICRON_CLINICAL_RATIO

OMICRON_REVERSE_CLINICAL_BY_AGE = np.ones(shape=OMICRON_SUBCLINICAL_RATIO.shape) / OMICRON_CLINICAL_RATIO

>>>>>>> Stashed changes
# Work force
LABOUR_FORCE_BY_AGE = np.array([0, 0, 0.010693183, 0.032079549, 0.083009492, 0.106146399, 0.106351741,
                                0.105920522, 0.103210008, 0.098220198, 0.10624907, 0.098138062, 0.065560558,
                                0.042213176, 0.025934691, 0.016273351, 0])

"""
    Age band names
"""

VACCINE_AGE_BANDS = ['05-11yrs', '12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs', '50-59yrs', '60-69yrs', '70-79yrs',
                     '80+']

AGE_BANDS = ['0-4yrs', '5-9yrs', '10-14yrs', '15-19yrs', '20-24yrs', '25-29yrs', '30-34yrs', '35-39yrs', '40-44yrs',
             '45-49yrs', '50-54yrs', '55-59yrs', '60-64yrs', '65-69yrs', '70-74yrs', '75+yrs']

"""
    Convolution kernels
"""

"""
    Convolutional kernels
"""

raw_age_df = pd.read_csv(os.getcwd()[:-5] + 'Model Dependencies/' + '1710000501-eng.csv')
raw_age_dist = raw_age_df['Persons'].to_numpy()[1:].reshape(21, 1) * np.ones(shape=(1, 5))
raw_age_dist = raw_age_dist.flatten() / 5
raw_age_dist = np.reshape(np.array(cv2.GaussianBlur(raw_age_dist.reshape(105, 1), (9, 9), 0)),
                                      newshape=(105,))

RAW_AGE_DISTRIBUTION = np.zeros(shape=(100, ))
RAW_AGE_DISTRIBUTION[:100] = raw_age_dist[:100]
RAW_AGE_DISTRIBUTION[99] = np.sum(raw_age_dist[100:])

EXP2ACT_RATIO = np.ones(shape=(16,))

kernel_size_1 = np.linspace(0, 12, 12)

EXP2ACT_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))
ACT2CLI_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))
ACT2SUB_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))

kernel_size_2 = np.linspace(0, 25, 25)

SUB2REC_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_2, a=1.6, scale=4))
CLI2REC_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_2, a=1.6, scale=4))

kernel_size_3 = np.linspace(0, 15, 15)

INF2HOS_CONVOLUTION_KERNEL = normalize(norm.pdf(kernel_size_3, loc=7.5, scale=1))
HOS2ICU_CONVOLUTION_KERNEL = normalize(norm.pdf(kernel_size_3, loc=11.1, scale=1))
HOS2DEA_CONVOLUTION_KERNEL = normalize(norm.pdf(kernel_size_3, loc=7.5, scale=math.sqrt(5)))
HOS2RMV_CONVOLUTION_KERNEL = normalize(lognorm.pdf(range(0, 40), s=1.2, loc=11.08))
ICU2RMV_CONVOLUTION_KERNEL = normalize(lognorm.pdf(range(0, 40), s=1.25, loc=13.33))

""" 
    These kernels are for hospitalization, ICU, and deaths
"""

kernel_size_4 = np.linspace(0, 30, 30)
CLI2DEA_CONVOLUTION_KERNEL = normalize(norm.pdf(kernel_size_4, loc=15, scale=2))

VACCINE_EFFICACY_KERNEL_DOSE1 = get_immunity_kernel(dose=1)
VACCINE_EFFICACY_KERNEL_DOSE2 = get_immunity_kernel(dose=2)
VACCINE_EFFICACY_KERNEL_DOSE3 = get_immunity_kernel(dose=3)

INFECTION_EFFICACY_KERNEL = get_immunity_kernel(dose=0)

if __name__ == '__main__':
    plt.plot(get_immunity_kernel(dose=0))
    plt.show()

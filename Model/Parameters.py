"""
    Version
"""
from scipy.stats import gamma
from Util import norm
from Util import normalize
import matplotlib.pyplot as plt
import numpy as np
import datetime

ver = '1.0'


def find_mean(v):
    tot = 0
    for i in range(len(v)):
        tot += v[i]
        if tot >= 0.5:
            print(i, len(v))
            break


def get_bayes(lst):
    bayes = np.zeros(shape=lst.shape)
    for i in range(lst.shape[0]):
        bayes[i] = 1 / lst[i]
    return bayes


def get_immunity_kernel(dose=0, length=2000):
    kernel = np.ones(shape=(length,), dtype=float)
    return kernel


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

ONT_CASE_DISTRIBUTION = np.array([0.05249542, 0.05111049, 0.04896265, 0.05790512, 0.07973513, 0.09132002, 0.09107228,
                                  0.0873147, 0.08164537, 0.07587614, 0.06908633, 0.06024627, 0.04856209, 0.03894042,
                                  0.03145489, 0.03427268])

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
SUSC_BY_AGE = np.array([3.991327254652800027e-01, 3.948075309394725174e-01, 4.022340898538860898e-01,
                        4.963382577801381812e-01, 6.774025858086478724e-01, 7.930450307450174208e-01,
                        8.374194527357667894e-01, 8.402912936299908875e-01, 8.176304599317949506e-01,
                        8.081257676036830429e-01, 8.168620238480317486e-01, 8.359357113555672125e-01,
                        8.574139000421464596e-01, 8.381473039806752734e-01, 7.793588002934677128e-01,
                        7.715186068117150242e-01
                        ])

CLINICAL_BY_AGE = np.array([0.2865309, 0.26724745, 0.22402391, 0.15445847, 0.07258315, 0.10150448, 0.2401337,
                               0.33644525, 0.38357917, 0.42265894, 0.47025061, 0.52584302, 0.59291004, 0.64122657,
                               0.67295434, 0.71677774])


SUBCLINICAL_BY_AGE = np.ones(shape=(16,), dtype=float) - CLINICAL_BY_AGE

"""
    Bayes
"""

BAYES_CLINICAL_BY_AGE = get_bayes(CLINICAL_BY_AGE)

# Work force
LABOUR_FORCE_BY_AGE = np.array([0, 0, 0.010693183, 0.032079549, 0.083009492, 0.106146399, 0.106351741,
                                0.105920522, 0.103210008, 0.098220198, 0.10624907, 0.098138062, 0.065560558,
                                0.042213176, 0.025934691, 0.016273351, 0])

"""
    Vaccine constants
"""
VACCINE_AGE_BANDS = ['05-11yrs', '12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs', '50-59yrs', '60-69yrs', '70-79yrs',
                     '80+']

"""
    Convolution kernels
"""

"""
    These kernels are for 
"""

kernel_size_1 = np.linspace(0, 12, 12)
EXP2ACT_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))
ACT2CLI_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))
ACT2SUB_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_1, a=1.2, scale=4))

"""
    These kernels are for
"""

kernel_size_2 = np.linspace(0, 25, 25)
SUB2REC_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_2, a=1.6, scale=4))

""" 
    These kernels are for hospitalization, ICU, and deaths
"""

kernel_size_3 = np.linspace(0, 15, 15)
INF2HOS_CONVOLUTION_KERNEL = normalize(norm(kernel_size_3, mean=7.5, scale=1))
HOS2ICU_CONVOLUTION_KERNEL = normalize(norm(kernel_size_3, mean=11 - 7.5, scale=1))

kernel_size_4 = np.linspace(0, 15, 15)
ICU2DEA_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_4, a=2.5, scale=1.9))

VACCINE_EFFICACY_KERNEL_DOSE1 = get_immunity_kernel(dose=1)
VACCINE_EFFICACY_KERNEL_DOSE2 = get_immunity_kernel(dose=2)
VACCINE_EFFICACY_KERNEL_DOSE3 = get_immunity_kernel(dose=3)

INFECTION_EFFICACY_KERNEL = get_immunity_kernel(dose=0)

if __name__ == '__main__':
    print(kernel_size_2)
    print(ICU2DEA_CONVOLUTION_KERNEL)
    find_mean(ICU2DEA_CONVOLUTION_KERNEL)
    plt.plot(kernel_size_4, ICU2DEA_CONVOLUTION_KERNEL)
    plt.show()

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


"""
    Matrix constants
"""
MATRIX_SIZE = 16
MATRIX_CATEGORIES = ['urban', 'rural']
MATRIX_CONTACT_TYPE = ['home', 'school', 'work', 'others']
MATRIX_COUNTRY_ISO = 'CA'

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
    Ontario age-specific parameters, need to update
"""

# TODO: Do some research and update the following parameters

ONT_CASE_DISTRIBUTION = np.ones(shape=(16, ), dtype=float) / 16
ONT_HOSP_DISTRIBUTION = np.ones(shape=(16, ), dtype=float) / 16
ONT_DEATH_DISTRIBUTION = np.ones(shape=(16, ), dtype=float) / 16

"""
    Age-specific parameters, all 16 entries long
"""

# susceptibility
SUSP_BY_AGE = np.array([1, 2])
# clinical ratio
CLINICAL_BY_AGE = np.ones(shape=(16, 1), dtype=float)
# subclinical ratio
SUBCLINICAL_BY_AGE = np.zeros(shape=(16, 1), dtype=float)
# critical rate
CRIT_BY_AGE = np.zeros(shape=(16, 1), dtype=float)
# fatality rate (calculated)
CFR_BY_AGE = np.zeros(shape=(16, 1), dtype=float)
# fatality rate (calculated)
EFFICACY_BY_AGE = np.zeros(shape=(16, 1), dtype=float)

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


# TODO: Need to multiply all of the hospitalization, ICU, deaths distribution with P(HOSP), P(ICU), P(CFR)


kernel_size_3 = np.linspace(0, 15, 15)
INF2HOS_CONVOLUTION_KERNEL = normalize(norm(kernel_size_3, mean=7.5, scale=1))
HOS2ICU_CONVOLUTION_KERNEL = normalize(norm(kernel_size_3, mean=11-7.5, scale=1))

kernel_size_4 = np.linspace(0, 15, 15)
ICU2DEA_CONVOLUTION_KERNEL = normalize(gamma.pdf(kernel_size_4, a=2.5, scale=1.9))


def find_mean(v):
    tot = 0
    for i in range(len(v)):
        tot += v[i]
        if tot >= 0.5:
            print(i, len(v))
            break


if __name__ == '__main__':
    print(kernel_size_2)
    print(ICU2DEA_CONVOLUTION_KERNEL)
    find_mean(ICU2DEA_CONVOLUTION_KERNEL)
    plt.plot(kernel_size_4, ICU2DEA_CONVOLUTION_KERNEL)
    plt.show()

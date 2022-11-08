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
matrix_size = 16
matrix_categories = ['urban', 'rural']
matrix_contact = ['home', 'school', 'work', 'others']
matrix_country_ISO = 'CA'

"""
    Geo constants
"""
province_name = 'Ontario'
num_phu = 34
num_county = 528
num_district = 49

"""
    Epidemiological constants
"""

first_day = datetime.datetime(2020, 1, 15)

# susceptibility
sup_by_age = []
# clinical ratio
clinical_rate = []
# subclinical ratio
subclinical_rate = []
# fatality rate (calculated)
cfr_by_age = []
# critical rate
crit_by_age = []

"""
    Vaccine constants
"""
vaccine_age_band = ['05-11yrs', '12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs', '50-59yrs', '60-69yrs', '70-79yrs',
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

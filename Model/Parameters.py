"""
    Version
"""
from scipy.stats import gamma
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

x = np.linspace(0, 40, 100)
E2I_CONVOLUTION_KERNEL = gamma.pdf(x, a=5, scale=3)
I2D_CONVOLUTION_KERNEL = []


if __name__ == '__main__':
    print(x)
    print(E2I_CONVOLUTION_KERNEL)
    plt.plot(x, E2I_CONVOLUTION_KERNEL)
    plt.show()

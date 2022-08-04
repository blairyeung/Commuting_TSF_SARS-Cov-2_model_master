import math


def difference_of_gaussian(lst):
    """
        Perform difference of Gaussian ()
        :param lst:
        :return:
    """
    simga_1 = 0.01
    sigma_2 = 0.8
    age_bands = [5, 12, 18, 29, 39, 49, 59, 69, 79, 89]

    for age in range(100):
        cuml = 0
        for mu in range(len(age_bands)-1):
            val = lst[mu]
            mean = (age_bands[mu] + age_bands[mu+1]) / 2
            diff = age - mean
            beta = 1
            sqrtpi = math.sqrt(math.pi)
            add = (beta * math.e ^ (- (diff ^ 2) / (2 * (simga_1 ^ 2)))) / (simga_1 * sqrtpi) * val
            minus = - (beta * math.e ^ (- (diff ^ 2) / (2 * (sigma_2 ^ 2)))) / (sigma_2 * sqrtpi) * val
            cuml += (add + minus)

    return


def integrate(begin: int, end: int):
    return


def age_dog_algo(lst):
    difference_of_gaussian(lst)
    return
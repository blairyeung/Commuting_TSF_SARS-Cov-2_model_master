import math

import Parameters


def difference_of_gaussian(lst):
    """
        Perform difference of Gaussian ()
        :param lst:
        :return:
    """
    sigma_1 = 10
    sigma_2 = 30
    age_bands = [5, 12, 18, 30, 40, 50, 60, 70, 80, 90]
    cont_dist = [0] * 90
    discr_dist = [0] * Parameters.matrix_size
    for age in range(90):
        cuml = 0
        for mu in range(len(age_bands)-1):
            val = lst[mu]
            mean = (age_bands[mu] + age_bands[mu+1]) / 2.0
            diff = age - mean
            beta = 1.0
            sqrtpi = math.sqrt(math.pi)
            add = (beta * math.e ** (- (diff ** 2.0) / (2.0 * (sigma_1 ** 2.0)))) / (sigma_1 * sqrtpi) * val
            minus = - (beta * math.e ** (- (diff ** 2.0) / (2.0 * (sigma_2 ** 2.0)))) / (sigma_2 * sqrtpi) * val
            minus = 0
            cuml += (add + minus)
        cont_dist[age] = cuml

    for age in range(90):
        cont_dist[age] = cont_dist[age] * 6.7

    matrix_16_bands = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 90]
    for i in range(Parameters.matrix_size):
        discr_dist[i] = integrate(cont_dist, matrix_16_bands[i], matrix_16_bands[i+1])

    # Compensation for the last entry for not having a right approximation
    discr_dist[15] *= 1.24

    for i in discr_dist:
        # print(i)
        pass

    return discr_dist


def integrate(lst: list, begin: int, end: int):
    tot = sum(lst[begin:end]) / (end - begin)
    return tot


def age_dog_algo(lst):
    if sum(lst) == 0:
        return [0.0] * 16
    else:
        return difference_of_gaussian(lst)

age_dog_algo([0.9999
, 0.9999
, 0.9999
, 0.9999
, 0.9999
, 0.9999
, 0.9999
, 0.9999
, 0.9999
])
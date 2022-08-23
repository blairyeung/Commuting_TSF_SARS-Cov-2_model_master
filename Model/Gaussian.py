import math


def difference_of_gaussian(lst):
    """
        Perform difference of Gaussian ()
        :param lst:
        :return:
    """
    simga_1 = 1
    sigma_2 = 0.8
    age_bands = [5, 12, 18, 30, 40, 50, 60, 70, 80, 90]
    cont_dist = [0] * 100
    for age in range(100):
        cuml = 0
        for mu in range(len(age_bands)-1):
            val = lst[mu]
            mean = (age_bands[mu] + age_bands[mu+1]) / 2.0
            diff = age - mean
            beta = 1.0
            sqrtpi = math.sqrt(math.pi)
            add = (beta * math.e ** (- (diff ** 2.0) / (2.0 * (simga_1 ** 2.0)))) / (simga_1 * sqrtpi) * val
            # minus = - (beta * math.e ** (- (diff ** 2.0) / (2.0 * (sigma_2 ** 2.0)))) / (sigma_2 * sqrtpi) * val
            minus = 0
            cuml += (add + minus)
        cont_dist[age] = cuml

    tot_unormalized = sum(cont_dist)
    tot_org = sum(lst)
    for age in range(100):
        cont_dist[age] * (tot_unormalized / tot_org)
    print(cont_dist)
    return


def integrate(begin: int, end: int):
    return


def age_dog_algo(lst):
    if sum(lst) == 0:
        return [0.0] * 16
    else:
        cont_dist = difference_of_gaussian(lst)
    return

age_dog_algo([0, 1, 0, 1, 0, 0, 0 ,0 ,0])
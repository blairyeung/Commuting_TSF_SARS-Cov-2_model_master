import os
import numpy as np

import Parameters

county_data = np.zeros((Parameters.num_county, 3), dtype=int)
commute_matrix = np.zeros((Parameters.num_county, Parameters.num_county), dtype=int)


age_to_population = list()
county_codes = list()
matrix_by_class = [[None] * 4, [None] * 4]


code_to_name = dict()
code_to_phu = dict()
phu_to_code = dict()
code_to_index = dict()

band_to_population = dict()
date_to_cases_by_phu = dict()


def get_dependency_path():
    """
    :return: path of the dependency files
    """
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    return path


def read_files():
    read_matrix()
    read_county_data()
    read_phu()
    read_commute_matrix()
    read_age()
    read_cases()


def read_matrix():
    """
        matrix: 16 * 16 float-valued np.array
        matrix_by_class: 2 * 4 np.array-valued list
    """
    read_path = get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
    for category in Parameters.matrix_categories:
        for contact in Parameters.matrix_contact:
            matrix = np.zeros((Parameters.matrix_size, Parameters.matrix_size), dtype=float)
            matrix_path = read_path + category + '/' + contact + '/' + Parameters.matrix_country_ISO + '.csv'
            with open(matrix_path) as file:
                contents = file.read()
            lines = contents.split('\n')
            for i in range(1, len(lines) - 1):
                elements = lines[i].split(',')[1:]
                for j in range(Parameters.matrix_size):
                    matrix[i-1][j] = float(elements[j])
            file.close()
            cate_ind = Parameters.matrix_categories.index(category)
            cont_ind = Parameters.matrix_contact.index(contact)
            matrix_by_class[cate_ind][cont_ind] = matrix
    return


def read_county_data():
    """
        county_data: 520 * 3 float-valued np.array
        county_data[0:520][0] is the county code
        county_data[0:520][1] is the district code
        county_data[0:520][2] is the population code
    """
    read_path = get_dependency_path() + 'Ontario_county_data.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        county_data[line - 1] = [elements[0], elements[5], elements[4]]
        code_to_name[elements[0]] = elements[1]
        county_codes.append(elements[0])
        code_to_index[elements[0]] = line - 1
    file.close()
    return


def read_phu():
    read_path = get_dependency_path() + 'district_to_phu.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = [lines[line][:lines[line].index(',')], lines[line][lines[line].index(',')+1:]]
        code_to_phu[int(elements[0])] = elements[1]
        if elements[1] not in phu_to_code:
            phu_to_code[elements[1]] = [elements[0]]
        else:
            phu_to_code[elements[1]].append(elements[0])
    file.close()
    return


def read_commute_matrix():
    read_path = get_dependency_path() + 'Ontario_commute.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        commute_matrix[code_to_index[elements[0]]][code_to_index[elements[2]]] = int(elements[6])
    file.close()
    return


def read_age():
    read_path = get_dependency_path() + 'population_by_age.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        age_to_population.append(float(elements[1]))
        band_to_population[elements[0]] = float(elements[1])
    file.close()
    return


def read_cases():
    # TODO: Read the file 'All case trends data.csv' to a dictionary in the following form
    """
        date_to_cases_by_phu = {'Eastern Ontario Health Unit': sub_arary = np.array, ... }
        where sub_arary is a one-dimensional array, in the form of
        sub_arary = [0.7, ...] where each entry is the ratio of infection
    :return:
    """
    read_path = get_dependency_path() + 'All case trends data.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        print(elements)
    file.close()
    return


read_files()
print(phu_to_code)

import os
import numpy as np

import Parameters

county_data = np.zeros((Parameters.num_county, 3), dtype=int)
matrix_by_class = [[None] * 4, [None] * 4]


code_to_name = dict()
code_to_phu = dict()


def get_dependency_path():
    """
    :return: path of the dependency files
    """
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    return path


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
        county_dstata[0:520][1] is the district code
        county_data[0:520][2] is the population code
    """
    read_path = get_dependency_path() + 'GeoCode.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        county_data[line - 1] = [elements[0], elements[5], elements[4]]
        code_to_name[elements[0]] = elements[1]
    file.close()
    return


def read_phu():
    read_path = get_dependency_path() + 'district_to_phu.csv'
    with open(read_path) as file:
        contents = file.read()
    lines = contents.split('\n')
    for line in range(1, len(lines) - 1):
        elements = lines[line].split(',')
        code_to_phu[int(elements[0])] = elements[1]
    file.close()
    return


def read_commute_matrix():
    return


read_matrix()
read_county_data()
read_phu()

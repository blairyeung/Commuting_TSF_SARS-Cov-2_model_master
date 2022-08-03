import os
import numpy as np

import Parameters

code_to_name = dict()
county_data = np.zeros((Parameters.num_county, 3), dtype=int)

def get_dependency_path():
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    return path

def read_matrix():
    # TODO: Read the matrices in the form of 2D np.array, and put them in a 2D list.
    """
        matrix: 16 * 16 float-valued np.array
        matrix_by_class: 2 * 4 np.array-valued list
    """
    matrix_by_class = []
    read_path = get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
    for category in Parameters.matrix_categories:
        for contact in Parameters.matrix_contact:
            matrix_path = read_path + category + '/' + contact + '/' + Parameters.matrix_country_ISO + '.csv'
            print(matrix_path)
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
    for line in range(1, len(lines)-1):
        elements = lines[line].split(',')
        county_data[line-1] = [elements[0], elements[5], elements[4]]
        code_to_name[elements[0]] = elements[1]
    return

read_county_data()
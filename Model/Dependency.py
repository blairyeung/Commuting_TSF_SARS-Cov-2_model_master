import os
import numpy as np

import Parameters


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
    # TODO: Read the data of each county in 2D np.array
    """
        county_data: 520 * 4 float-valued np.array
    """
    county_data = np.array([520, 4])
    return

read_matrix()
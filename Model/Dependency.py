import os
import numpy

import Parameters


def get_dependency_path():
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    return path

def read_matrix():
    read_path = get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
    for category in Parameters.matrix_categories:
        for contact in Parameters.matrix_contact:
            matrix_path = read_path + category + '/' + contact + '/' + Parameters.matrix_country_ISO + '.csv'
            print(matrix_path)
    return

read_matrix()
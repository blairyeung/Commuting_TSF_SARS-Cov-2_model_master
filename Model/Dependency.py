import os
import numpy

def get_dependency_path():
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    return path

def read_matrix():
    matrix_path = get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
    print(matrix_path)
    return

read_matrix()
import math
import numpy as np


class Model:

    def __init__(self):
        self._get_today_cases()

    def one_cycle(self, epidemiological_data):
        self._one_cycle_county(epidemiological_data)

    def _one_cycle_county(self, epidemiological_data):
        for i in range(len(epidemiological_data)):
            return self._get_today_cases(epidemiological_data[i])


    def _get_commuting_matrix(self, matrix):
        return None


    def _get_today_new_cases(self, cases, contact_matrix, immunity):
        """
        Computer the number of cases in a specific county in a day
        :param cases:
        :param contact_matrix:
        :param immunity:
        :return: a 16-dimensional vector
        """
        contacts = np.prod(contact_matrix, cases)
        effective_contact = np.multiply(contacts, immunity)
        return effective_contact

    def _get_today_new_recovered(self, cases, contact_matrix, immunity):
        """
        Computer the number of cases in a specific county in a day
        :param cases:
        :param contact_matrix:
        :param immunity:
        :return: a 16-dimensional vector
        """
        # TODO: Add the number of recovered cases

    def _get_today_new_deaths(self, cases, contact_matrix, immunity):
        """
        Computer the number of cases in a specific county in a day
        :param cases:
        :param contact_matrix:
        :param immunity:
        :return: a 16-dimensional vector
        """
        # TODO: Add the number of deaths
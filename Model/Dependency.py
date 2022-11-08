from datetime import datetime
import numpy as np
import os

import Gaussian
import Parameters

class Dependency:
    county_data = np.zeros((Parameters.num_county, 3), dtype=int)
    commute_matrix = np.zeros((Parameters.num_county, Parameters.num_county), dtype=int)

    total_days = 0

    age_to_population = list()
    county_codes = list()
    matrix_by_class = [[None] * 4, [None] * 4]

    code_to_name = dict()
    code_to_phu = dict()
    phu_to_code = dict()
    code_to_index = dict()

    band_to_population = dict()
    date_to_cases_by_phu = dict()
    date_to_vaccines_by_age = np.zeros((0, 3, 9))

    def __init__(self):
        self.read_files()

    def get_dependency_path(self):
        """
        :return: path of the dependency files
        """
        path = os.getcwd()[:-5] + 'Model Dependencies/'
        return path

    def read_files(self):
        self.read_matrix()
        self.read_county_data()
        self.read_phu()
        self.read_commute_matrix()
        self.read_age()
        self.read_cases()
        self.read_vaccine()
        self.reshape_vaccine()
        self.differentiate()
        print(self.date_to_cases_by_phu)
        for i in range(len(self.date_to_cases_by_phu.keys())):
            print(i)
            # print(list(self.date_to_cases_by_phu.keys()))
            # print(list(self.date_to_cases_by_phu.keys())[i])
            # print(list(self.phu_to_code.keys())[i])

    def read_matrix(self):
        """
            matrix: 16 * 16 float-valued np.array
            matrix_by_class: 2 * 4 np.array-valued list
        """
        read_path = self.get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
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
                        matrix[i - 1][j] = float(elements[j])
                file.close()
                cate_ind = Parameters.matrix_categories.index(category)
                cont_ind = Parameters.matrix_contact.index(contact)
                self.matrix_by_class[cate_ind][cont_ind] = matrix
        return

    def read_county_data(self):
        """
            county_data: 520 * 3 float-valued np.array
            county_data[0:520][0] is the county code
            county_data[0:520][1] is the district code
            county_data[0:520][2] is the population code
        """
        read_path = self.get_dependency_path() + 'Ontario_county_data.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            self.county_data[line - 1] = [elements[0], elements[5], elements[4]]
            self.code_to_name[elements[0]] = elements[1]
            self.county_codes.append(elements[0])
            self.code_to_index[elements[0]] = line - 1
        file.close()
        return

    def read_phu(self):
        read_path = self.get_dependency_path() + 'district_to_phu.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = [lines[line][:lines[line].index(',')], lines[line][lines[line].index(',') + 1:]]
            self.code_to_phu[int(elements[0])] = elements[1]
            if elements[1] not in self.phu_to_code:
                self.phu_to_code[elements[1]] = [elements[0]]
            else:
                self.phu_to_code[elements[1]].append(elements[0])
        file.close()
        return

    def read_commute_matrix(self):
        read_path = self.get_dependency_path() + 'Ontario_commute.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            self.commute_matrix[self.code_to_index[elements[0]]][self.code_to_index[elements[2]]] = int(elements[6])
        file.close()
        return

    def read_age(self):
        read_path = self.get_dependency_path() + 'population_by_age.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            self.age_to_population.append(float(elements[1]))
            self.band_to_population[elements[0]] = float(elements[1])
        file.close()
        return

    def read_cases(self):
        """
               date_to_cases_by_phu = {'Eastern Ontario Health Unit': sub_arary = np.array, ... }
               where sub_arary is a one-dimensional array, in the form of
               sub_array = [0.7, ...] where each entry is the ratio of infection
           :return:
        """
        read_path = self.get_dependency_path() + 'All case trends data.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')

        self.find_max_date()

        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            string = elements[0]
            this_day = datetime.strptime(string, '%d-%b-%y')
            after_outbreak = (this_day - Parameters.first_day).days
            phu = elements[1]
            if phu == 'Ontario':
                pass
            else:
                if phu not in self.date_to_cases_by_phu:
                    self.date_to_cases_by_phu[phu] = [0] * self.total_days
                else:
                    if elements[7] == '-':
                        self.date_to_cases_by_phu[phu][after_outbreak] = 0.0
                    else:
                        self.date_to_cases_by_phu[phu][after_outbreak] = float(elements[7])
        file.close()
        return

    def read_vaccine(self):
        read_path = self.get_dependency_path() + 'vaccines_by_age.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')

        self.find_max_date()

        # global date_to_vaccines_by_age
        self.date_to_vaccines_by_age = np.zeros((self.total_days, 3, 9))
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            string = elements[0]
            this_day = datetime.strptime(string, '%m/%d/%Y')
            after_outbreak = (this_day - Parameters.first_day).days
            band = elements[1]

            for i in [7, 8, 9]:
                if elements[i] == '':
                    elements[i] = 0

            if band in Parameters.vaccine_age_band:
                for i in [7, 8, 9]:
                    if elements[i] == '':
                        elements[i] = 0
                self.date_to_vaccines_by_age[after_outbreak - 1][0][Parameters.vaccine_age_band.index(band)] = float(
                    elements[7])
                self.date_to_vaccines_by_age[after_outbreak - 1][1][Parameters.vaccine_age_band.index(band)] = float(
                    elements[8])
                self.date_to_vaccines_by_age[after_outbreak - 1][2][Parameters.vaccine_age_band.index(band)] = float(
                    elements[9])

        file.close()

        if min(len(lines) - 1, self.total_days) != self.total_days:
            for i in range(self.total_days, len(lines) - 1):
                self.date_to_vaccines_by_age[i] = self.date_to_vaccines_by_age[self.total_days]
        return

    def reshape_vaccine(self):
        """
        reshape the 10-years age band into 5-years age band
        :return:
        """

        # global date_to_vaccines_by_age
        reshaped = np.zeros((self.total_days, 3, 16))

        for date in range(self.total_days):
            for dose in [0, 1, 2]:
                lst = self.date_to_vaccines_by_age[date][dose]
                lst = Gaussian.age_dog_algo(lst)
                reshaped[date][dose] = lst

        self.date_to_vaccines_by_age = reshaped
        return

    def differentiate(self):
        """
        Find the increment of a time-series data
        :return:
        """

        # global date_to_vaccines_by_age
        vaccine_differentiated = np.zeros((self.total_days, 3, 16))
        for dose in [0, 1, 2]:
            for age in range(Parameters.matrix_size):
                yesterday_cuml = 0.0
                for date in range(self.total_days):
                    today_cuml = self.date_to_vaccines_by_age[date][dose][age]
                    delta = today_cuml - yesterday_cuml
                    self.date_to_vaccines_by_age[date][dose][age] = delta
                    yesterday_cuml = today_cuml

        self.date_to_vaccines_by_age = vaccine_differentiated

        # global date_to_cases_by_phu

        cases_differentiated = dict()
        for phu in self.date_to_cases_by_phu:
            phu_data = self.date_to_cases_by_phu[phu]
            yesterday_cuml = 0.0
            for date in range(self.total_days):
                today_cuml = phu_data[date]
                delta = today_cuml - yesterday_cuml
                phu_data[date] = delta
                yesterday_cuml = today_cuml

        date_to_cases_by_phu = cases_differentiated

        return

    def find_max_date(self):
        read_path = self.get_dependency_path() + 'All case trends data.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        # global total_days

        # First we count the number of total days:
        max_size = 0
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            string = elements[0]
            this_day = datetime.strptime(string, '%d-%b-%y')
            after_outbreak = (this_day - Parameters.first_day).days
            max_size = max(after_outbreak, max_size)

        self.total_days = max_size

        file.close()

        read_path = self.get_dependency_path() + 'vaccines_by_age.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        max_size = 0
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            string = elements[0]
            this_day = datetime.strptime(string, '%m/%d/%Y')
            after_outbreak = (this_day - Parameters.first_day).days
            max_size = max(after_outbreak, max_size)

        self.total_days = max(max_size, self.total_days)


if __name__ == '__main__':
    dependency = Dependency()
    dependency.read_files()

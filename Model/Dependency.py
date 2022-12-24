import copy
from datetime import datetime
import numpy as np
import os
import csv

import pandas as pd

import Gaussian
import Parameters
import Util

class Dependency:
    county_data = np.zeros((Parameters.NO_COUNTY, 3), dtype=int)
    commute_matrix = np.zeros((Parameters.NO_COUNTY, Parameters.NO_COUNTY), dtype=int)

    total_days = 0

    age_to_population = list()
    county_codes = list()
    matrix_by_class = [[None] * 4, [None] * 4]

    """
        Look up table
    """

    code_to_name = dict()
    district_to_phu = dict()
    phu_to_district = dict()
    code_to_index = dict()
    county_to_district = dict()
    district_to_county = dict()

    """
        PHU-specific data
    """

    band_to_population = dict()
    date_to_incidence_rate_by_phu = dict()
    date_to_hospitalization_rate_by_phu = dict()
    date_to_death_rate_by_phu = dict()
    date_to_vaccines_by_age = np.zeros((0, 3, 9))

    """
        County-specific data
    """

    date_to_cases_by_county = None
    date_to_hospitalizations_by_county = None
    date_to_deaths_by_county = None
    date_to_vaccines_by_county = None
    date_to_ICU_by_county = None

    population_by_phu = dict()
    population_by_district = dict()
    index_to_population = list()

    age_band_to_ratio = dict()
    population_by_age_band = np.zeros(shape=(16,), dtype=int)
    ratio_by_age_band = np.zeros(shape=(16,), dtype=int)

    ontario_population = 0

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
        """
            Do not differentiate cases
        """
        self.differentiate()
        self.code_district_linking()
        self.compute_phu_population()
        self.distribute_to_counties()
        self.read_age()

    def read_matrix(self):
        """
            matrix: 16 * 16 float-valued np.array
            matrix_by_class: 2 * 4 np.array-valued list
        """
        read_path = self.get_dependency_path() + 'Matrix_IO/Matrix_by_Category/'
        for category in Parameters.MATRIX_CATEGORIES:
            for contact in Parameters.MATRIX_CONTACT_TYPE:
                matrix = np.zeros((Parameters.MATRIX_SIZE, Parameters.MATRIX_SIZE), dtype=float)
                matrix_path = read_path + category + '/' + contact + '/' + Parameters.MATRIX_COUNTRY_ISO + '.csv'
                with open(matrix_path) as file:
                    contents = file.read()
                lines = contents.split('\n')
                for i in range(1, len(lines) - 1):
                    elements = lines[i].split(',')[1:]
                    for j in range(Parameters.MATRIX_SIZE):
                        matrix[i - 1][j] = float(elements[j])
                file.close()
                cate_ind = Parameters.MATRIX_CATEGORIES.index(category)
                cont_ind = Parameters.MATRIX_CONTACT_TYPE.index(contact)
                self.matrix_by_class[cate_ind][cont_ind] = matrix
        return

    def read_county_data(self):
        """
            county_data: shape=(520, 3) float-valued np.array
            county_data[0:520][0] is the county code
            county_data[0:520][1] is the district code
            county_data[0:520][2] is the population
        """
        read_path = self.get_dependency_path() + 'Ontario_county_data.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            self.county_data[line - 1] = [int(elements[0]), int(elements[5]), int(elements[4])]
            self.code_to_name[int(elements[0])] = elements[1]
            self.county_codes.append(int(elements[0]))
            self.code_to_index[int(elements[0])] = line - 1
        file.close()
        self.county_data = np.array(self.county_data)
        self.index_to_population = copy.deepcopy(self.county_data).transpose(1, 0)[2]
        # self.index_to_population = self.index_to_population.reshape(3, self.county_data.shape[0])[2]
        return

    def read_phu(self):
        read_path = self.get_dependency_path() + 'district_to_phu.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = [lines[line][:lines[line].index(',')], lines[line][lines[line].index(',') + 1:]]
            elements[1] = elements[1].replace('"', '')
            self.district_to_phu[int(elements[0])] = elements[1]
            if elements[1] not in self.phu_to_district:
                self.phu_to_district[elements[1]] = [int(elements[0])]
            else:
                self.phu_to_district[elements[1]].append(int(elements[0]))
        file.close()
        return

    def read_commute_matrix(self):
        read_path = self.get_dependency_path() + 'Ontario_commute.csv'
        with open(read_path) as file:
            contents = file.read()
        lines = contents.split('\n')
        for line in range(1, len(lines) - 1):
            elements = lines[line].split(',')
            self.commute_matrix[self.code_to_index[int(elements[0])]][self.code_to_index[int(elements[2])]] = \
                int(elements[6])
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
               date_to_incidence_rate_by_phu = {'Eastern Ontario Health Unit': sub_arary = np.array, ... }
               where sub_arary is a one-dimensional array, in the form of
               sub_array = [0.7, ...] where each entry is the ratio of infection
           :return:
        """
        read_path = self.get_dependency_path() + 'All case trends data.csv'
        with open(read_path, newline='') as file:
            reader = csv.reader(file, delimiter=',')
            lines = list(reader)

        self.find_max_date()

        last_recorded = 0

        for line in range(1, len(lines) - 1):
            elements = lines[line]
            elements[1] = elements[1].replace('"', '')
            string = elements[0]
            this_day = datetime.strptime(string, '%d-%b-%y')
            after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
            phu = elements[1]
            if after_outbreak > last_recorded:
                last_recorded = after_outbreak
            if phu == 'Ontario':
                pass
            else:
                if phu not in self.date_to_incidence_rate_by_phu:
                    self.date_to_incidence_rate_by_phu[phu] = np.zeros(shape=(self.total_days, ), dtype=float)
                    self.date_to_hospitalization_rate_by_phu[phu] = np.zeros(shape=(self.total_days, ), dtype=float)
                    self.date_to_death_rate_by_phu[phu] = np.zeros(shape=(self.total_days, ), dtype=float)
                else:
                    if elements[7] == '-':
                        self.date_to_incidence_rate_by_phu[phu][after_outbreak] = 0.0
                        self.date_to_hospitalization_rate_by_phu[phu][after_outbreak] = 0.0
                        self.date_to_death_rate_by_phu[phu][after_outbreak] = 0.0
                    else:
                        self.date_to_incidence_rate_by_phu[phu][after_outbreak] = float(elements[7])
                        self.date_to_hospitalization_rate_by_phu[phu][after_outbreak] = float(elements[9])
                        self.date_to_death_rate_by_phu[phu][after_outbreak] = float(elements[11])

        for i in range(last_recorded-1, self.total_days):
            for phu in self.date_to_incidence_rate_by_phu.keys():
                self.date_to_incidence_rate_by_phu[phu][i] = self.date_to_incidence_rate_by_phu[phu][last_recorded]

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
            after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
            band = elements[1]

            for i in [7, 8, 9]:
                if elements[i] == '':
                    elements[i] = 0

            if band in Parameters.VACCINE_AGE_BANDS:
                for i in [7, 8, 9]:
                    if elements[i] == '':
                        elements[i] = 0
                self.date_to_vaccines_by_age[after_outbreak - 1][0][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
                    elements[7])
                self.date_to_vaccines_by_age[after_outbreak - 1][1][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
                    elements[8])
                self.date_to_vaccines_by_age[after_outbreak - 1][2][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
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
        """
                for dose in [0, 1, 2]:
            for age in range(Parameters.MATRIX_SIZE):
                yesterday_cuml = 0.0
                for date in range(self.total_days):
                    today_cuml = self.date_to_vaccines_by_age[date][dose][age]
                    delta = today_cuml - yesterday_cuml
                    self.date_to_vaccines_by_age[date][dose][age] = delta
                    yesterday_cuml = today_cuml

        """


        for i in range(self.total_days - 1):

            # TODO: CHANGE THIS!!!!

            vaccine_differentiated[i+1] = self.date_to_vaccines_by_age[i+1] - self.date_to_vaccines_by_age[i]
        self.date_to_vaccines_by_age = np.clip(vaccine_differentiated, a_min=0, a_max=0.2)
        """
                cases_differentiated = dict()
        for phu in self.date_to_incidence_rate_by_phu:
            phu_data = self.date_to_incidence_rate_by_phu[phu]
            yesterday_cuml = 0.0
            for date in range(self.total_days):
                today_cuml = phu_data[date]
                delta = today_cuml - yesterday_cuml
                phu_data[date] = delta
                yesterday_cuml = today_cuml

        self.date_to_incidence_rate_by_phu = cases_differentiated
        """

        data = self.date_to_vaccines_by_age.transpose(1, 0, 2)

        dose1 = np.clip((data[0] - data[1]), a_min=0, a_max=1)
        dose2 = np.clip((data[1] - data[2]), a_min=0, a_max=1)
        dose3 = (data[2])

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
            after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
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
            after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
            max_size = max(after_outbreak, max_size)

        self.total_days = max(max_size, self.total_days)

    def code_district_linking(self):
        for i in self.county_data:
            county = i[0]
            district = i[1]
            population = i[2]
            if district not in self.district_to_county:
                self.district_to_county[district] = list()
                self.population_by_district[district] = 0
            else:
                pass
            self.district_to_county[district].append(county)
            self.population_by_district[district] += population
            self.county_to_district[county] = district

    def compute_phu_population(self):
        for i in self.phu_to_district.keys():
            districts = self.phu_to_district[i]
            self.population_by_phu[i] = 0
            for d in districts:
                self.population_by_phu[i] += self.population_by_district[d]

    def distribute_to_counties(self):
        self.date_to_cases_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 16), dtype=float)
        self.date_to_hospitalizations_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 16),
                                                           dtype=float)
        self.date_to_ICU_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 16), dtype=float)
        self.date_to_deaths_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 16), dtype=float)
        self.date_to_vaccines_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 3, 16), dtype=float)

        # print(self.date_to_incidence_rate_by_phu.keys())
        for i in range(len(self.county_data)):
            county = self.county_data[i][0]
            district = self.county_data[i][1]
            population = self.county_data[i][2]
            phu = self.district_to_phu[district]

            # print(population)

            incidences = self.date_to_incidence_rate_by_phu[phu].reshape(self.total_days, 1)
            cases_ratio = Parameters.ONT_CASE_DISTRIBUTION.reshape(16, 1)
            self.date_to_cases_by_county[i] = np.matmul(incidences, cases_ratio.T) * population / 100000.0

            hospitalizations = self.date_to_hospitalization_rate_by_phu[phu].reshape(self.total_days, 1)

            hospitalization_ratio = Parameters.ONT_HOSP_DISTRIBUTION.reshape(16, 1)
            self.date_to_hospitalizations_by_county[i] = np.matmul(hospitalizations, hospitalization_ratio.T) * \
                                                         population / 100000.0


            # THESE ARE INCORRECT!!!!!!!
            # TODO: this are ratios, not the probability!

            ICU = np.convolve(hospitalizations.reshape(self.total_days),
                              Parameters.HOS2ICU_CONVOLUTION_KERNEL, mode='same').reshape(self.total_days, 1) * \
                  Parameters.ICU_HOSP

            ICU_ratio = Parameters.ONT_ICU_RATIO.reshape(16, 1)

            self.date_to_ICU_by_county[i] =  np.matmul(ICU, ICU_ratio.T) * population / 100000.0

            deaths = self.date_to_death_rate_by_phu[phu].reshape(self.total_days, 1)
            deaths_ratio = Parameters.ONT_DEATH_DISTRIBUTION.reshape(16, 1)
            self.date_to_deaths_by_county[i] = np.matmul(deaths, deaths_ratio.T) * population / 100000.0

            self.date_to_vaccines_by_county[i] = ((population / 100.0) * self.date_to_vaccines_by_age)

    def read_age(self):
        read_path = self.get_dependency_path() + '1710000501-eng.csv'
        df = pd.read_csv(read_path)
        population = np.array(list(df.Persons), dtype=int)
        self.ontario_population = population[0]
        self.population_by_age_band[0:15] = population[1:16]
        self.population_by_age_band[15] = np.sum(population[16:])
        self.ratio_by_age_band = Util.normalize(self.population_by_age_band)


if __name__ == '__main__':
    dependency = Dependency()
    pass

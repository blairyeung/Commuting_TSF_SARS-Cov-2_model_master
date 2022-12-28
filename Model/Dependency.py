import copy
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
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
    date_to_vaccines_by_age_un_reshaped = np.zeros((0, 3, 9))
    date_to_vaccines_by_age_un_differentaited = np.zeros((0, 3, 9))

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

    raw_mobility = None

    mobility = None

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
        self.read_mobility()
        self.mobility_reshape()

    def read_mobility(self):
        path = self.get_dependency_path()
        read_path = path + 'Region_Mobility_Report_CSVs/'
        path_2020 = read_path + '2020_CA_Region_Mobility_Report.csv'
        path_2021 = read_path + '2021_CA_Region_Mobility_Report.csv'
        path_2022 = read_path + '2022_CA_Region_Mobility_Report.csv'
        df1 = pd.read_csv(path_2020)
        df2 = pd.read_csv(path_2021)
        df3 = pd.read_csv(path_2022)

        df = pd.concat([df1, df2, df3])

        df['date'] = pd.to_datetime(df['date'])

        groups = df.groupby('iso_3166_2_code')
        Ontario = groups.get_group('CA-ON')
        max_date = (Ontario['date'].max() - Parameters.OUTBREAK_FIRST_DAY).days
        min_date = (Ontario['date'].min() - Parameters.OUTBREAK_FIRST_DAY).days

        mobility = np.zeros(shape=(6, 3000))

        retail = Ontario['retail_and_recreation_percent_change_from_baseline'].to_numpy()
        grocery = Ontario['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()
        park = Ontario['parks_percent_change_from_baseline'].to_numpy()
        trainsit = Ontario['transit_stations_percent_change_from_baseline'].to_numpy()
        workplace = Ontario['workplaces_percent_change_from_baseline'].to_numpy()
        residential = Ontario['residential_percent_change_from_baseline'].to_numpy()
        mobility[0][min_date:max_date + 1] = retail
        mobility[1][min_date:max_date + 1] = grocery
        mobility[2][min_date:max_date + 1] = park
        mobility[3][min_date:max_date + 1] = trainsit
        mobility[4][min_date:max_date + 1] = workplace
        mobility[5][min_date:max_date + 1] = residential
        mobility = mobility / 100
        self.raw_mobility = mobility.T
        return

    def mobility_reshape(self):
        year_forecast = 10
        summer_break_day_start = datetime.strptime('2020-04-25', '%Y-%m-%d')
        summer_break_day_end = datetime.strptime('2020-09-04', '%Y-%m-%d')

        winter_break_day_start = datetime.strptime('2020-12-20', '%Y-%m-%d')
        winter_break_day_end = datetime.strptime('2021-01-09', '%Y-%m-%d')

        summer_break_start = (summer_break_day_start - Parameters.OUTBREAK_FIRST_DAY).days
        summer_break_end = (summer_break_day_end - Parameters.OUTBREAK_FIRST_DAY).days

        christmas_start = (winter_break_day_start - Parameters.OUTBREAK_FIRST_DAY).days
        christmas_end = (winter_break_day_end - Parameters.OUTBREAK_FIRST_DAY).days

        count = 0

        work = np.mean(self.raw_mobility.T[3:5], axis=0).T
        residential = self.raw_mobility.T[5]
        other = np.mean(self.raw_mobility.T[0:2], axis=0).T
        school = np.zeros(shape=(3000,))

        while count * 365 + summer_break_start < 3000:
            count += 1
            start = count * 365 + summer_break_start
            end = min(count * 365 + summer_break_end, 3000)
            # print(start, end)
            school[start:end] = - 0.65
            pass

        count = 0

        while count * 365 + christmas_start < 3000:
            count += 1
            start = count * 365 + christmas_start
            end = min(count * 365 + christmas_end, 3000)
            # print(start, end)
            school[start:end] = - 0.65
            pass

        school = np.reshape(cv2.GaussianBlur(school.reshape(3000, 1), (7, 7), 0), newshape=(3000,)) \
                 - 0.3 * np.ones(shape=school.shape)

        conct = np.concatenate([residential.reshape(3000, 1), school.reshape(3000, 1),
                                work.reshape(3000, 1), other.reshape(3000, 1)], axis=1)
        self.mobility = conct + np.ones(shape=conct.shape)

        # print(np.max(self.mobility), np.min(self.mobility))
        # print(np.max(self.raw_mobility), np.min(self.raw_mobility))

        return

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
            this_day = datetime.strptime(string, '%B %d, %Y')
            after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
            phu = elements[1]
            if after_outbreak > last_recorded:
                last_recorded = after_outbreak
            if phu == 'Ontario':
                pass
            else:
                if phu not in self.date_to_incidence_rate_by_phu:
                    self.date_to_incidence_rate_by_phu[phu] = np.zeros(shape=(self.total_days,), dtype=float)
                    self.date_to_hospitalization_rate_by_phu[phu] = np.zeros(shape=(self.total_days,), dtype=float)
                    self.date_to_death_rate_by_phu[phu] = np.zeros(shape=(self.total_days,), dtype=float)
                else:
                    if elements[7] == '-':
                        self.date_to_incidence_rate_by_phu[phu][after_outbreak] = 0.0
                        self.date_to_hospitalization_rate_by_phu[phu][after_outbreak] = 0.0
                        self.date_to_death_rate_by_phu[phu][after_outbreak] = 0.0
                    else:
                        self.date_to_incidence_rate_by_phu[phu][after_outbreak] = float(elements[7])
                        self.date_to_hospitalization_rate_by_phu[phu][after_outbreak] = float(elements[9])
                        self.date_to_death_rate_by_phu[phu][after_outbreak] = float(elements[11])

        for i in range(last_recorded - 1, self.total_days):
            for phu in self.date_to_incidence_rate_by_phu.keys():
                self.date_to_incidence_rate_by_phu[phu][i] = self.date_to_incidence_rate_by_phu[phu][last_recorded]

        file.close()
        return

    def read_vaccine(self):
        read_path = self.get_dependency_path() + 'vaccine_by_age_auto_update.csv'
        vaccine_df = pd.read_csv(read_path)

        vaccine_df['Percent_at_least_one_dose'] = vaccine_df['Percent_at_least_one_dose'].fillna(0)
        vaccine_df['Percent_fully_vaccinated'] = vaccine_df['Percent_fully_vaccinated'].fillna(0)
        vaccine_df['Percent_3doses'] = vaccine_df['Percent_3doses'].fillna(0)

        vaccine_df['Date'] = pd.to_datetime(vaccine_df['Date'])

        self.find_max_date()

        self.date_to_vaccines_by_age = np.zeros((self.total_days, 3, 9))

        for i in range(len(vaccine_df)):
            row = vaccine_df.iloc[i]
            after_outbreak = (row['Date'] - Parameters.OUTBREAK_FIRST_DAY).days
            band = row['Agegroup']

            if band in Parameters.VACCINE_AGE_BANDS:
                self.date_to_vaccines_by_age[after_outbreak - 1][0][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
                    row['Percent_at_least_one_dose'])
                self.date_to_vaccines_by_age[after_outbreak - 1][1][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
                    row['Percent_fully_vaccinated'])
                self.date_to_vaccines_by_age[after_outbreak - 1][2][Parameters.VACCINE_AGE_BANDS.index(band)] = float(
                    row['Percent_3doses'])

        return

    def reshape_vaccine(self):
        """
        reshape the 10-years age band into 5-years age band
        :return:
        """

        # global date_to_vaccines_by_age

        self.date_to_vaccines_by_age_un_reshaped = copy.deepcopy(self.date_to_vaccines_by_age)

        # print(self.date_to_vaccines_by_county.shape)

        reshaped = np.zeros((self.total_days, 3, 16))

        for date in range(self.total_days):
            for dose in [0, 1, 2]:
                lst = self.date_to_vaccines_by_age[date][dose]
                lst = Gaussian.age_dog_algo(lst)
                reshaped[date][dose] = lst

        self.date_to_vaccines_by_age = reshaped

        self.date_to_vaccines_by_age_un_differentaited = copy.deepcopy(self.date_to_vaccines_by_age )

        return

    def differentiate(self):
        """
        Find the increment of a time-series data
        :return:
        """

        vaccine_differentiated = np.zeros((self.total_days, 3, 16))

        data = self.date_to_vaccines_by_age.transpose(1, 0, 2)
        dose1 = np.clip((data[0] - data[1]), a_min=0, a_max=1)
        dose2 = np.clip((data[1] - data[2]), a_min=0, a_max=1)

        # dose1 = data[0]
        # dose2 = data[1]
        dose3 = data[2]

        self.date_to_vaccines_by_age = np.array([dose1, dose2, dose3]).transpose(1, 0, 2)

        for i in range(self.total_days - 1):
            vaccine_differentiated[i + 1] = self.date_to_vaccines_by_age[i + 1] - self.date_to_vaccines_by_age[i]

        self.date_to_vaccines_by_age = vaccine_differentiated

        read_path = self.get_dependency_path() + 'vaccine_by_age_admin.csv'
        vaccine_df_dose_admin = pd.read_csv(read_path)

        vaccine_df_dose_admin = vaccine_df_dose_admin.fillna(0)

        vaccine_df_dose_admin['report_date'] = pd.to_datetime(vaccine_df_dose_admin['report_date'])

        for i in range(len(vaccine_df_dose_admin)):
            row = vaccine_df_dose_admin.iloc[i]
            after_outbreak = (row['report_date'] - Parameters.OUTBREAK_FIRST_DAY).days
            more_dose = 0
            if row['previous_day_3doses'] != 0:
                more_dose = row['previous_day_total_doses_administered'] - row['previous_day_at_least_one'] - \
                            row['previous_day_fully_vaccinated'] - row['previous_day_3doses']
            if np.sum(self.date_to_vaccines_by_age[after_outbreak - 1][2]) != 0:
                ratio = more_dose * self.date_to_vaccines_by_age[after_outbreak - 1][2] / \
                        np.sum(self.date_to_vaccines_by_age[after_outbreak - 1][2])
            else:
                ratio = more_dose * Parameters.ONT_AGE_BAND_POPULATION / Parameters.ONT_POPULATOIN

            vaccine_raito = ratio / Parameters.ONT_AGE_BAND_POPULATION

            # print(self.date_to_vaccines_by_age[after_outbreak - 1][2])

            self.date_to_vaccines_by_age[after_outbreak - 1][2] = self.date_to_vaccines_by_age[after_outbreak - 1][2] \
                                                                  + vaccine_raito

        # vaccine_raito print('new', self.date_to_vaccines_by_age[after_outbreak - 1][2] )

        self.date_to_vaccines_by_age = np.clip(vaccine_differentiated, a_min=0, a_max=0.2)

        return

    def find_max_date(self):
        case_path = self.get_dependency_path() + 'All case trends data.csv'
        vaccine_path = self.get_dependency_path() + 'vaccine_by_age_auto_update.csv'

        cases = pd.read_csv(case_path)
        vaccines = pd.read_csv(vaccine_path)

        # First we count the number of total days:

        cases['Date'] = pd.to_datetime(cases['Date'])
        case_max_size = (cases['Date'].max() - Parameters.OUTBREAK_FIRST_DAY).days

        vaccines['Date'] = pd.to_datetime(vaccines['Date'])
        vaccine_max_size = (vaccines['Date'].max() - Parameters.OUTBREAK_FIRST_DAY).days

        after_outbreak = max(case_max_size, vaccine_max_size)

        self.total_days = after_outbreak

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

            ICU = np.convolve(hospitalizations.reshape(self.total_days),
                              Parameters.HOS2ICU_CONVOLUTION_KERNEL, mode='same').reshape(self.total_days, 1) * \
                  Parameters.ICU_HOSP

            ICU_ratio = Parameters.ONT_ICU_RATIO.reshape(16, 1)

            self.date_to_ICU_by_county[i] = np.matmul(ICU, ICU_ratio.T) * population / 100000.0

            deaths = self.date_to_death_rate_by_phu[phu].reshape(self.total_days, 1)
            deaths_ratio = Parameters.ONT_DEATH_DISTRIBUTION.reshape(16, 1)
            self.date_to_deaths_by_county[i] = np.matmul(deaths, deaths_ratio.T) * population / 100000.0

            # This is wrong, and idk why

            self.date_to_vaccines_by_county[i] = ((population) * self.date_to_vaccines_by_age)

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
    reshaped_data = dependency.date_to_vaccines_by_age_un_differentaited.transpose(1, 0, 2)[0]
    data = dependency.date_to_vaccines_by_age_un_reshaped.transpose(1, 0, 2)[0]
    print()
    plt.plot(data)
    plt.show()
    plt.plot(reshaped_data)
    plt.show()
    pass

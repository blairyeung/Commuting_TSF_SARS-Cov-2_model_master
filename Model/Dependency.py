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

    # date_to_vaccines_by_age = np.zeros((0, 3, 9))
    # date_to_vaccines_by_age_un_reshaped = np.zeros((0, 3, 9))
    # date_to_vaccines_by_age_un_differentaited = np.zeros((0, 3, 9))

    date_to_vaccines_by_phu = dict()

    phu_id_pairing = {2226: 'Algoma Public Health', 2227: 'Brant County Health Unit',
                      2230: 'Durham Region Health Department ', 2233: 'Grey Bruce Health Unit',
                      2234: 'Haldimand-Norfolk Health Unit',
                      2235: 'Haliburton, Kawartha, Pine Ridge District Health Unit ',
                      2236: 'Halton Region Public Health', 2237: 'City of Hamilton Public Health Services',
                      2238: 'Hastings Prince Edward Public Health', 2240: 'Chatham-Kent Public Health',
                      2241: 'Kingston, Frontenac and Lennox & Addington Public Health',
                      2242: 'Lambton Public Health', 2243: 'Leeds, Grenville & Lanark District Health Unit',
                      2244: 'Middlesex-London Health Unit', 2246: 'Niagara Region Public Health',
                      2247: 'North Bay Parry Sound District Health Unit', 2249: 'Northwestern Health Unit',
                      2251: 'Ottawa Public Health', 2253: 'Peel Public Health', 2255: 'Peterborough Public Health ',
                      2256: 'Porcupine Health Unit', 2257: 'Renfrew County and District Health Unit',
                      2258: 'Eastern Ontario Health Unit', 2260: 'Simcoe Muskoka District Health Unit ',
                      2261: 'Public Health Sudbury & Districts', 2262: 'Thunder Bay District Health Unit',
                      2263: 'Timiskaming Health Unit', 2265: 'Region of Waterloo Public Health and Emergency Services',
                      2266: 'Wellington-Dufferin-Guelph Public Health', 2268: 'Windsor-Essex County Health Unit ',
                      2270: 'York Region Public Health', 3895: 'Toronto Public Health',
                      4913: 'Southwestern Public Health', 5183: 'Huron Perth Health Unit', 9999: 'Unknown'}

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
        # self.read_vaccine()
        # self.read_vaccine_by_phu()
        self.read_vaccine_by_phu()
        # self.reshape_vaccine()
        self.reshape_vaccine_by_phu()
        """
            Do not differentiate cases
        """
        self.differentiate_by_phu()
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
        blurred_mobility = np.zeros(shape=(6, 3000))

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

        retail_blurred = cv2.GaussianBlur(retail.reshape(retail.shape[0], 1), (15, 15), 0)
        grocery_blurred = cv2.GaussianBlur(grocery.reshape(grocery.shape[0], 1), (15, 15), 0)
        park_blurred = cv2.GaussianBlur(park.reshape(park.shape[0], 1), (15, 15), 0)
        trainsit_blurred = cv2.GaussianBlur(trainsit.reshape(trainsit.shape[0], 1), (15, 15), 0)
        workplace_blurred = cv2.GaussianBlur(workplace.reshape(workplace.shape[0], 1), (15, 15), 0)
        residential_blurred = cv2.GaussianBlur(residential.reshape(residential.shape[0], 1), (15, 15), 0)

        blurred_mobility[0][min_date:max_date + 1] = retail_blurred.flatten()
        blurred_mobility[1][min_date:max_date + 1] = grocery_blurred.flatten()
        blurred_mobility[2][min_date:max_date + 1] = park_blurred.flatten()
        blurred_mobility[3][min_date:max_date + 1] = trainsit_blurred.flatten()
        blurred_mobility[4][min_date:max_date + 1] = workplace_blurred.flatten()
        blurred_mobility[5][min_date:max_date + 1] = residential_blurred.flatten()

        blurred_mobility = blurred_mobility / 100

        # self.raw_mobility = mobility.T
        self.raw_mobility = blurred_mobility.T

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
            school[start:end] = - 0.65
            pass

        school = np.reshape(cv2.GaussianBlur(school.reshape(3000, 1), (7, 7), 0), newshape=(3000,)) \
                 - 0.3 * np.ones(shape=school.shape)

        conct = np.concatenate([residential.reshape(3000, 1), school.reshape(3000, 1),
                                work.reshape(3000, 1), other.reshape(3000, 1)], axis=1)

        mask_lift_date = (datetime.strptime('2020-07-01', '%Y-%m-%d') - Parameters.OUTBREAK_FIRST_DAY).days

        unmodified = conct.T
        unmodified[1][:mask_lift_date] = school[:mask_lift_date] * 0.6

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


    # def read_vaccine_by_phu(self):
    #     path = path = self.get_dependency_path() + 'vaccines_by_age_phu.csv'
    #     vaccine_df = pd.read_csv(path)
    #     vaccine_df['Percent_at_least_one_dose'] = vaccine_df['Percent_at_least_one_dose'].fillna(0)
    #     vaccine_df['Percent_fully_vaccinated'] = vaccine_df['Percent_fully_vaccinated'].fillna(0)
    #     vaccine_df['Percent_3doses'] = vaccine_df['Percent_3doses'].fillna(0)
    #     vaccine_df['Date'] = pd.to_datetime(vaccine_df['Date'])
    #     # vaccine_df['After outbreak'] = (vaccine_df['Date'] - Parameters.OUTBREAK_FIRST_DAY).dt.days
    #
    #     vaccine_df['PHU name'] = vaccine_df['PHU ID']
    #     grouped = vaccine_df.groupby('PHU ID')
    #
    #     self.find_max_date()
    #
    #     self.date_to_vaccines_by_phu = dict()
    #
    #     for g in grouped.groups:
    #         # Group by age band
    #         # Find the min time delta
    #         # Convert to numpy
    #         # transpose
    #         group = grouped.get_group(g)
    #         age_strat = group.groupby('Agegroup')
    #
    #         phu_data = np.zeros(shape=(len(Parameters.VACCINE_AGE_BANDS), self.total_days, 3))
    #
    #         for age_group in age_strat.groups:
    #
    #             age_group_data = age_strat.get_group(age_group)
    #
    #             min_date = (age_group_data['Date'].min() - Parameters.OUTBREAK_FIRST_DAY).days
    #             max_date = (age_group_data['Date'].max() - Parameters.OUTBREAK_FIRST_DAY).days
    #
    #             first_dose = age_group_data['Percent_at_least_one_dose'].to_numpy()
    #             second_dose = age_group_data['Percent_fully_vaccinated'].to_numpy()
    #             third_dose = age_group_data['Percent_3doses'].to_numpy()
    #
    #             all_dose_data = np.concatenate([first_dose.reshape(first_dose.shape[0], 1),
    #                                             second_dose.reshape(second_dose.shape[0], 1),
    #                                             third_dose.reshape(third_dose.shape[0], 1)], axis=1)
    #
    #             # print(all_dose_data.shape)
    #
    #             all_dose_data_concat = np.concatenate([np.zeros(shape=(min_date, 3)), all_dose_data,
    #                                                    np.zeros(shape=(self.total_days - max_date, 3))], axis=0)
    #
    #             if age_group in Parameters.VACCINE_AGE_BANDS:
    #                 phu_data[Parameters.VACCINE_AGE_BANDS.index(age_group)] = all_dose_data_concat
    #
    #             else:
    #                 pass
    #
    #         self.date_to_vaccines_by_phu[self.phu_id_pairing[g]] = phu_data.transpose(1, 2, 0)
    #
    #     return

    def read_vaccine_by_phu(self):

        age_bands = ['under 5', '5 - 11', '12 - 17', '18 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79',
                     '80+']
        vaccine_dose_descrpition = ['At least 1 dose coverage (%):', 'Completed primary series coverage (%):',
                                    'Completed primary series and 1 booster dose coverage (%):',
                                    'Completed primary series and 2 booster doses coverage (%):']

        vaccine_dose_extra = ['Dose 5', 'Dose 6']

        path = self.get_dependency_path() + 'All Covid-19 vaccine trends data.csv'
        vaccine_df = pd.read_csv(path)
        vaccine_df['Date'] = pd.to_datetime(vaccine_df['Date'])
        vaccine_df.replace('At least 1 dose coverage (%): ', '')

        grouped = vaccine_df.groupby('Geographic area')

        for g in grouped.groups:
            if g not in self.phu_to_district:
                continue

            population = self.population_by_phu[g]

            data = np.zeros(shape=(len(vaccine_dose_descrpition), len(age_bands), self.total_days))
            extra_dose_data = np.zeros(shape=(len(vaccine_dose_extra), len(age_bands), self.total_days))

            group = grouped.get_group(g)

            pre = (group['Date'].min() - Parameters.OUTBREAK_FIRST_DAY).days - 1
            sur = self.total_days - (group['Date'].max() - Parameters.OUTBREAK_FIRST_DAY).days

            for i in range(len(vaccine_dose_descrpition)):
                for j in range(len(age_bands)):
                    combined = vaccine_dose_descrpition[i] + ' ' + age_bands[j]
                    if combined not in vaccine_df.keys():
                        continue
                    stratified_data = np.concatenate(
                        [np.zeros(shape=pre, ), group[combined].to_numpy(), np.zeros(shape=sur, )])
                    data[i][j] = stratified_data

            for i in range(len(vaccine_dose_descrpition)):
                stratified_data = np.concatenate(
                    [np.zeros(shape=pre, ), group[vaccine_dose_extra[i]].to_numpy(), np.zeros(shape=sur, )])
                extra_dose_data[i] = stratified_data / population

            data = data.transpose(2, 0, 1)
            extra_dose = np.prod([data, extra_dose_data], axis=0)

            self.date_to_vaccines_by_phu[g] = copy.deepcopy(data)

        return

    def reshape_vaccine_by_phu(self):
        """
            reshape the 10-years age band into 5-years age band
        """

        reshaped = np.zeros((self.total_days, 4, 16))

        for phu in self.date_to_vaccines_by_phu:
            phu_data = self.date_to_vaccines_by_phu[phu]
            for date in range(self.total_days):
                for dose in [0, 1, 2]:
                    lst = phu_data[date][dose]
                    lst = Gaussian.age_dog_algo(lst)
                    reshaped[date][dose] = lst

            self.date_to_vaccines_by_phu[phu] = copy.deepcopy(reshaped)
        return

    def differentiate_by_phu(self):
        for phu in self.date_to_vaccines_by_phu:
            phu_data = self.date_to_vaccines_by_phu[phu]

            vaccine_differentiated = np.zeros((self.total_days, 3, 16))

            data = phu_data.transpose(1, 0, 2)

            print(data.shape)

            dose1 = np.clip((data[0] - data[1]), a_min=0, a_max=1)
            dose2 = np.clip((data[1] - data[2]), a_min=0, a_max=1)
            dose3 = np.clip((data[2] - data[3]), a_min=0, a_max=1)
            dose4 = data[3]

            phu_data = np.array([dose1, dose2, dose3, dose4]).transpose(1, 0, 2)

            print(phu_data.shape)


            vaccine_differentiated = np.diff(phu_data, axis=0)

            phu_data = vaccine_differentiated


            phu_data = np.clip(phu_data, a_min=0, a_max=0.2)
            self.date_to_vaccines_by_phu[phu] = phu_data

        return

    def find_max_date(self):
        case_path = self.get_dependency_path() + 'All case trends data.csv'
        vaccine_path = self.get_dependency_path() + 'All Covid-19 vaccine trends data.csv'

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
        self.date_to_vaccines_by_county = np.zeros(shape=(Parameters.NO_COUNTY, self.total_days, 4, 16), dtype=float)

        for i in range(len(self.county_data)):

            county = self.county_data[i][0]
            district = self.county_data[i][1]
            population = self.county_data[i][2]
            phu = self.district_to_phu[district]

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

            vaccinated = self.date_to_vaccines_by_phu[phu]

            print(vaccinated.shape)

            self.date_to_vaccines_by_county[i] = np.concatenate([np.zeros(shape=(1, 4, 16)),
                                                                              vaccinated], axis=0)

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

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Dependency as Dependency
import Util
from ModelData import ModelData
import Parameters


class Model:
    dependency = None
    _model_data = None

    date = 0

    day_outflow = None
    day_inflow = None

    infectiousness = 0

    model_fitting = False

    def __init__(self, forecast_days=1000, infectiousness=Parameters.INFECTIOUSNESS, prior_immunity=0, fitting=False):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency, load_from_dependency=True, prior_immunity=prior_immunity)
        self.date = self.dependency.total_days - 1
        self.infectiousness = infectiousness
        self.model_fitting = fitting
        return

    def run_one_cycle(self, display_status=False, recompute_immunity=True):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self.date += 1

        if recompute_immunity:
            self._compute_immunity(self.date)

        self._model_transition(time_step='day')

        if recompute_immunity:
            self._compute_immunity(self.date)

        self._model_transition(time_step='night')

        if display_status:
            self.print_data(self.date)
            print(self.dependency.mobility[self.date])

        return

    def print_data(self, date):
        print(date)
        today_cases = self._model_data.time_series_active_cases.transpose(1, 0, 2)
        print('Today_new_cases', np.sum(today_cases[date]))
        print('Total cases', np.sum(self._model_data.time_series_active_cases))
        print('Immunity', np.max(self._model_data.time_series_immunity.transpose(1, 0, 2)[date][0]))

    def _compute_immunity(self, date):
        """
        TODO: Find the natural immunity waning function from papers
        """

        dose1 = (self._model_data.time_series_vaccinated[0])[:date]
        dose2 = (self._model_data.time_series_vaccinated[1])[:date]
        dose3 = (self._model_data.time_series_vaccinated[2])[:date]

        dose1 = (np.ones(shape=(Parameters.NO_COUNTY, dose1.shape[0], dose1.shape[1])) * dose1).transpose(1, 0, 2)
        dose2 = (np.ones(shape=(Parameters.NO_COUNTY, dose2.shape[0], dose2.shape[1])) * dose2).transpose(1, 0, 2)
        dose3 = (np.ones(shape=(Parameters.NO_COUNTY, dose3.shape[0], dose3.shape[1])) * dose3).transpose(1, 0, 2)

        county_population = np.array(self.dependency.index_to_population)
        county_population = county_population.reshape(county_population.shape[0], 1)

        age_population = self.dependency.population_by_age_band
        age_population = age_population.reshape(age_population.shape[0], 1)

        ratio = age_population / np.sum(age_population)

        population = np.matmul(county_population, ratio.T)

        today_population = np.ones(shape=(population.shape[0], population.shape[1])) * population

        today_cases = self._model_data.time_series_active_cases.transpose(1, 0, 2)[:date]
        # today_cases = self._model_data.time_series_infected.transpose(1, 0, 2)[:date]

        today_incidence = today_cases

        self._model_data.time_series_incidence = today_incidence.transpose(1, 0, 2)

        raw_kernel_dose_1 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE1[:date])[::-1]
        raw_kernel_dose_2 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE2[:date])[::-1]
        raw_kernel_dose_3 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE3[:date])[::-1]
        raw_kernel_infection = (Parameters.INFECTION_EFFICACY_KERNEL[:date])[::-1]

        ratio = np.ones(shape=(1, Parameters.NO_COUNTY, 1))
        kernel_dose_1 = np.multiply(raw_kernel_dose_1.reshape(date, 1, 16), ratio)
        kernel_dose_2 = np.multiply(raw_kernel_dose_2.reshape(date, 1, 16), ratio)
        kernel_dose_3 = np.multiply(raw_kernel_dose_3.reshape(date, 1, 16), ratio)
        kernel_infection = np.multiply(raw_kernel_infection.reshape(date, 1, 16), ratio)

        raw_sum = np.sum(today_incidence, axis=0)

        immunity_dose1 = np.sum(np.multiply(dose1, kernel_dose_1), axis=0)
        immunity_dose2 = np.sum(np.multiply(dose2, kernel_dose_2), axis=0)
        immunity_dose3 = np.sum(np.multiply(dose3, kernel_dose_3), axis=0)

        immunity_dose1_rmv = np.sum(np.multiply(dose2, kernel_dose_1), axis=0)
        immunity_dose2_rmv = np.sum(np.multiply(dose3, kernel_dose_2), axis=0)

        infection_immunized = np.sum(np.multiply(today_incidence, kernel_infection), axis=0)

        # vaccine_immunity = immunity_dose1 + immunity_dose2 + immunity_dose3 - immunity_dose1_rmv - immunity_dose2_rmv
        vaccine_immunity = immunity_dose1 + immunity_dose2 + immunity_dose3

        infection_immunity = infection_immunized / today_population
        vaccine_immunized = vaccine_immunity * today_population

        # today_infection_immunity = np.sum(infection_immunity, axis=0)
        # today_vaccine_immunity = np.sum(vaccine_immunity, axis=0)

        padding = vaccine_immunity.T

        padding[0] = np.zeros(shape=(padding.shape[1]))

        vaccine_immunity = padding.T

        # Today infection immunity looks weird. Fix it!

        today_immunity = vaccine_immunity + (np.ones(shape=vaccine_immunity.shape) -
                                                   vaccine_immunity) * infection_immunity

        # print(vaccine_immunity[0])
        # print(infection_immunity[0])
        # print(today_immunity[0])

        # TODO: DEBUG THIS

        data = self._model_data.time_series_immunity.transpose(1, 0, 2)
        data[date] = today_immunity
        self._model_data.time_series_immunity = data.transpose(1, 0, 2)

        #
        # plt.bar(range(0, 16), vaccine_immunity[0], color='red')
        # plt.show()
        #
        # plt.bar(range(0, 16), today_immunity[0])
        # plt.show()

        return None

    def _model_transition(self, time_step='day'):
        self._susceptible_to_exposed(self.date, time_step=time_step)
        if time_step == 'night':
            self._exposed_to_cases(self.date)
            self._infected_to_hospitalized(self.date)
            self._infected_to_icu(self.date)
            self._infected_to_removed(self.date)
            self._infected_to_death(self.date)

    def _get_new_cases(self, cases, contact_type=0, contact_pattern='day', immunity=np.ones(shape=(16, ))):
        susceptibility = Parameters.SUSC_RATIO
        matrix = self._synthesize_matrix(contact_type, contact_pattern)
        rslt = np.matmul(matrix, cases) * susceptibility * np.clip(immunity, a_min=0, a_max=1) * self.infectiousness
        return rslt

    def _susceptible_to_exposed(self, date, time_step='day'):


        for c in range(Parameters.NO_COUNTY):
            immunity_level = self._model_data.time_series_immunity[c][date - 1]

            immunity = np.ones(shape=(16,)) - immunity_level

            if c == 0:
                print('Coefficient', immunity)
                print('RAW_IMMUNITY', immunity_level)
                print('Mobility', self.dependency.mobility[self.date])

            clinical_infectious = np.sum(self._model_data.time_series_clinical_cases[c][date - 2:date], axis=0)
            # sub_clinical_infectious = np.sum(self._model_data.time_series_clinical_cases[c][date - 3:date], axis=0) ?
            sub_clinical_infectious = np.sum(self._model_data.time_series_clinical_cases[c][date - 2:date], axis=0)
            exposed_infectious = np.sum(self._model_data.time_series_exposed[c][date - 3:date], axis=0)

            tot_infectiouesness = clinical_infectious + 0.5 * (sub_clinical_infectious + exposed_infectious)

            county_data = self.dependency.county_data[c]

            if county_data[2] > 10000:
                county_type = 0
            else:
                county_type = 1

            rslt = self._get_new_cases(tot_infectiouesness, contact_type=county_type, contact_pattern=time_step,
                                       immunity=np.clip(immunity, a_min=0, a_max=1))

<<<<<<< Updated upstream
            rslt = self._get_new_cases(tot_infectiouesness, contact_type=type, contact_pattern=time_step) * immunity
            # print(rslt)
=======
>>>>>>> Stashed changes
            if time_step == 'day':
                self._model_data.time_series_exposed[c][date] = rslt
                self._model_data.time_series_infected[c][date] = rslt
            else:
                self._model_data.time_series_exposed[c][date] += rslt
                self._model_data.time_series_infected[c][date] += rslt

    """
        Exposed to cases
    """

    def _exposed_to_cases(self, date):

        raw_kernel = Parameters.EXP2ACT_CONVOLUTION_KERNEL

        for c in range(Parameters.NO_COUNTY):
            kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), np.ones(shape=(1, 16)))[::-1]
            kernel_size = kernel.shape[0]
            county_data = self._model_data.time_series_exposed[c]
            data = county_data[date - kernel_size:date]

            rslt = np.sum(np.multiply(data, kernel), axis=0)

            self._model_data.time_series_active_cases[c][date] = rslt
            self._model_data.time_series_clinical_cases[c][date] = np.multiply(rslt,
                                                                               Parameters.CLINICAL_RATIO)
            self._model_data.time_series_sub_clinical_cases[c][date] = np.multiply(rslt,
                                                                                   Parameters.SUBCLINICAL_RATIO)

    def _infected_to_hospitalized(self, date):
        ratio = Parameters.ONT_HOSP_RATIO.reshape(16, 1)
        raw_kernel = Parameters.INF2HOS_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_hospitalized[c][date] = rslt

    def _infected_to_death(self, date):
        ratio = Parameters.ONT_CFR.reshape(16, 1)
        raw_kernel = Parameters.CLI2DEA_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_deaths[c][date] = rslt

        return

    def _infected_to_icu(self, date):
        ratio = Parameters.ONT_ICU_RATIO.reshape(16, 1)
        raw_kernel = Parameters.HOS2ICU_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]

        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_ICU[c][date] = rslt

    def _infected_to_removed(self, date):
        self._subclinical_to_removed(date)
        self._clinical_to_removed(date)
        self._hospitalized_to_removed(date)
        self._icu_to_removed(date)

    def _subclinical_to_removed(self, date):
        ratio = np.ones(shape=(16, 1))

        raw_kernel = Parameters.SUB2REC_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_sub_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_recovered[c][date] = rslt

    def _clinical_to_removed(self, date):
        ratio = np.ones(shape=(16, 1))

        raw_kernel = Parameters.CLI2REC_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_recovered[c][date] += rslt

    def _hospitalized_to_removed(self, date):
        ratio = np.ones(shape=(16, 1))

        raw_kernel = Parameters.HOS2RMV_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_recovered[c][date] += rslt

    def _icu_to_removed(self, date):
        ratio = np.ones(shape=(16, 1))

        raw_kernel = Parameters.ICU2RMV_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)[::-1]
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_recovered[c][date] += rslt

    def _synthesize_matrix(self, contact_type=0, contact_pattern='day'):
        matrices = self.dependency.matrix_by_class
        preset = Parameters.MATRIX_PRESETS[contact_pattern]
        matrix = np.zeros(shape=(16, 16))
        for j in range(4):
<<<<<<< Updated upstream
            matrix = np.add(matrix, preset[j] * matrices[contact_type][j]) * self.dependency.mobility[self.date][j]
        return matrix * self.infectiousness
=======
            # print(Parameters.MATRIX_CONTACT_TYPE[j], self.dependency.mobility[self.date][j])
            matrix = np.add(matrix, preset[j] * matrices[contact_type][j] * self.dependency.mobility[self.date][j])
        # return matrix * self.infectiousness
        matrix = matrix * Parameters.SEASONALITY[self.date]
        return matrix.T
>>>>>>> Stashed changes

    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()

    def save_to_file(self):
        """
        self.save_data(data=self._model_data.time_series_clinical_cases, tag='clinical_cases')
        self.save_data(data=self._model_data.time_series_sub_clinical_cases, tag='subclinical_cases')
        self.save_data(data=self._model_data.time_series_active_cases, tag='total_cases')
        self.save_data(data=self._model_data.time_series_hospitalized, tag='hospitalized')
        self.save_data(data=self._model_data.time_series_ICU, tag='ICU')
        self.save_data(data=self._model_data.time_series_deaths, tag='deaths')
        self.save_data(data=self._model_data.time_series_recovered, tag='recovered')
        self.save_data(data=self._model_data.time_series_immunity, tag='immunity')
        """

        phu = self.phu_ordering()
        """
        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_clinical_cases), phus=phu,
                              tag='clinical_cases', moving_avg=True)
        """
        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_immunity), phus=phu,
                              tag='immunity', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_active_cases), phus=phu,
                              tag='total_cases', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_clinical_cases), phus=phu,
                              tag='clinical_cases', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_hospitalized), phus=phu,
                              tag='hospitalized', moving_avg=True)
        return

    def save_data(self, data=None, tag='', moving_avg=False):
        for c in range(Parameters.NO_COUNTY):
            county_data = data[c]
            if moving_avg:
                county_data = Util.moving_average(county_data)
            df = pd.DataFrame(county_data, columns=Parameters.AGE_BANDS)
            dir_path = os.getcwd()[:-5] + \
                       'model_output/' + \
                       self.dependency.code_to_name[self.dependency.county_codes[c]]
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = dir_path + '/' + tag + '.csv'
            df.to_csv(path_or_buf=path)
        return

    def phu_ordering(self):
        phus = []
        phu_count = 0
        for c in range(Parameters.NO_COUNTY):
            county = self.dependency.county_data[c]
            district = county[1]
            phu = self.dependency.district_to_phu[district]
            if phu not in phus:
                phus.append(phu)
                phu_count += 1
            else:
                pass
        return phus

    def save_data_by_phu(self, data=None, phus=[], tag='', moving_avg=False):
        print(phus)
        for p in range(len(phus)):
            phu_data = data[p]
            if moving_avg:
                phu_data = Util.moving_average(phu_data)
            df = pd.DataFrame(phu_data, columns=Parameters.AGE_BANDS)
            phu = phus[p]
            if phu[-1] == ' ':
                phu = phu[:-1]
            dir_path = os.getcwd()[:-5] + \
                       'model_output/' + \
                       phu
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = dir_path + '/' + tag + '.csv'
            df.to_csv(path_or_buf=path)
            print(phu, np.sum(phu_data))
        return

    def convert_data_to_phu(self, data=None):
        phus = []
        phu_count = 0
        phu_data = np.zeros(shape=(Parameters.NO_PHU, data.shape[1], data.shape[2]))
        for c in range(Parameters.NO_COUNTY):
            county = self.dependency.county_data[c]
            district = county[1]

            phu = self.dependency.district_to_phu[district]

            if phu not in phus:
                phus.append(phu)
                phu_count += 1
            else:
                pass
            phu_index = phus.index(phu)
            phu_data[phu_index] += data[c]
        return phu_data


if __name__ == '__main__':

    forecast_days = 300

    m = Model(forecast_days=forecast_days)

    m.date -= forecast_days

    for i in range(forecast_days):
        m.run_one_cycle(display_status=True)
        pass

    plt.plot(m._model_data.time_series_vaccinated[0])
    # plt.plot(m._model_data.time_series_immunity[0])
    plt.show()
    print('Model run done')
    # m.save_to_file()
    print('Done')

    pass


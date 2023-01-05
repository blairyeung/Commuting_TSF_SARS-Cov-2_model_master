import datetime
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

    enable_commute = True

    def __init__(self, forecast_days=1000, infectiousness=Parameters.INFECTIOUSNESS, prior_immunity=0, fitting=False,
                 enable_commute=True):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency, load_from_dependency=True,
                                     prior_immunity=prior_immunity)
        self.date = self.dependency.total_days - 1
        self.infectiousness = infectiousness
        self.model_fitting = fitting
        self.enable_commute = enable_commute
        return

    def run_one_cycle(self, display_status=False, recompute_immunity=True):
        self.date += 1

        # Compute immunity
        if recompute_immunity:
            self._compute_immunity(self.date)

        # Compute the day time transmission cycle
        self._model_transition(time_step='day')

        # recompute immunity for night time transmission
        if recompute_immunity:
            self._compute_immunity(self.date)

        # Compute the night time transmission cycle
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

        print(self._model_data.time_series_vaccinated.shape)

        # Get the provincial vaccination data
        dose1_raw = (self._model_data.time_series_vaccinated[0])[:, :date]
        dose2_raw = (self._model_data.time_series_vaccinated[1])[:, :date]
        dose3_raw = (self._model_data.time_series_vaccinated[2])[:, :date]

        # Broadcast the data to the whole province

        dose1 = dose1_raw.transpose(1, 0, 2)
        dose2 = dose2_raw.transpose(1, 0, 2)
        dose3 = dose3_raw.transpose(1, 0, 2)

        # dose1 = (np.ones(shape=(Parameters.NO_COUNTY, dose1_raw.shape[0],
        #                         dose1_raw.shape[1])) * dose1_raw).transpose(1, 0, 2)
        # dose2 = (np.ones(shape=(Parameters.NO_COUNTY, dose2_raw.shape[0],
        #                         dose2_raw.shape[1])) * dose2_raw).transpose(1, 0, 2)
        # dose3 = (np.ones(shape=(Parameters.NO_COUNTY, dose3_raw.shape[0],
        #                         dose3_raw.shape[1])) * dose3_raw).transpose(1, 0, 2)

        # Compute incidence rate

        county_population = np.array(self.dependency.index_to_population)
        county_population = county_population.reshape(county_population.shape[0], 1)

        age_population = self.dependency.population_by_age_band
        age_population = age_population.reshape(age_population.shape[0], 1)

        ratio = age_population / np.sum(age_population)

        population = np.matmul(county_population, ratio.T)

        today_population = np.ones(shape=(population.shape[0], population.shape[1])) * population

        today_cases = self._model_data.time_series_active_cases.transpose(1, 0, 2)[:date]

        today_incidence = today_cases

        self._model_data.time_series_incidence = today_incidence.transpose(1, 0, 2)

        # Get the convolutional kernels

        raw_kernel_dose_1 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE1[:date])[::-1]
        raw_kernel_dose_2 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE2[:date])[::-1]
        raw_kernel_dose_3 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE3[:date])[::-1]
        raw_kernel_infection = (Parameters.INFECTION_EFFICACY_KERNEL[:date])[::-1]

        raw_kernel_dose_rmv_1 = (Parameters.ONE_DOSE_EFFICACY_RMV_TRANSMISSION[:date])[::-1]
        raw_kernel_dose_rmv_2 = (Parameters.TWO_DOSE_EFFICACY_RMV_TRANSMISSION[:date])[::-1]

        # Reshape the convolutional kernels

        ratio = np.ones(shape=(1, Parameters.NO_COUNTY, 1))
        kernel_dose_1 = np.multiply(raw_kernel_dose_1.reshape(date, 1, 16), ratio)
        kernel_dose_2 = np.multiply(raw_kernel_dose_2.reshape(date, 1, 16), ratio)
        kernel_dose_3 = np.multiply(raw_kernel_dose_3.reshape(date, 1, 16), ratio)
        kernel_infection = np.multiply(raw_kernel_infection.reshape(date, 1, 16), ratio)

        kernel_dose_rmv_1 = np.multiply(raw_kernel_dose_rmv_1.reshape(date, 1, 16), ratio)
        kernel_dose_rmv_2 = np.multiply(raw_kernel_dose_rmv_2.reshape(date, 1, 16), ratio)

        # Convolve at one specific data and get the immunity

        immunity_dose1 = np.sum(np.multiply(dose1, kernel_dose_1), axis=0)
        immunity_dose2 = np.sum(np.multiply(dose2, kernel_dose_2), axis=0)
        immunity_dose3 = np.sum(np.multiply(dose3, kernel_dose_3), axis=0)

        immunity_dose1_rmv = np.sum(np.multiply(dose2, kernel_dose_rmv_1), axis=0)
        immunity_dose2_rmv = np.sum(np.multiply(dose3, kernel_dose_rmv_2), axis=0)

        infection_immunized = np.sum(np.multiply(today_incidence, kernel_infection), axis=0)
        vaccine_immunity = immunity_dose1 + immunity_dose2 + immunity_dose3 - immunity_dose1_rmv - immunity_dose2_rmv

        infection_immunity = infection_immunized / today_population
        vaccine_immunized = vaccine_immunity * today_population

        padding = vaccine_immunity.T

        padding[0] = np.zeros(shape=(padding.shape[1]))

        vaccine_immunity = padding.T

        today_immunity = vaccine_immunity + (np.ones(shape=vaccine_immunity.shape) -
                                             vaccine_immunity) * infection_immunity

        data = self._model_data.time_series_immunity.transpose(1, 0, 2)
        infection_data = self._model_data.time_series_infection_immunity.transpose(1, 0, 2)
        vaccine_data = self._model_data.time_series_vaccine_immunity.transpose(1, 0, 2)

        data[date] = today_immunity
        infection_data[date] = infection_immunity
        vaccine_data[date] = vaccine_immunity

        self._model_data.time_series_immunity = data.transpose(1, 0, 2)
        self._model_data.time_series_infection_immunity = infection_data.transpose(1, 0, 2)
        self._model_data.time_series_vaccine_immunity = vaccine_data.transpose(1, 0, 2)

        return None

    def _model_transition(self, time_step='day'):
        self._susceptible_to_exposed(self.date, time_step=time_step)
        if time_step == 'night':
            self._exposed_to_cases(self.date)
            self._infected_to_hospitalized(self.date)
            self._infected_to_icu(self.date)
            self._infected_to_removed(self.date)
            self._infected_to_death(self.date)

    def _commuting_mix_forward(self, inf_by_county):

        if not self.enable_commute:
            return np.identity(inf_by_county.shape[0])

        matrix = np.clip(self.dependency.commute_matrix, a_min=1, a_max=Parameters.ONT_POPULATOIN)

        inflow_sum = np.sum(matrix, axis=0)
        outflow_sum = np.sum(matrix, axis=1)

        col_normed = matrix / np.sum(matrix, axis=0)
        row_normed = (matrix.T / np.sum(matrix, axis=1)).T

        rslt = np.matmul(col_normed, inf_by_county)

        return rslt

    def _commuting_mix_backward(self, inf_by_county):

        if not self.enable_commute:
            return np.identity(inf_by_county.shape[0])

        matrix = np.clip(self.dependency.commute_matrix, a_min=1, a_max=Parameters.ONT_POPULATOIN)

        inflow_sum = np.sum(matrix, axis=0)
        outflow_sum = np.sum(matrix, axis=1)

        col_normed = matrix / np.sum(matrix, axis=0)
        row_normed = (matrix.T / np.sum(matrix, axis=1)).T

        rslt = np.matmul(row_normed, inf_by_county)

        return rslt

    def _get_new_cases(self, cases, contact_type=0, contact_pattern='day', work=False, immunity=np.ones(shape=(16,))):
        susceptibility = Parameters.SUSC_RATIO
        matrix = self._synthesize_matrix(contact_type, contact_pattern, work=work)

        if work:
            matrix = self._synthesize_working_matrix(contact_type)
        else:
            matrix = self._synthesize_matrix(contact_type, contact_pattern, work=work)
        rslt = np.matmul(matrix, cases) * susceptibility * np.clip(immunity, a_min=0, a_max=1) * self.infectiousness
        return rslt

    def _susceptible_to_exposed(self, date, time_step='day'):

        infectious_by_county = np.zeros(shape=(Parameters.NO_COUNTY, 16))
        immunity_by_county = np.zeros(shape=(Parameters.NO_COUNTY, 16))

        for c in range(Parameters.NO_COUNTY):

            # For non-working population, do local transmission here

            immunity_level = self._model_data.time_series_immunity[c][date - 1]

            immunity = np.ones(shape=(16,)) - immunity_level

            clinical_infectious = np.sum(self._model_data.time_series_clinical_cases[c][date - 2:date], axis=0)
            sub_clinical_infectious = np.sum(self._model_data.time_series_clinical_cases[c][date - 2:date], axis=0)

            exposed_infectious = np.sum(self._model_data.time_series_exposed[c][date - 3:date], axis=0)

            tot_infectiouesness = clinical_infectious + 0.5 * (sub_clinical_infectious + exposed_infectious)

            infectious_by_county[c] = tot_infectiouesness
            immunity_by_county[c] = immunity

            county_data = self.dependency.county_data[c]

            if county_data[2] > 10000:
                county_type = 0
            else:
                county_type = 1

            rslt = self._get_new_cases(tot_infectiouesness, contact_type=county_type, contact_pattern=time_step,
                                       work=False, immunity=np.clip(immunity, a_min=0, a_max=1))

            if time_step == 'day':
                self._model_data.time_series_exposed[c][date] = rslt
                self._model_data.time_series_infected[c][date] = rslt
            else:
                self._model_data.time_series_exposed[c][date] += rslt
                self._model_data.time_series_infected[c][date] += rslt

        if time_step == 'day':
            commuting_mix = self._commuting_mix_forward(infectious_by_county)
            infected_by_county = np.zeros(shape=infectious_by_county.shape)
            for c in range(Parameters.NO_COUNTY):
                rslt = self._get_new_cases(commuting_mix[c], contact_type=county_type, contact_pattern=time_step,
                                           work=True, immunity=np.clip(immunity, a_min=0, a_max=1))
                infected_by_county[c] = rslt

            reverse_mix = self._commuting_mix_backward(infected_by_county)

            for c in range(Parameters.NO_COUNTY):
                rslt = reverse_mix[c]
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

    def _synthesize_matrix(self, contact_type=0, contact_pattern='day', work=False):
        matrices = self.dependency.matrix_by_class
        preset = Parameters.MATRIX_PRESETS[contact_pattern]
        matrix = np.zeros(shape=(16, 16))
        if work:
            for j in range(4):
                matrix = np.add(matrix, preset[j] * matrices[contact_type][j] * self.dependency.mobility[self.date][j])

        else:
            for j in [0, 2, 3]:
                matrix = np.add(matrix, preset[j] * matrices[contact_type][j] * self.dependency.mobility[self.date][j])

        matrix = matrix * Parameters.SEASONALITY[self.date]
        return matrix.T

    def _synthesize_working_matrix(self, contact_type=0):
        matrices = self.dependency.matrix_by_class
        preset = Parameters.MATRIX_PRESETS['day']
        matrix = preset[1] * matrices[contact_type][1] * self.dependency.mobility[self.date][1]
        matrix = matrix * Parameters.SEASONALITY[self.date]
        return matrix.T

    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()

    def save_to_file(self):

        self.save_provincial_data(data=self._model_data.time_series_immunity,
                                  tag='immunity', moving_avg=True)

        self.save_provincial_data(data=self._model_data.time_series_active_cases,
                                  tag='total_cases', moving_avg=True)

        self.save_provincial_data(data=self._model_data.time_series_clinical_cases,
                                  tag='clinical_cases', moving_avg=True)

        self.save_provincial_data(data=self._model_data.time_series_hospitalized,
                                  tag='hospitalized', moving_avg=True)

        phu = self.phu_ordering()

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_immunity), phus=phu,
                              tag='immunity', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_active_cases), phus=phu,
                              tag='total_cases', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_clinical_cases), phus=phu,
                              tag='clinical_cases', moving_avg=True)

        self.save_data_by_phu(self.convert_data_to_phu(data=self._model_data.time_series_hospitalized), phus=phu,
                              tag='hospitalized', moving_avg=True)
        return

    def save_provincial_data(self, data=None, tag='', moving_avg=False):
        ont_data = np.sum(data, axis=0)
        if moving_avg:
            ont_data = Util.moving_average(ont_data)

        df = pd.DataFrame(ont_data, columns=Parameters.AGE_BANDS)

        dir_path = os.getcwd()[:-5] + \
                   'model_output/' + \
                   'Ontario'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df['date'] = [Parameters.OUTBREAK_FIRST_DAY + datetime.timedelta(days=j) for j in
                      range(ont_data.shape[0])]
        path = dir_path + '/' + tag + '.csv'
        df.to_csv(path_or_buf=path)

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
    plt.show()
    print('Model run done')
    m.save_to_file()
    print('Done')

    pass

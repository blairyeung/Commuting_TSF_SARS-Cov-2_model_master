import math
import numpy as np
import Dependency as Dependency
from ModelData import ModelData
import Parameters


class Model:
    dependency = None
    _model_data = None

    date = 0

    def __init__(self, forecast_days=1000):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency, True)
        self.date = self.dependency.total_days - 1
        # TODO: Initialize
        return

    def run_one_cycle(self):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self.date += 1
        print(self.date)
        self._model_data.compute_immunity(self.date)
        self._model_transition()
        return

    def _get_new_cases(self, cases, contact_type=0, contact_pattern='day'):
        susceptibility = Parameters.SUSC_BY_AGE
        matrix = self._synthesize_matrix(contact_type, contact_pattern)
        self._model_data.compute_immunity(self.date)
        # print(self._model_data.time_series_immunity.shape)
        immunity = self._model_data.time_series_immunity.transpose(1, 0, 2)[self.date]
        # print(immunity.shape)
        rslt = np.matmul(matrix, cases) * susceptibility * immunity
        # print(rslt.shape)
        return rslt

    def _model_transition(self):
        self._susceptible_to_exposed(self.date)
        self._exposed_to_cases(self.date)
        self._infected_to_hospitalized(self.date)
        self._hospitalized_to_icu(self.date)
        self._infected_to_removed(self.date)

    def _susceptible_to_exposed(self, date):
        for c in range(Parameters.NO_COUNTY):
            exposed_cases = self._model_data.time_series_exposed[c][date]
            active_transmissible = self._model_data.time_series_clinical_cases[c][date]
            # TODO: Replace this with self._model_data.time_series_immunity[c][date] after update
            # immunity_level = np.zeros(shape=(16, ))
            immunity_level = self._model_data.time_series_immunity[c][date]
            immunity_coeff = np.ones(shape=(16, )) - immunity_level
            self._get_new_cases(np.multiply(exposed_cases + active_transmissible, immunity_coeff))


    """
        Exposed to cases
    """
    def _exposed_to_cases(self, date):
        ratio = Parameters.SUSC_BY_AGE.reshape(16, 1)

        raw_kernel = Parameters.EXP2ACT_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_exposed[c]
            data = county_data[date - kernel_size:date]
            data = data[::-1]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_active_cases[c][date] = rslt
            self._model_data.time_series_clinical_cases[c][date] = np.multiply(rslt,
                                                                               Parameters.CLINICAL_BY_AGE)
            self._model_data.time_series_clinical_cases[c][date] = np.multiply(rslt,
                                                                               Parameters.SUBCLINICAL_BY_AGE)

    def _infected_to_hospitalized(self, date):

        ratio = Parameters.ONT_HOSP_RATIO.reshape(16, 1)
        raw_kernel = Parameters.INF2HOS_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)

        # kernel = np.matmul(ratio, np.transpose(Parameters.INF2HOS_CONVOLUTION_KERNEL))

        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_clinical_cases[c]
            data = county_data[date - kernel_size:date]
            data = data[::-1]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_hospitalized[c][date] = rslt


    def _hospitalized_to_icu(self, date):
        ratio = (Parameters.ONT_ICU_RATIO / Parameters.ONT_HOSP_RATIO).reshape(16, 1)
        raw_kernel = Parameters.HOS2ICU_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)

        # kernel = np.multiply(Parameters.HOS2ICU_CONVOLUTION_KERNEL, np.transpose(ratio))
        kernel_size = kernel.shape[0]

        for c in range(Parameters.NO_COUNTY):
            county_data = self._model_data.time_series_hospitalized[c]
            data = county_data[date - kernel_size:date]
            data = data[::-1]
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data.time_series_ICU[c][date] = rslt

    def _infected_to_removed(self, date):
        self._subclinical_to_removed(date)
        self._clinical_to_removed(date)
        self._hospitalized_to_removed(date)
        self._icu_to_removed(date)

    def _subclinical_to_removed(self, date):
        kernel = Parameters.SUB2REC_CONVOLUTION_KERNEL
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._model_data.time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._model_data.time_series_active_cases[date] = rslt
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _clinical_to_removed(self, date):
        kernel = Parameters
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._model_data.time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._model_data.time_series_active_cases[date] = rslt
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _hospitalized_to_removed(self, date):
        kernel = Parameters
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._model_data.time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._model_data.time_series_active_cases[date] = rslt
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                              Parameters.clinical_rate)
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _icu_to_removed(self, date):
        """
        :param date:
        :return:
        """
        kernel = Parameters.ICU2DEA_CONVOLUTION_KERNEL
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._model_data.time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=0)

        self._model_data.time_series_active_cases[date] = rslt
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._model_data.time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _synthesize_matrix(self, contact_type=0, contact_pattern='day'):
        matrices = self.dependency.matrix_by_class
        preset = Parameters.MATRIX_PRESETS[contact_pattern]
        matrix = np.zeros(shape=(16, 16))
        for j in range(4):
            matrix = np.add(matrix, preset[j] * matrices[contact_type][j])
        return matrix

    def _initialize_dependencies(self):
         self.dependency = Dependency.Dependency()


if __name__ == '__main__':
    m = Model(forecast_days=100)

    for i in range(100):
        m.  run_one_cycle()

    pass

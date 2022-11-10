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
        self.date = self.dependency.total_days
        # TODO: Initialize
        return

    def run_one_cycle(self):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self.date += 1
        # self._model_data
        self._model_transition()
        return

    def _get_new_cases(self, cases, contact_type=0, contact_pattern='day'):
        susceptibility = Parameters.sup_by_age
        matrix = self._synthesize_matrix(contact_type, contact_pattern)
        return np.matmul(matrix, cases)

    def _model_transition(self):
        self._exposed_to_cases(self.date)
        # self._infected_to_hospitalization(self.date)
        # self._hospitalized_to_icu(self.date)
        # self._infected_to_removed(self.date)

    def _exposed_to_cases(self, date):
        ratio = np.ones(shape=(16, 1), dtype=float)

        raw_kernel = Parameters.EXP2ACT_CONVOLUTION_KERNEL
        kernel = np.matmul(raw_kernel.reshape((raw_kernel.shape[0], 1)), ratio.T)
        kernel_size = kernel.shape[0]

        for i in range(Parameters.NO_COUNTY):
            county_data = self._model_data._time_series_active_cases[i]
            data = county_data[date - kernel_size:date]
            data = data[::-1]
            # mat = np.multiply(data, kernel)
            rslt = np.sum(np.multiply(data, kernel), axis=0)
            self._model_data._time_series_active_cases[i][date] = rslt
            self._model_data._time_series_clinical_cases[i][date] = np.multiply(rslt,
                                                                 Parameters.CLINICAL_BY_AGE)
            self._model_data._time_series_clinical_cases[i][date] = np.multiply(rslt,
                                                                 Parameters.SUBCLINICAL_BY_AGE)


    def _infected_to_hospitalized(self, date):
        ratio = np.zeros(shape=(16, 1), dtype=float)
        kernel = np.matmul(ratio, np.transpose(Parameters.INF2HOS_CONVOLUTION_KERNEL))
        # kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _hospitalized_to_icu(self, date):
        # TODO: This parameter is from bayesian inference
        ratio = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        kernel = np.multiply(Parameters.HOS2ICU_CONVOLUTION_KERNEL, np.transpose(ratio))
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _infected_to_removed(self, date):
        self._subclinical_to_removed(date)
        self._clinical_to_removed(date)
        self._hospitalized_to_removed(date)
        self._icu_to_removed(date)

    def _subclinical_to_removed(self, date):
        kernel = Parameters
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _clinical_to_removed(self, date):
        kernel = Parameters
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _hospitalized_to_removed(self, date):
        kernel = Parameters
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=1)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _icu_to_removed(self, date):
        """
        :param date:
        :return:
        """
        kernel = Parameters.ICU2DEA_CONVOLUTION_KERNEL
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=0)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
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
    m._synthesize_matrix()
    """
    for i in range(99):
        m.run_one_cycle()
    """
    pass

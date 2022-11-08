import math
import numpy as np
import Dependency as Dependency
import ModelData
import Parameters


class Model:
    dependency = None
    _model_data = None

    date = 0

    def __init__(self, forecast_days=1000):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency)
        # TODO: Initialize
        return

    def run_one_cycle(self):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self.date += 1
        self._model_data
        self._model_transition()
        return

    def _get_new_cases(self, cases, contact_type=(0, 0)):
        return np.dot(self._model_data.dependency.matrix_by_class[contact_type[0]][contact_type[1]], cases)

    def _model_transition(self):
        self._exposed_to_cases(self.date)
        self._infected_to_hospitalization(self.date)
        self._hospitalized_to_icu(self.date)
        self._infected_to_removed(self.date)

    def _exposed_to_cases(self, date):
        ratio = np.ones(shape=(16, ), dtype=float)
        kernel = np.multiply(Parameters.EXP2ACT_CONVOLUTION_KERNEL, ratio)
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=0)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _infected_to_hospitalized(self, date):
        ratio = np.zeros(shape=(16, ), dtype=float)
        kernel = np.multiply(Parameters.INF2HOS_CONVOLUTION_KERNEL, ratio)
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=0)

        self._time_series_active_cases[date] = rslt
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(rslt,
                                                             Parameters.subclinical_rate)

    def _hospitalized_to_icu(self, date):
        ratio = np.zeros(shape=(16, ), dtype=float)
        kernel = np.multiply(Parameters.HOS2ICU_CONVOLUTION_KERNEL, ratio)
        kernel = kernel.reshape((kernel.shape[0], 1))
        kernel_size = kernel.shape[0]
        rslt = np.sum(np.multiply(self._time_series_active_cases[date - kernel_size:date],
                                  kernel), axis=0)

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
                                  kernel), axis=0)

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
                                  kernel), axis=0)

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
                                  kernel), axis=0)

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

    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()


if __name__ == '__main__':
    m = Model(forecast_days=100)

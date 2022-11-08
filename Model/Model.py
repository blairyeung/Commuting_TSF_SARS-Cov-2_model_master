import math
import numpy as np
import Dependency as Dependency
import ModelData
import Parameters

class Model:

    dependency = None
    _model_data = None

    def __init__(self, forecast_days=1000):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency)
        # TODO: Initialize
        return

    def run_one_cycle(self):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self._model_data
        # TODO: Run one cycle
        return

    def _get_new_cases(self, cases, contact_type=(0, 0)):
        return np.dot(self._model_data.dependency.matrix_by_class[contact_type[0]][contact_type[1]], cases)

    def exposed_to_cases(self, date):
        kernel = Parameters.EXP2ACT_CONVOLUTION_KERNEL
        convolution_rslt = np.convolve(self._time_series_active_cases[date-10:date], kernel)
        new_cases = np.sum(convolution_rslt)
        self._time_series_active_cases[date] = new_cases
        self._time_series_clinical_cases[date] = np.multiply(self._time_series_active_cases[date],
                                                             Parameters.clinical_rate)
        self._time_series_clinical_cases[date] = np.multiply(self._time_series_active_cases[date],
                                                             Parameters.subclinical_rate)

    def infected_to_hospitalization(self, date):
        pass

    def infected_to_removed(self, date):
        pass

    def hospitalization_to_removed(self, date):
        pass

    def icu_to_removed(self, date):
        pass

    def icu_to_removed(self, date):
        """
        :param date:
        :return:
        """
        self._time_series_deaths[date] = np.convolve()
        self._time_series_recovered[date] = np.convolve()


    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()


if __name__ == '__main__':
    m = Model(forecast_days=100)

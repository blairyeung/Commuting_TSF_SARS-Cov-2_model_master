import math
import numpy as np
import Dependency as Dependency
import Parameters


class ModelData:
    """
        Attributes
    """

    _time_series_len = 0

    _time_series_infected = None
    _time_series_exposed= None
    _time_series_active_cases = None
    _time_series_clinical_cases = None
    _time_series_sub_clinical_cases = None
    _time_series_hospitalized = None
    _time_series_ICU = None
    _time_series_recovered = None
    _time_series_deaths = None

    _time_series_immunized = None

    def __init__(self, forecast_days=1000, dependency=None):
        """
            Initialize the tensors
            shape=(x, y, z)
            Here, it should be (x * 528 * 16)
             x = number of days to forecast + dates elapsed since Covid outbreak
            y = number of counties
            z = 16 age bands
        """
        self.dependency = dependency
        self._time_series_len = self.dependency.total_days + forecast_days
        x = self._time_series_len
        y = len(self.dependency.county_data)
        z = 16

        # print(self._time_series_len)
        self._time_series_infected = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_exposed = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_active_cases = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_clinical_cases = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_sub_clinical_cases = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_hospitalized = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_ICU = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_recovered = np.zeros(shape=(x, y, z), dtype=int)
        self._time_series_deaths = np.zeros(shape=(x, y, z), dtype=int)

    def exposed_to_cases(self, date):
        kernel = Parameters.E2I_CONVOLUTION_KERNELh
        self._time_series_active_cases[date] = self._time_series_exposed[date - 3]
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




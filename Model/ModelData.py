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


if __name__ == '__main__':
    pass
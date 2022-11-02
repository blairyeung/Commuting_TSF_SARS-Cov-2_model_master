import math
import numpy as np
import Dependency as Dependency


class ModelData:
    """
        Attributes
    """

    dependency = None

    _time_series_len = 0

    _time_series_infected = None
    _time_series_exposed= None
    _time_series_active_cases = None
    _time_series_clinical_cases = None
    _time_series_sub_clinical_cases = None
    _time_series_recovered = None
    _time_series_deaths = None

    _time_series_immunized = None

    def __init__(self, forecast_days=1000):
        """
            Initialize the tensors
            shape=(x, y, z)
            x = number of counties
            y = number of days to forecast + dates elapsed since Covid outbreak
            z = 16 age bands
        """
        self._initialize_dependencies()
        self._time_series_len = self.dependency.total_days + forecast_days

        x = self._time_series_len
        y = 16

        print(self._time_series_len)
        self._time_series_infected = np.zeros(shape=(x, y), dtype=int)
        self._time_series_exposed = np.zeros(shape=(x, y), dtype=int)
        self._time_series_active_cases = np.zeros(shape=(x, y), dtype=int)
        self._time_series_clinical_cases = np.zeros(shape=(x, y), dtype=int)
        self._time_series_sub_clinical_cases = np.zeros(shape=(x, y), dtype=int)
        self._time_series_recovered = np.zeros(shape=(x, y), dtype=int)
        self._time_series_deaths = np.zeros(shape=(x, y), dtype=int)

    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()
        # self.dependency.read_files()
        print(self.dependency.county_data)


class Model:

    _model_data = None

    def __init__(self, forecast_days=1000):
        self._model_data = ModelData(forecast_days)
        # TODO: Initialize
        return

    def run_one_cycle(self):
        # TODO: Run one cycle
        return


if __name__ == '__main__':
    m = Model(forecast_days=100)
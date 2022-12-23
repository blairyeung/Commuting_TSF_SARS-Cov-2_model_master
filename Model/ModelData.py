import copy
import math
import numpy as np
import numexpr as ne
import Dependency as Dependency
import Parameters


class ModelData:
    """
        Attributes
    """

    forecast_days = 0

    time_series_len = 0

    time_series_infected = None
    time_series_incidence = None
    time_series_exposed = None
    time_series_active_cases = None
    time_series_clinical_cases = None
    time_series_sub_clinical_cases = None
    time_series_hospitalized = None
    time_series_ICU = None
    time_series_recovered = None
    time_series_deaths = None
    time_series_population = None

    time_series_immunized = None
    time_series_vaccinated = None
    time_series_vaccine_immunity = None

    time_series_immunity = None

    dependency = None

    def __init__(self, forecast_days=1000, dependency=None, load_from_dependency=False):
        """
            Initialize the tensors
            shape=(x, y, z)
            Here, it should be (x * 528 * 16)
            x = number of counties
            y = number of days to forecast + dates elapsed since Covid outbreak
            z = 16 age bands
        """
        # TODO: Load data from
        self.forecast_days = forecast_days
        self.dependency = dependency
        self.time_series_len = self.dependency.total_days + forecast_days

        if load_from_dependency:
            self._load_from_dependencies()
        else:
            x = len(self.dependency.county_data)
            y = self._time_series_len
            z = 16

            self.time_series_population = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_infected = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_exposed = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_active_cases = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_clinical_cases = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_sub_clinical_cases = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_hospitalized = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_ICU = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_recovered = np.zeros(shape=(x, y, z), dtype=int)
            self.time_series_deaths = np.zeros(shape=(x, y, z), dtype=int)

            """
                Note that!!! We asssumed an consistent vaccination level acrross ALL counties in ON!
            """
            self.time_series_vaccinated = np.zeros(shape=(x, y, z), dtype=int)

            self.time_series_immunity = np.zeros(shape=(x, y, z), dtype=float)

    def _load_from_dependencies(self):
        x = len(self.dependency.county_data)
        y = self.forecast_days
        z = 16

        self.time_series_clinical_cases = np.concatenate([self.dependency.date_to_cases_by_county,
                                                          np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_active_cases = np.multiply(self.time_series_clinical_cases,
                                                    Parameters.BAYES_CLINICAL_BY_AGE.T)

        self.time_series_exposed = copy.deepcopy(self.time_series_active_cases)

        self.time_series_infected = copy.deepcopy(self.time_series_active_cases)

        self.time_series_sub_clinical_cases = self.time_series_active_cases - self.time_series_clinical_cases

        # TODO: update this

        self.time_series_recovered = np.apply_along_axis(lambda m:
                                                         np.convolve(m,
                                                                     Parameters.CLI2REC_CONVOLUTION_KERNEL,
                                                                     mode='same'), axis=1,
                                                         arr=self.time_series_clinical_cases)

        self.time_series_deaths = np.concatenate([self.dependency.date_to_deaths_by_county,
                                                  np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_hospitalized = np.concatenate([self.dependency.date_to_hospitalizations_by_county,
                                                        np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_ICU = np.concatenate([self.dependency.date_to_hospitalizations_by_county,
                                                        np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_vaccinated = np.concatenate([self.dependency.date_to_vaccines_by_age,
                                                      np.zeros(shape=(y, 3, z))], axis=0)

        self.time_series_vaccinated = self.time_series_vaccinated.transpose(1, 0, 2)

        self.time_series_immunized = np.zeros(shape=(x, self.dependency.date_to_deaths_by_county.shape[1] + y,
                                                     z), dtype=float)
        self.time_series_immunity = np.zeros(shape=(x, self.dependency.date_to_deaths_by_county.shape[1] + y,
                                                    z), dtype=float)



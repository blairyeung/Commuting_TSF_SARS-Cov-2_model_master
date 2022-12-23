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
            self.time_series_vaccinated = np.zeros(shape=(y, z), dtype=int)

            self.time_series_immunity = np.zeros(shape=(x, self.time_series_len, z), dtype=float)

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


        self.time_series_immunized = np.zeros(shape=(x, y, z), dtype=float)
        self.time_series_immunity = np.zeros(shape=(x, y, z), dtype=float)

    def compute_immunity(self, date):
        """
        For EXTERNAL call only
        TODO: Find the vaccine efficacy and immunity waning function from papers, and
        TODO: calcualte the immunity from the vaccination and cases data
        This is hard and trciky!
        :return:
        """

        dose1 = (self.time_series_vaccinated[0])[:date]
        dose2 = (self.time_series_vaccinated[1])[:date]
        dose3 = (self.time_series_vaccinated[2])[:date]

        print(np.sum(self.time_series_vaccinated[0], axis=0))
        print(np.sum(self.time_series_vaccinated[1], axis=0))
        print(np.sum(self.time_series_vaccinated[2], axis=0))

        print('WHY?')

        print(np.sum(self.time_series_vaccinated[0]))
        print(np.sum(self.time_series_vaccinated[1]))
        print(np.sum(self.time_series_vaccinated[2]))

        print('FUCK?')

        print(np.sum(dose1))
        print(np.sum(dose2))
        print(np.sum(dose3))

        """
        print(np.sum(dose1, axis=0))
        print(np.sum(dose2, axis=0))
        print(np.sum(dose3, axis=0))
        """

        dose1 = (np.ones(shape=(Parameters.NO_COUNTY, dose1.shape[0], dose1.shape[1])) * dose1).transpose(1, 0, 2)
        dose2 = (np.ones(shape=(Parameters.NO_COUNTY, dose2.shape[0], dose2.shape[1])) * dose2).transpose(1, 0, 2)
        dose3 = (np.ones(shape=(Parameters.NO_COUNTY, dose3.shape[0], dose3.shape[1])) * dose3).transpose(1, 0, 2)


        county_population = np.array(self.dependency.index_to_population)
        county_population = county_population.reshape(county_population.shape[0], 1)

        age_population = self.dependency.population_by_age_band
        age_population = age_population.reshape(age_population.shape[0], 1)

        ratio = age_population / np.sum(age_population)

        print(county_population.shape)

        population = np.matmul(county_population, ratio.T)

        today_population = np.ones(shape=(date, population.shape[0], population.shape[1])) * population

        print(today_population.shape)

        today_cases = self.time_series_infected.transpose(1, 0, 2)[:date]

        today_incidence = today_cases / today_population

        self.time_series_incidence = today_incidence.transpose(1, 0, 2)

        # TODO: convolution on these three categories each with type (x, 16), respectively, and add up.


        raw_kernel_dose_1 = Parameters.VACCINE_EFFICACY_KERNEL_DOSE1[::-1]
        raw_kernel_dose_2 = Parameters.VACCINE_EFFICACY_KERNEL_DOSE2[::-1]
        raw_kernel_dose_3 = Parameters.VACCINE_EFFICACY_KERNEL_DOSE3[::-1]
        raw_kernel_infection = Parameters.INFECTION_EFFICACY_KERNEL[::-1]

        ratio = np.ones(shape=(1, Parameters.NO_COUNTY, 16))
        kernel_dose_1 = np.multiply(raw_kernel_dose_1[:date].reshape(date, 1, 1), ratio)
        kernel_dose_2 = np.multiply(raw_kernel_dose_2[:date].reshape(date, 1, 1), ratio)
        kernel_dose_3 = np.multiply(raw_kernel_dose_3[:date].reshape(date, 1, 1), ratio)
        kernel_infection = np.multiply(raw_kernel_infection[:date].reshape(date, 1, 1), ratio)

        print(kernel_dose_1.shape)

        immunity_dose1 = np.multiply(dose1, kernel_dose_1)

        immunity_dose2_rmv = np.multiply(dose2, kernel_dose_1)

        immunity_dose2 = np.multiply(dose2, kernel_dose_2)

        immunity_dose3 = np.multiply(dose3, kernel_dose_3)

        immunity_dose3_rmv = np.multiply(dose3, kernel_dose_2)


        immunity_infection = np.multiply(today_incidence, kernel_infection)

        print('HOLY!')
        print(np.max(immunity_infection))
        print(np.max(immunity_dose1))
        print(np.max(immunity_dose2))

        vaccine_immunity = immunity_dose1 + immunity_dose2 + immunity_dose3 - immunity_dose3_rmv - immunity_dose2_rmv

        immunity = vaccine_immunity + (1 - vaccine_immunity) * immunity_infection

        print(immunity.shape)
        print(np.max(immunity_infection))
        print(np.max(vaccine_immunity))
        print(np.max(immunity))

        today_immunity = np.sum(immunity, axis=0)
        today_vaccine_immunity = np.sum(vaccine_immunity, axis=0)

        print(today_immunity.shape)
        print(np.max(today_immunity))
        print(np.max(today_vaccine_immunity))



        return None

        """
        population = self.dependency.index_to_population

        vaccine_immunized_individuals = np.matmul(population.reshape(population.shape[0], 1),
                                          immunity_from_vaccine.reshape(16, 1).T)

        infected = self.time_series_infected

        infected = infected.transpose(1, 0, 2)[:date]
        infected = infected.transpose(1, 2, 0)
        raw_kernel_infected = Parameters.INFECTION_EFFICACY_KERNEL[::-1]
        kernel_infected = raw_kernel_infected[:date].reshape(date, 1)

        immunity_infected = np.matmul(infected, kernel_infected).reshape(528, 16)

        # print(immunity_infected.shape, vaccine_immunized_individuals.shape)
        infected_immunized_individuals = None

        self.time_series_immunized = self.time_series_immunized.transpose(1, 0, 2)

        self.time_series_immunized[date] = immunity_infected + vaccine_immunized_individuals

        # print(self.time_series_immunized.shape)

        self.time_series_immunized = self.time_series_immunized.transpose(1, 0, 2)
        # print(self.time_series_immunized.shape)
        # print(vaccine_immunized_individuals.shape)



         """



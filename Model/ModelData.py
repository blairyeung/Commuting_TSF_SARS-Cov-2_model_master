import copy
import numpy as np
import Parameters


class ModelData:
    """
        Attributes
    """

    forecast_days = 0

    time_series_len = 0

    prior_immunity = 0

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

    time_series_infection_immunity = None
    time_series_vaccinated = None
    time_series_vaccine_immunity = None

    time_series_immunity = None

    dependency = None

    def __init__(self, forecast_days=1000, dependency=None, load_from_dependency=False, prior_immunity=0):
        """
            Initialize the tensors
            shape=(x, y, z)
            Here, it should be (x * 528 * 16)
            x = number of counties
            y = number of days to forecast + dates elapsed since Covid outbreak
            z = 16 age bands
        """
        self.prior_immunity = prior_immunity
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

        # TODO: split it, to delta and omicron

        non_omicron = (self.time_series_clinical_cases.transpose(1, 0, 2)[:700] *
                       Parameters.REVERSE_CLINICAL_BY_AGE.T).transpose(1, 0, 2)

        omicron = (self.time_series_clinical_cases.transpose(1, 0, 2)[700:] *
                       Parameters.OMICRON_REVERSE_CLINICAL_BY_AGE.T).transpose(1, 0, 2)

        self.time_series_active_cases = np.concatenate([non_omicron, omicron], axis=1)

        # self.time_series_active_cases = np.multiply(self.time_series_clinical_cases,
        #                                             Parameters.REVERSE_CLINICAL_BY_AGE.T)

        self.time_series_exposed = np.zeros(shape=self.time_series_active_cases.shape).transpose(1, 0, 2)

        shifted = np.concatenate([self.time_series_active_cases.transpose(1, 0, 2)[3:],
                                  self.time_series_active_cases.transpose(1, 0, 2)[-1] *
                                  np.ones(shape=(3, 1, 1))]).transpose(1, 0, 2)

        self.time_series_exposed = shifted

        self.time_series_infected = copy.deepcopy(self.time_series_active_cases)

        self.time_series_sub_clinical_cases = self.time_series_active_cases - self.time_series_clinical_cases

        # Note: these values are obtained from convolution.

        self.time_series_recovered = np.apply_along_axis(lambda m:
                                                         np.convolve(m,
                                                                     Parameters.CLI2REC_CONVOLUTION_KERNEL,
                                                                     mode='same'), axis=1,
                                                         arr=self.time_series_clinical_cases)

        self.time_series_deaths = np.concatenate([self.dependency.date_to_deaths_by_county,
                                                  np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_hospitalized = np.concatenate([self.dependency.date_to_hospitalizations_by_county,
                                                        np.zeros(shape=(x, y, z))], axis=1)

        self.time_series_ICU = np.concatenate([self.dependency.date_to_ICU_by_county,
                                                        np.zeros(shape=(x, y, z))], axis=1)

        # TODO: Vaccination assumptions

        vaccine_adjust = np.concatenate([np.zeros(shape=(x, y, 2, z)), 0.001 * np.ones(shape=(x, y, 2, z))], axis=2)

        print(np.max(self.dependency.date_to_vaccines_by_county))

        self.time_series_vaccinated = np.concatenate([self.dependency.date_to_vaccines_by_county,
                                                      vaccine_adjust], axis=1)

        self.time_series_vaccinated = self.time_series_vaccinated.transpose(2, 0, 1, 3)

        self.time_series_infection_immunity = np.zeros(shape=(x, self.dependency.date_to_deaths_by_county.shape[1] + y,
                                                    z), dtype=float)

        self.time_series_vaccine_immunity = np.zeros(shape=(x, self.dependency.date_to_deaths_by_county.shape[1] + y,
                                                    z), dtype=float)

        self.time_series_immunity = np.zeros(shape=(x, self.dependency.date_to_deaths_by_county.shape[1] + y,
                                                    z), dtype=float)
        #
        # dose1 = (self.time_series_vaccinated[0])[:self.time_series_len]
        # dose2 = (self.time_series_vaccinated[1])[:self.time_series_len]
        # dose3 = (self.time_series_vaccinated[2])[:self.time_series_len]
        #
        # dose1 = (np.ones(shape=(Parameters.NO_COUNTY, dose1.shape[0], dose1.shape[1])) * dose1).transpose(1, 0, 2)
        # dose2 = (np.ones(shape=(Parameters.NO_COUNTY, dose2.shape[0], dose2.shape[1])) * dose2).transpose(1, 0, 2)
        # dose3 = (np.ones(shape=(Parameters.NO_COUNTY, dose3.shape[0], dose3.shape[1])) * dose3).transpose(1, 0, 2)
        #
        # county_population = np.array(self.dependency.index_to_population)
        # county_population = county_population.reshape(county_population.shape[0], 1)
        #
        # age_population = self.dependency.population_by_age_band
        # age_population = age_population.reshape(age_population.shape[0], 1)
        #
        # ratio = age_population / np.sum(age_population)
        #
        # population = np.matmul(county_population, ratio.T)
        #
        # today_population = np.ones(shape=(self.time_series_len, population.shape[0], population.shape[1])) * population
        #
        # today_cases = self.time_series_infected.transpose(1, 0, 2)[:self.time_series_len]
        #
        # today_incidence = today_cases / today_population
        #
        # self.time_series_incidence = today_incidence.transpose(1, 0, 2)
        #
        # raw_kernel_dose_1 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE1[:self.time_series_len])
        # raw_kernel_dose_2 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE2[:self.time_series_len])
        # raw_kernel_dose_3 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE3[:self.time_series_len])
        # raw_kernel_infection = (Parameters.INFECTION_EFFICACY_KERNEL[:self.time_series_len])
        #
        # ratio = np.ones(shape=(1, Parameters.NO_COUNTY, 1))
        # kernel_dose_1 = np.multiply(raw_kernel_dose_1.reshape(self.time_series_len, 1, 16), ratio)
        # kernel_dose_2 = np.multiply(raw_kernel_dose_2.reshape(self.time_series_len, 1, 16), ratio)
        # kernel_dose_3 = np.multiply(raw_kernel_dose_3.reshape(self.time_series_len, 1, 16), ratio)
        # kernel_infection = np.multiply(raw_kernel_infection.reshape(self.time_series_len, 1, 16), ratio)
        #
        #
        # for date in range(self.time_series_len - self.forecast_days - self.prior_immunity,
        #                   self.time_series_len - self.forecast_days):
        #     dose1 = (self.time_series_vaccinated[0])[:date]
        #     dose2 = (self.time_series_vaccinated[1])[:date]
        #     dose3 = (self.time_series_vaccinated[2])[:date]
        #
        #     dose1 = (np.ones(shape=(Parameters.NO_COUNTY, dose1.shape[0], dose1.shape[1])) * dose1).transpose(1, 0, 2)
        #     dose2 = (np.ones(shape=(Parameters.NO_COUNTY, dose2.shape[0], dose2.shape[1])) * dose2).transpose(1, 0, 2)
        #     dose3 = (np.ones(shape=(Parameters.NO_COUNTY, dose3.shape[0], dose3.shape[1])) * dose3).transpose(1, 0, 2)
        #
        #     county_population = np.array(self.dependency.index_to_population)
        #     county_population = county_population.reshape(county_population.shape[0], 1)
        #
        #     age_population = self.dependency.population_by_age_band
        #     age_population = age_population.reshape(age_population.shape[0], 1)
        #
        #     ratio = age_population / np.sum(age_population)
        #
        #     population = np.matmul(county_population, ratio.T)
        #
        #     today_population = np.ones(
        #         shape=(date, population.shape[0], population.shape[1])) * population
        #
        #     today_cases = self.time_series_infected.transpose(1, 0, 2)[:date]
        #
        #     today_incidence = today_cases / today_population
        #
        #     self.time_series_incidence = today_incidence.transpose(1, 0, 2)
        #
        #     raw_kernel_dose_1 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE1[:date])
        #     raw_kernel_dose_2 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE2[:date])
        #     raw_kernel_dose_3 = (Parameters.VACCINE_EFFICACY_KERNEL_DOSE3[:date])
        #     raw_kernel_infection = (Parameters.INFECTION_EFFICACY_KERNEL[:date])
        #
        #     ratio = np.ones(shape=(1, Parameters.NO_COUNTY, 1))
        #     kernel_dose_1 = np.multiply(raw_kernel_dose_1.reshape(date, 1, 16), ratio)
        #     kernel_dose_2 = np.multiply(raw_kernel_dose_2.reshape(date, 1, 16), ratio)
        #     kernel_dose_3 = np.multiply(raw_kernel_dose_3.reshape(date, 1, 16), ratio)
        #     kernel_infection = np.multiply(raw_kernel_infection.reshape(date, 1, 16), ratio)
        #
        #     immunity_dose1 = np.multiply(dose1, kernel_dose_1)
        #     immunity_dose2 = np.multiply(dose2, kernel_dose_2)
        #     immunity_dose3 = np.multiply(dose3, kernel_dose_3)
        #
        #     infection_immunity = np.multiply(today_incidence, kernel_infection)
        #     vaccine_immunity = immunity_dose1 + immunity_dose2 + immunity_dose3
        #
        #     vaccine_immunity[:][:][0] = 0
        #
        #     today_infection_immunity = np.sum(infection_immunity, axis=0)
        #     today_vaccine_immunity = np.sum(vaccine_immunity, axis=0)
        #
        #
        #     today_immunity = today_vaccine_immunity + (np.ones(shape=(Parameters.NO_COUNTY, 16)) -
        #                                                today_vaccine_immunity) \
        #                      * today_infection_immunity
        #
        #     data = self.time_series_immunity.transpose(1, 0, 2)
        #     data[date] = today_immunity
        #     self.time_series_immunity = data.transpose(1, 0, 2)
        #     print(date)
        #
        #

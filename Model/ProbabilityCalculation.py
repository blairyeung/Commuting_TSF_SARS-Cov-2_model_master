import cv2
import pandas as pd
import os
import csv
import numpy as np
import Dependency as Dependency
from datetime import datetime
import Parameters

tot_cases = []
cfr_by_pr = dict()
weekly_cfr_by_pr = dict()
no_cases_by_pr = dict()
no_hosp_by = dict()
no_icu_by = dict()
no_death_by = dict()


def fill_array(lst):
    for i in range(1, len(lst)):
        if lst[i] == 0:
            lst[i] = lst[i - 1]


def to_days(lst):
    new_lst = []
    date_lst = []
    for i in lst:
        this_day = datetime.strptime(i, '%Y-%m-%d')
        after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
        new_lst.append(after_outbreak)
        date_lst.append(this_day)
    return new_lst, date_lst


def get_cases_total():
    cases_read_path = d.get_dependency_path() + 'covid19-download.csv'
    df = pd.read_csv(cases_read_path)
    grouped = df.groupby(df.prname)
    tot_cases = np.array([])
    for i in range(len(grouped.groups)):
        # print(grouped.groups.keys())
        group_name = list(grouped.groups.keys())[i]
        group = grouped.get_group(group_name)
        time = group['date'].tolist()
        cases = np.clip(np.array(group['totalcases'].tolist()), a_max=100000000000, a_min=1)
        new_cases = np.clip(np.array(group['numtotal_last7'].tolist()), a_max=100000000000, a_min=1)
        deaths = np.array(group['numdeaths'].tolist())
        new_deaths = np.array(group['numdeaths_last7'].tolist())
        cfr = np.divide(deaths, cases)
        weekly_cfr = np.divide(new_deaths, new_cases)
        indices, dates = to_days(time)
        data = np.zeros(shape=(max(indices) + 1))
        deathdata = np.zeros(shape=(max(indices) + 1))
        for j in range(len(indices)):
            data[indices[j]] = new_cases[j]
            deathdata[indices[j]] = new_deaths[j]
        fill_array(data)
        fill_array(deathdata)
        length = data.shape[0]
        # Use Gaussian Blur to augment
        data = np.reshape(np.array(cv2.GaussianBlur(data, (25, 25), 5)), newshape=(length,))
        deathdata = np.reshape(np.array(cv2.GaussianBlur(deathdata, (25, 25), 5)), newshape=(length,))
        no_cases_by_pr[group_name] = list(np.round(data))
        if i == 0:
            tot_cases = np.array(list(np.round(data)))
            tot_deaths = np.array(list(np.round(deathdata)))
        else:
            tot_cases += np.array(list(np.round(data)))
            tot_deaths += np.array(list(np.round(deathdata)))
        cfr_by_pr[group_name] = cfr
        weekly_cfr_by_pr[group_name] = weekly_cfr

    cfr_df = pd.DataFrame(data=cfr_by_pr)
    weekly_cfr_df = pd.DataFrame(data=weekly_cfr_by_pr)
    cases_df = pd.DataFrame(data=no_cases_by_pr)

    # cfr_df.to_csv(d.get_dependency_path() + 'cfr_by_pr.csv')
    # weekly_cfr_df.to_csv(d.get_dependency_path() + 'weekly_cfr_by_pr.csv')
    # cases_df.to_csv(d.get_dependency_path() + 'cases_by_pr.csv')

    icu_read_path = d.get_dependency_path() + 'covid19-epiSummary-hospVentICU.csv'
    df2 = pd.read_csv(icu_read_path)
    # print(df2)
    time2 = df2['Date']
    days2, dates2 = to_days(time2)
    print(days2)
    hospitalized = df2['COVID_HOSP']
    icu = df2['COVID_ICU']
    vent = df2['COVID_VENT']

    no_icu_by_pr = np.zeros(shape=(max(days2) + 1))
    no_hosp_by_pr = np.zeros(shape=(max(days2) + 1))
    no_vent_by_pr = np.zeros(shape=(max(days2) + 1))

    for i in range(len(days2)):
        no_icu_by_pr[days2[i]] = icu[i]
        no_hosp_by_pr[days2[i]] = hospitalized[i]
        no_vent_by_pr[days2[i]] = vent[i]

    no_icu = np.divide(no_icu_by_pr[:tot_cases.shape[0]], np.clip(tot_cases[:no_icu_by_pr.shape[0]],
                                                                  a_max=1000000, a_min=1))
    no_hosp = np.divide(no_hosp_by_pr[:tot_cases.shape[0]], np.clip(tot_cases[:no_icu_by_pr.shape[0]],
                                                                    a_max=1000000, a_min=1))
    no_vent = np.divide(no_vent_by_pr[:tot_cases.shape[0]], np.clip(tot_cases[:no_icu_by_pr.shape[0]],
                                                                    a_max=1000000, a_min=1))
    no_deaths = np.divide(tot_deaths[:tot_cases.shape[0]], np.clip(tot_cases[:no_icu_by_pr.shape[0]],
                                                                   a_max=1000000, a_min=1))

    print(len(dates2))

    icu_rate_df = pd.DataFrame({'ICU': no_icu,
                                'Ventilator': no_vent,
                                'Hospitalization': no_hosp,
                                'CFR': no_deaths})

    # icu_rate_df.to_csv(d.get_dependency_path() + 'Hospitalizations.csv')


def get_hosp_total():
    return


if __name__ == '__main__':
    d = Dependency.Dependency()
    get_cases_total()

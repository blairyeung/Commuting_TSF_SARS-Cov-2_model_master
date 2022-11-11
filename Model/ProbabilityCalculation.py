import cv2
import pandas as pd
import os
import csv
import numpy as np
import Dependency as Dependency
from datetime import datetime
import Parameters


cfr_by_pr = None
no_cases_by_pr = None
no_hosp_by_pr = None
no_icu_by_pr = None
no_death_by_pr = None


def fill_array(lst):
    for i in range(1, len(lst)):
        if lst[i] == 0:
            lst[i] = lst[i-1]
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
    read_path = d.get_dependency_path() + 'covid19-download.csv'
    print(read_path)
    df = pd.read_csv(read_path)
    # print(df)
    grouped = df.groupby(df.prname)
    # print(grouped)

    no_cases_by_pr = dict()
    cfr_by_pr = dict()

    for i in range(len(grouped.groups)):
        # print(grouped.groups.keys())
        group_name = list(grouped.groups.keys())[i]
        group = grouped.get_group(group_name)
        time = group['date'].tolist()
        cases = np.clip(np.array(group['totalcases'].tolist()), a_max=100000000000, a_min=1)
        deaths = np.array(group['numdeaths'].tolist())
        # print(cases)
        # print(deaths)
        cfr = np.divide(deaths, cases)
        # print(cfr)
        # time = group.tolist()
        indices, dates = to_days(time)
        data = np.zeros(shape=(max(indices)+1))
        for j in range(len(indices)):
            data[indices[j]] = cases[j]
        fill_array(data)
        length = data.shape[0]
        data = np.reshape(np.array(cv2.GaussianBlur(data, (25, 25), 5)), newshape=(length, ))
        print(data.shape)
        no_cases_by_pr[group_name] = list(np.round(data))
        cfr_by_pr[group_name] = cfr

   #  print(no_cases_by_pr)

    # cfr_df = pd.DataFrame(data=cfr_by_pr)
    cases_df = pd.DataFrame(data=no_cases_by_pr)

    # cfr_df.to_csv(d.get_dependency_path() + 'cfr_by_pr.csv')
    cases_df.to_csv(d.get_dependency_path() + 'cases_by_pr.csv')


    pass

def get_hosp_total():
    return

if __name__ == '__main__':
    d = Dependency.Dependency()
    get_cases_total()

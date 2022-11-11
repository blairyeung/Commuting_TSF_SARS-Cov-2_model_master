import pandas as pd
import os
import csv
import numpy as np
import Dependency as Dependency
from datetime import datetime
import Parameters


death_rate_by_pr = None
no_cases_by_pr = None
no_hosp_by_pr = None
no_icu_by_pr = None
no_death_by_pr = None

def to_days(lst):
    new_lst = []
    for i in lst:
        this_day = datetime.strptime(i, '%Y-%m-%d')
        after_outbreak = (this_day - Parameters.OUTBREAK_FIRST_DAY).days
        new_lst.append(after_outbreak)
        #print(after_outbreak)
    return new_lst

def get_cases_total():
    read_path = d.get_dependency_path() + 'covid19-download.csv'
    print(read_path)
    df = pd.read_csv(read_path)
    # print(df)
    grouped = df.groupby(df.prname)
    # print(grouped)

    no_cases_by_pr = [None] * len(grouped.groups)

    for i in range(len(grouped.groups)):
        group = grouped.get_group(grouped.groups[i])
        time = group['date'].tolist()
        cases = np.array(group['avgcases_last7'].tolist())
        deaths = np.array(group['numdeaths_last7'].tolist())
        # time = group.tolist()
        dates = to_days(time)
        no_cases_by_pr[i] = data
    pass

def get_hosp_total():
    return

if __name__ == '__main__':
    d = Dependency.Dependency()
    get_cases_total()

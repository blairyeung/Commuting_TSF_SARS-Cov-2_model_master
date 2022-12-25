import csv
import Dependency
import os


def group_by_county(county_name, data, index):
    subpartition = []
    for row in data:
        if row[index] == county_name:
            subpartition.append(row)
    return subpartition


def slice_ontario():
    path = os.getcwd()[:-5] + 'Model Dependencies/'
    per_path = path + 'Region_Mobility_Report_CSVs/'
    years = ['2020', '2021', '2022']
    sur_path = '_CA_Region_Mobility_Report.csv'
    concat = []
    counties = []
    partition = []
    for year in years:
        read_path = per_path + year + sur_path
        with open(read_path) as file:
            contents = file.read()
            lines = contents.split('\n')
            first_line = lines[0].split(',')[1:]
            for i in range(1, len(lines) - 1):
                if 'Ontario' in lines[i]:
                    line = lines[i].replace('"', '')
                    raw = line.split(',')[1:]
                    elements = raw[0:3] + raw[7:-1]
                    county = elements[2]
                    if county not in counties:
                        counties.append(county)
                    concat.append(elements)
            file.close()

    counties[counties.index('')] = 'Ontario'

    for county_name in counties:
        partition.append(group_by_county(county_name, concat, index=2))

    first_line = first_line[0:3] + first_line[7:-1]

    for i in range(len(partition)):
        write_path = path + 'Ontario_mobility/' + counties[i] + '.csv'
        with open(write_path, 'w') as file:
            write = csv.writer(file, delimiter=',')
            write.writerow(first_line)
            write.writerows(partition[i])
            file.close()

    concat_file = path + 'Ontario_mobility/Concat_ontario.csv'
    with open(concat_file, 'w') as file:
        write = csv.writer(file, delimiter=',')
        write.writerow(first_line)
        write.writerows(concat)
        file.close()


slice_ontario()

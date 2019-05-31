import os, csv
import xlrd
import numpy as np



features = ['head_body_dist', 'body_speed_x',
            'head_speed_x' , 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

for status in ['Healthy', 'Sick']:
    sheet = xlrd.open_workbook('{} period_mouse.xlsx'.format(status))
    sheet = sheet.sheet_by_index(0)

    experiment_id = [1, 2, 3]
    mouse_id = ['L', 'R']
    day_id = []

    info = []
    print(sheet.ncols)
    for i in range(1, 16 * 2 + 1):
        info.append(sheet.col_values(i)[1:])

    info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20]]

    dates = sheet.col_values(0)[1:]

    exp_split_id = [i for i in range(1, len(dates) - 1) if dates[i] == '' and dates[i + 1] == '']
    exp_split_id.insert(0, 0)
    exp_split_id.append(len(dates))

    splitted = [[info[i][exp_split_id[j - 1]: exp_split_id[j]] for i in range(len(info))] for j in range(1, len(exp_split_id))]
    splitted_dates = [dates[exp_split_id[j - 1]: exp_split_id[j]] for j in range(1, len(exp_split_id))]
    #print(splitted_dates)

    to_csv = []
    for k in range(len(splitted)):
        splitted[k] = np.array(splitted[k]).T.tolist()

        day_split_id = [i for i in range(len(splitted_dates[k]) - 1) if splitted_dates[k][i] == '' and splitted_dates[k][i - 1] != '']
        day_split_id.insert(0, 0)
        day_split_id.append(len(splitted[k]))

        splitted_by_day = [splitted[k][day_split_id[j - 1]: day_split_id[j]] for j in
                    range(1, len(day_split_id))]
        splitted_dates_by_day = [splitted_dates[k][day_split_id[j - 1]: day_split_id[j]] for j in
                           range(1, len(day_split_id))]

        empty_ind = [i for i in range(len(splitted_dates_by_day)) if splitted_dates_by_day[i] == []]

        splitted_dates_by_day = [splitted_dates_by_day[i] for i in range(len(splitted_dates_by_day)) if i not in empty_ind]
        splitted_by_day = [splitted_by_day[i] for i in range(len(splitted_by_day)) if i not in empty_ind]

        for d in range(len(splitted_by_day)):
            for date, line in enumerate(splitted_by_day[d]):
                if '' not in line[:len(line) // 2]:
                    to_csv.append([k, status, 'L', d] + line[:len(line) // 2])

        for d in range(len(splitted_by_day)):
            for date, line in enumerate(splitted_by_day[d]):
                if '' not in line[len(line) // 2:]:
                    to_csv.append([k, status, 'R', d] + line[len(line) // 2:])

    to_csv.sort(key=lambda x: x[2])
    to_csv.insert(0, ['exp_id', 'status', 'mouse', 'day'] + features)


    with open('new_{}_table.csv'.format(status), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(to_csv)

# NORM!!!!!:
for status in ['Healthy', 'Sick']:
    with open('new_{}_table.csv'.format(status), newline='') as csvfile:
        data = list(csv.reader(csvfile))[1:]

    data = np.array(data)

    sorted_by_mouse_and_exp = [0, 0]
    sorted_by_mouse_and_exp[0] = data[np.where(data[:, 2] == 'L')]
    sorted_by_mouse_and_exp[1] = data[np.where(data[:, 2] == 'R')]

    for m in range(2):
        print(sorted(set(sorted_by_mouse_and_exp[m][:, 0].tolist())))
        sorted_by_mouse_and_exp[m] = [sorted_by_mouse_and_exp[m][np.where(sorted_by_mouse_and_exp[m][:, 0] == exp)]
                                      for exp in sorted(set(sorted_by_mouse_and_exp[m][:, 0].tolist()))]

    for m in range(2):
        for exp in range(len(sorted_by_mouse_and_exp[m])):
            sorted_by_mouse_and_exp[m][exp] = sorted_by_mouse_and_exp[m][exp][:, 4:].astype(float)

            print(np.mean(sorted_by_mouse_and_exp[m][exp], axis=0)[0], (np.quantile(sorted_by_mouse_and_exp[m][exp], 0.9, axis=0) - np.quantile(
                                                  sorted_by_mouse_and_exp[m][exp], 0.1, axis=0))[0])
            sorted_by_mouse_and_exp[m][exp] = (sorted_by_mouse_and_exp[m][exp] - np.mean(
                sorted_by_mouse_and_exp[m][exp], axis=0)) / \
                                              (np.quantile(sorted_by_mouse_and_exp[m][exp], 0.9, axis=0) - np.quantile(
                                                  sorted_by_mouse_and_exp[m][exp], 0.1, axis=0))[0]

    # SAVE NORM TABLES:
    to_csv = []
    data_ind = 0
    for m in range(2):
        for exp in range(len(sorted_by_mouse_and_exp[m])):
            for d in range(len(sorted_by_mouse_and_exp[m][exp])):
                to_csv.append(data[data_ind][:4].tolist() + sorted_by_mouse_and_exp[m][exp][d].tolist())
                data_ind += 1

    to_csv.insert(0, ['exp_id', 'status', 'mouse', 'day'] + features)

    with open('new_norm_{}_table.csv'.format(status), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(to_csv)

    ###
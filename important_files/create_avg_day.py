import os, re, sys
import csv
import json
import xlrd, xlwt
import numpy as np


N_FEATURES = 16

features = ['head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp', 'img', 'date'#, 'frames',
            ]


os.chdir('/root/Novosad/mouses/')


def create_avg(dirname, mouse):
    if not os.path.isdir(os.path.join('check', dirname, 'xlses', 'avg')):
        os.mkdir(os.path.join('check', dirname, 'xlses', 'avg'))

    data_path = os.path.join('check', dirname, 'xlses', 'data')
    avg_path = os.path.join('check', dirname, 'xlses', 'avg')
    
    xlses = [f for f in os.listdir(data_path) if f.split('_')[0] == mouse]
    xlses.sort(key=lambda x: (int(re.split('[._]', x)[3]), re.split('[._]', x)[1],
                              int(re.split('[._]', x)[2]), re.split('[._]', x)[6],
                              int(re.split('[._]', x)[4]) % 12, int(re.split('[._]', x)[5])))

    data = [[] for i in range(N_FEATURES)]
    dates = [[] for i in range(2)]

    for _, xls in enumerate(xlses):
        print(xls)

        with open(os.path.join(data_path, xls), newline='') as csvfile:
            cur_data = list(csv.reader(csvfile))
            if len(cur_data) > 20:
                cur_data = np.array(cur_data).T.tolist()

        for i in range(N_FEATURES):
            data[i] += np.array(cur_data[i]).astype(float).tolist()

        dates[0] += cur_data[-2]
        dates[1] += cur_data[-1]

        # sheet = xlrd.open_workbook(os.path.join(data_path, xls))
        # sheet = sheet.sheet_by_index(0)
        #
        # for i in range(N_FEATURES):
        #     feature = np.array(sheet.col_values(i)).astype(float)
        #     data[i] += feature.tolist()
        #
        # for i in [17, 18]:
        #     feature = sheet.col_values(i)
        #     dates[i - 17] += feature

    data = np.array(data).T.tolist()
    bad = np.where(np.isnan(data[:]))[0].tolist()
    for i in bad:
        data[i][2] = 0
        data[i][4] = 0

    times = dates[1]

    dates = np.array(dates).T.tolist()

    day_indexes = [i for i in range(1, len(times)) if '12:00:00 AM' in times[i] and '12:00:00 AM' not in times[i - 1]]
    print(day_indexes)
    day_indexes = [0] + day_indexes + [len(times) - 1]

    data_split_by_days = [data[day_indexes[i]: day_indexes[i + 1]] for i in range(len(day_indexes) - 1)]
    dates_split_by_days = [dates[day_indexes[i]: day_indexes[i + 1]] for i in range(len(day_indexes) - 1)]

    data_split_by_days = data_split_by_days[1:-1]
    data_split_by_days = data_split_by_days[-3:]

    dates_split_by_days = dates_split_by_days[1:-1]
    dates_split_by_days = dates_split_by_days[-3:]

    tt = 24 * 6
    if not os.path.isdir(os.path.join('check', dirname, 'xlses', 'avg', str(tt))):
        os.mkdir(os.path.join('check', dirname, 'xlses', 'avg', str(tt)))

    mouse = 'left_' if mouse == '0' else 'right_'
    sheet1 = xlwt.Workbook()
    table = sheet1.add_sheet('data from mouse')

    for kk in range(len(features)):
        table.write(0, kk, mouse + features[kk])

    xls_ind = 1

    avg_day = [[0 for i in range(len(data_split_by_days))] for j in range(tt)]
    avg_dates = [0 for i in range(tt)]

    for day in range(len(data_split_by_days)):
        frames = len(data_split_by_days[day]) // tt

        splitted_data = [data_split_by_days[day][i * frames: (i + 1) * frames] for i in range(tt)]
        splitted_dates = [dates_split_by_days[day][i * frames: (i + 1) * frames] for i in range(tt)]

        for _ in range(tt):
            mean_values = np.mean(splitted_data[_], axis=0)
            avg_day[_][day] = mean_values.tolist()
            date = splitted_dates[_][0]
            if day == 0:
                avg_dates[_] = date

            # TABLE

    for t in range(tt):
   #     print(np.array([avg_day[t][day] for day in range(len(data_split_by_days))]))
        avg_day[t] = np.mean(np.array([avg_day[t][day] for day in range(len(data_split_by_days))]), axis=0).tolist()

    for t in range(tt):
        for s in range(len(avg_day[t])):
            table.write(xls_ind, s, avg_day[t][s])

        table.write(xls_ind, s + 1, avg_dates[t][0])
        table.write(xls_ind, s + 2, avg_dates[t][1])

        xls_ind += 1

    sheet1.save(os.path.join('check', dirname, 'xlses', 'avg', str(tt), '{}avg_day.xls'.format(mouse)))


if __name__ == '__main__':
    create_avg(sys.argv[1], sys.argv[2])
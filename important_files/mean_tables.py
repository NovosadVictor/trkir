import os, re, sys

import json
import xlrd, xlwt
import numpy as np

from datetime import datetime, timedelta


N_FEATURES = 16
TIME = 10000

features = ['head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp', 'img', 'date', 'time',
            ]


def mean_tables(mouse_n, dirname, subdirname):
    if not os.path.isdir(os.path.join('check', dirname, 'xlses', 'mean')):
        os.mkdir(os.path.join('check', dirname, 'xlses', 'mean'))

    mouse = 'left' if mouse_n == '0' else 'right'

    xlses = [f for f in os.listdir(os.path.join('check', dirname, 'xlses', 'data'))
             if f.split('_')[0] == mouse_n and subdirname in f]

    data = [[] for i in range(N_FEATURES)]
    dates = [[] for i in range(2)]
    
    for _, xls in enumerate(xlses):
        print(xls)
    
        sheet = xlrd.open_workbook(os.path.join('check', dirname, 'xlses', 'data', xls))
        sheet = sheet.sheet_by_index(0)
    
        for i in range(N_FEATURES):
            feature = np.array(sheet.col_values(i)).astype(float)
            data[i] += feature.tolist()

        for i in [16, 17]:
            feature = sheet.col_values(i)
            dates[i - 16] += feature
    
    data = np.array(data).T.tolist()
    dates = np.array(dates).T.tolist()

    bad = np.where(np.isnan(data[:]))[0].tolist()
    for i in bad:
        data[i][2] = 0
        data[i][4] = 0

    splitted_data = [data[i * TIME: (i + 1) * TIME] for i in range(len(data) // TIME)]
    splitted_dates = [dates[i * TIME: (i + 1) * TIME] for i in range(len(dates) // TIME)]

    sheet1 = xlwt.Workbook()
    table = sheet1.add_sheet('data from mouse')

    for kk in range(len(features)):
        table.write(0, kk, mouse[0] + '_' + features[kk])

    xls_ind = 1
    for _ in range(len(splitted_data)):
        mean_values = np.mean(splitted_data[_], axis=0)
        date = splitted_dates[_][0]

        # TABLE
        for s in range(len(mean_values.tolist())):
            table.write(xls_ind, s, mean_values[s])

        table.write(xls_ind, s + 1, date[0])
        table.write(xls_ind, s + 2, date[1])

        xls_ind += 1

    sheet1.save(os.path.join('check', dirname, 'xlses', 'mean', xls))

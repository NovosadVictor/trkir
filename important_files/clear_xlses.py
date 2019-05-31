import os,re, sys
import csv

import xlrd, xlwt
import numpy as np

from datetime import datetime, timedelta


os.chdir('/root/Novosad/mouses/')


def clear(dirname, subdirname):
    print('Clear start')
    xlses = [f for f in os.listdir(os.path.join('check', dirname, 'xlses', subdirname)) if f[-4:] == '.xls']
    xlses.sort(key=lambda x: int(re.split('[._]', x)[-3]))

    print(xlses)

    dates = [[], [], [], []]
    info = [[] for i in range(14)]

    for xls in xlses:
        sheet = xlrd.open_workbook(os.path.join('check', dirname, 'xlses', subdirname, xls))
        sheet = sheet.sheet_by_index(0)

        for i in range(14):
            info[i] += sheet.col_values(i)

        dates[0] += sheet.col_values(14)
        dates[1] += sheet.col_values(15)
        dates[2] += sheet.col_values(16)
        dates[3] += sheet.col_values(17)

    dates = np.array(dates).T.tolist()
    info = np.array(info).T.tolist()

    print(len(dates), len(info))

    # splitted_dates = [dates[: 60000]]
    # splitted_info = [info[: 60000]]

    data = np.concatenate([info, dates], axis=1)

    if not os.path.isdir(os.path.join('check', dirname, 'xlses', 'cleared')):
        os.mkdir(os.path.join('check', dirname, 'xlses', 'cleared'))

    with open(os.path.join('check', dirname, 'xlses', 'cleared', '{}.csv'.format(subdirname)), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    #
    # for _ in range(len(splitted_dates)):
    #     sheet1 = xlwt.Workbook()
    #     table = sheet1.add_sheet('data from mouse')
    #     for i in range(len(splitted_dates[_])):
    #         for col, elem in enumerate((splitted_info[_][i] + splitted_dates[_][i])):
    #             table.write(i, col, elem)
    #
    #     sheet1.save(os.path.join('check', dirname, 'xlses', 'cleared', '{}.xls'.format(subdirname)))
    print('Clear finish')


if __name__ == '__main__':
    clear(sys.argv[1], sys.argv[2])

import os, sys
import re
import csv
from datetime import datetime, timedelta

import xlrd, xlwt
import numpy as np


os.chdir('/root/Novosad/mouses/')


def add_empt(dirname, subdirname):
    print('Empty start')

    data_path = os.path.join('check', dirname, 'xlses', 'cleared')

    xlses = [f for f in os.listdir(data_path) if subdirname in f]
    
    for xls in xlses:
        with open(os.path.join(data_path, xls), newline='') as csvfile:
            data = list(csv.reader(csvfile))
            if len(data) > 20:
                data = np.array(data).T.tolist()

        mouses_info = [np.array([data[7 * m + i] for i in range(7)]).T.tolist() for m in range(2)]
        dates = data[-4:]

        # mouses_info = [[[] for i in range(7)], [[] for i in range(7)]]
        #
        # dates = [[], [], [], []]
        #
        # sheet = xlrd.open_workbook(os.path.join(data_path, xls))
        # sheet = sheet.sheet_by_index(0)
        #
        # for m in range(2):
        #     for f in range(7):
        #         mouses_info[m][f] = sheet.col_values(7 * m + f)
        #
        # for d in range(4):
        #     dates[d] = sheet.col_values(14 + d)
        #
        # for m in range(2):
        #     mouses_info[m] = np.array(mouses_info[m]).T.tolist()
        #
        dates = np.array(dates).T.tolist()

        for i in range(len(dates)):
                if '' in mouses_info[0][i] + mouses_info[1][i] or '' in dates[i]:
                    print(dates[i - 1])
                    mouses_info[0][i] = mouses_info[0][i - 1]
                    mouses_info[1][i] = mouses_info[1][i - 1]
                    dates[i] = dates[i - 1]

        for i in range(len(dates)):
            date = '%02d/%02d/%4d' % (
            int(dates[i][1].split('/')[0]), int(dates[i][1].split('/')[1]), int(dates[i][1].split('/')[2]))
            datetime_object = datetime.strptime(date + dates[i][2] + dates[i][3], '%m/%d/%Y%I:%M:%S%p')
            new_data = datetime_object - timedelta(hours=6)
            new_data = new_data.strftime('%m/%d/%Y %I:%M:%S %p')
            dates[i] = [dates[i][0], new_data]

        print(xls, len(dates), len(mouses_info[0]))

        data = np.concatenate([mouses_info[0], mouses_info[1], dates], axis=1)

        with open(os.path.join(data_path, xls), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

        # sheet1 = xlwt.Workbook()
        # table = sheet1.add_sheet('data from ' + ' mouse')
        # for i in range(len(dates)):
        #     for col, elem in enumerate(mouses_info[0][i] + mouses_info[1][i] + dates[i]):
        #         table.write(i, col, elem)
        #
        # sheet1.save(os.path.join(data_path, xls))
    print('Empty finish')


if __name__ == '__main__':
    add_empt(sys.argv[1], sys.argv[2])
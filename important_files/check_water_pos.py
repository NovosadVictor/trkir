import os
import sys
import re
from shutil import rmtree

from send_email import send_mail

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

import csv
import numpy as np

from datetime import datetime, timedelta

os.chdir('/root/Novosad/mouses/')


def water_position(dirname):

    csv_path = os.path.join('check', dirname, 'xlses', 'cleared')
    data_path = os.path.join('check', dirname)

    if os.path.isdir(os.path.join(data_path, 'drinking_pos')):
        rmtree(os.path.join(data_path, 'drinking_pos'))

    os.mkdir(os.path.join(data_path, 'drinking_pos'))

    csv_files = os.listdir(csv_path)

    csv_files.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[0]),
                                  int(x.split('_')[1]), x.split('_')[5],
                                  int(x.split('_')[3]) % 12, int(x.split('_')[4])))

    csv_files = csv_files[:]

    data = [[] for i in range(2 * 7)]
    dates = []

    for xls in csv_files:
        print(xls)

        with open(os.path.join('check', dirname, 'xlses', 'cleared', xls), newline='') as csvfile:
            cur_data = list(csv.reader(csvfile))
            if len(cur_data) > 20:
                cur_data = np.array(cur_data).T.tolist()

        for i in range(len(data)):
            data[i] += cur_data[i]
        dates += cur_data[-1]

    mouses_info = [[data[7 * m + i] for i in range(2)] for m in range(2)]

    for m in range(2):
        mouses_info[m] = np.array(mouses_info[m]).T.astype(float).tolist()

    is_water_pos = [[0] * len(dates), [0] * len(dates)]

    for i in range(len(mouses_info[0])):
        if (140 <= mouses_info[0][i][0] <= 220) and mouses_info[0][i][1] <= 105:
            is_water_pos[0][i] = 1
    for i in range(len(mouses_info[1])):
        if (400 <= mouses_info[1][i][0] <= 480) and mouses_info[1][i][1] <= 105:
            is_water_pos[1][i] = 1

    start_date = datetime.strptime(dates[0], '%m/%d/%Y %I:%M:%S %p')

    bias = (int(dates[0].split()[1].split(':')[0]) + 1) % 12
    if 'PM' in dates[0]:
        bias += 12

   # bias = 0  # !!!!!!!!!!!

    start_date = start_date.replace(microsecond=0, second=0, minute=0) + timedelta(hours=1)
    print(start_date)

    last_date = datetime.strptime(dates[-1], '%m/%d/%Y %I:%M:%S %p')
    last_date = last_date.replace(microsecond=0, second=0, minute=0)
    print(last_date)

    total_hours = last_date - start_date
    total_hours = total_hours.days * 24 + total_hours.seconds // 3600

    print(total_hours)

    start_date = start_date.strftime('%m/%d/%Y %I:%M:%S %p')
    last_date = last_date.strftime('%m/%d/%Y %I:%M:%S %p')

    print(last_date, dates[-10])

    for i in range(len(dates)):
        if dates[i] == start_date:
            start_ind = i
            break

    for i in reversed(range(len(dates))):
        if dates[i] == last_date:
            end_ind = i
            break

    dates = dates[start_ind: end_ind]

    for i in range(2):
        is_water_pos[i] = is_water_pos[i][start_ind: end_ind]

        frames = len(is_water_pos[i]) // total_hours

        splitted_data = [is_water_pos[i][j * frames: (j + 1) * frames] for j in range(total_hours)]

        cur_feature = []
        for _ in range(total_hours):
            mean_value = np.sum(splitted_data[_])
            cur_feature.append(mean_value)

        is_water_pos[i] = cur_feature

        start_time = datetime.strptime('%d:30' % bias, '%H:%M')

        mouse = 'left' if i == 0 else 'right'
        color = 'blue' if i == 0 else 'red'

        plt.figure(figsize=(14, 8))
        plt.bar([i for i in range(len(is_water_pos[i]))], is_water_pos[i], color=color)
        plt.xticks([4 * i for i in range(len(is_water_pos[i]))], [(start_time + timedelta(hours=4 * i)).strftime("%H:%M") for i in range(len(is_water_pos[i]))], rotation=60)
        plt.axis([0, len(is_water_pos[i]), 0, np.max(is_water_pos[i])])
        plt.title('Time in drinking position\n{} mouse, sum for {} hour intervals'.format(mouse, 1))
        plt.savefig(os.path.join(data_path, 'drinking_pos', mouse + '.png'))

    send_mail('drinking position plots, {} record'.format(dirname), img_dir=os.path.join(data_path, 'drinking_pos'))


if __name__ == '__main__':
    water_position(sys.argv[1])
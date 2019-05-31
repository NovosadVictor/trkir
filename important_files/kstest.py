import os, re, sys, json

import xlrd, xlwt
import csv
import numpy as np
from datetime import datetime, timedelta
from shutil import rmtree

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
from send_email import send_mail

from scipy.stats import ks_2samp


features = ['head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
         #   'temp',
            ]

profit_f = features#['head_body_dist', #'body_speed_x', 'body_speed_y',
   #         'head_speed_x', 'head_speed_y',
   #         'head_rotation', 'body_speed', 'head_speed',
   #         'head_coord_x', 'head_coord_y',
   #         'temp',
           # ]

N_FEATURES = len(features)

os.chdir('/root/Novosad/mouses/')


def test(mouse_n, dirname, ks=True):
    mouse = 'left' if mouse_n == '0' else 'right'

    data_path = os.path.join('check', dirname)

    if os.path.isdir(os.path.join(data_path, 'p_value_plots')):
        rmtree(os.path.join(data_path, 'p_value_plots'))
    if os.path.isdir(os.path.join(data_path, 'xlses', 'mean')):
        rmtree(os.path.join(data_path, 'xlses', 'mean'))


    os.mkdir(os.path.join(data_path, 'p_value_plots'))
    os.mkdir(os.path.join(data_path, 'xlses', 'mean'))

    xlses = [f for f in os.listdir(os.path.join(data_path, 'xlses', 'concatenated')) if f[0] == mouse_n]

    plot_features = {feature: [] for feature in profit_f}
    dates = []
    avg_day = {feature: [] for feature in features}

    with open(os.path.join(data_path, 'xlses', 'concatenated', xlses[0]), newline='') as csvfile:
        data = list(csv.reader(csvfile))
        print(len(data), len(data[0]))
        #
        # dates = data[-1][:]
        #
        # ind = dates.index('09/27/2018 12:52:25 AM')
        # for i in range(len(data)):
        #     data[i] = data[i][ind:]
        # print(ind)

        dates = data[-1][:]
        print(dates[0], dates[-1])

        for i in range(N_FEATURES):
            if features[i] in profit_f:
                print(features[i])
                plot_features[features[i]] = np.array(data[i][:]).astype(float).tolist()

        print('deleting')
        del data

    start_ind = 0
    end_ind = len(dates)

    start_date = datetime.strptime(dates[0], '%m/%d/%Y %I:%M:%S %p')

    bias = (int(dates[0].split()[1].split(':')[0]) + 1) % 12
    if 'PM' in dates[0]:
        bias += 12

  #  bias = 0 # !!!!!!!!!!!

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

    for i in range(len(dates)):
        if dates[i] == start_date:
            start_ind = i
            break

    for i in reversed(range(len(dates))):
        if dates[i] == last_date:
            end_ind = i
            break

    dates = dates[start_ind: end_ind]

    for i in range(N_FEATURES):
        if features[i] in profit_f:
            plot_features[features[i]] = plot_features[features[i]][start_ind: end_ind]

            frames = len(plot_features[features[i]]) // (total_hours * 6)

            splitted_data = [plot_features[features[i]][j * frames: (j + 1) * frames] for j in range(total_hours * 6)]

            cur_feature = []
            cur_dates = []
            for _ in range(total_hours * 6):
                mean_value = np.mean(splitted_data[_])
                cur_feature.append(mean_value)

                if i == 0:
                    cur_dates.append(dates[frames * _])

            if i == 0:
                dates = cur_dates[:]
            plot_features[features[i]] = cur_feature

    # CSV
    data_to_csv = [plot_features[feature] for feature in profit_f]
    data_to_csv.insert(0, dates)
    data_to_csv = np.array(data_to_csv).T.tolist()
    data_to_csv = [['id'] + features] + data_to_csv

    with open(os.path.join(data_path, 'xlses', 'mean', mouse + '_mean_table.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_csv)

    if ks:
        avg_day_xlses = [f for f in os.listdir(os.path.join(data_path, 'xlses', 'avg', '144')) if f[0] == mouse[0]]
        for xls in avg_day_xlses:
            sheet = xlrd.open_workbook(os.path.join(data_path, 'xlses', 'avg', '144', xls))
            sheet = sheet.sheet_by_index(0)

            for feature in range(N_FEATURES):
                avg_day[features[feature]] += np.array(sheet.col_values(feature)[1:]).astype(float).tolist()

        for s in range(2):
            for feature in profit_f:
                cur_mean_list = []
                for i in range(len(avg_day[feature]) - 1):
                    cur_mean_list.append(avg_day[feature][i])
                    cur_mean_list.append((avg_day[feature][i] + avg_day[feature][i + 1]) / 2)
                cur_mean_list.append(avg_day[feature][-1])
                avg_day[feature] = cur_mean_list
                cur_list = []
                cur_dates = []
                for i in range(len(plot_features[feature]) - 1):
                    cur_list.append(plot_features[feature][i])
                    cur_list.append((plot_features[feature][i] + plot_features[feature][i + 1]) / 2)

                    if feature == features[0]:
                        cur_dates.append(dates[i])
                        cur_dates.append(dates[i])
                cur_list.append(plot_features[feature][-1])

                if feature == features[0]:
                    cur_dates.append(dates[-1])
                    dates = cur_dates[:]
                plot_features[feature] = cur_list

        for feature in profit_f:
            if feature not in [features[2], features[3]]:
                p_values = []
                start_time = datetime.strptime('%d:30' % bias, '%H:%M')
                q = len(plot_features[feature]) // total_hours
                qq = float(len(plot_features[feature])) / total_hours
                for k in range(total_hours):

                    p_value = list(ks_2samp(plot_features[feature][k * q: (k + 1) * q],
                                            avg_day[feature][((bias + k) % 24) * q: (((bias + k) % 24) + 1) * q]))[-1]

                    pred = p_values[-1] if len(p_values) > 0 else 0
                    p_values.append(pred + 1 if p_value < 0.05 else 0)

                plt.figure(figsize=(14, 8))
                plt.subplot(211)
                plt.title('feature plot; {}\n{} mouse'.format(feature, mouse))
                plt.plot(plot_features[feature], color='C0')
                plt.xticks([2 * i * qq for i in range(total_hours // 2)],
                           [(start_time + timedelta(hours=2 * i - 0.5)).strftime("%H:%M") for i in range(total_hours // 2)], rotation=60)
                plt.subplot(212)
                plt.plot(avg_day[feature], color='C3')
                start_time_fn = datetime.strptime('12:30 AM', '%I:%M %p')
                plt.xticks([i * qq for i in range(24)],
                            [(start_time_fn + timedelta(hours=i - 0.5)).strftime("%H:%M") for i in range(24)], rotation=60)
                #plt.axis([0, qq * 24, np.min(plot_features[feature]), np.max(plot_features[feature])])
                plt.savefig(os.path.join(data_path, 'p_value_plots', mouse + '_data_' + feature + '.png'))
                plt.close()

                plt.figure(figsize=(14, 8))
                plt.plot([i for i in range(len(p_values))], p_values, 'C%d' % (3 * int(mouse_n)))
                plt.title('K-S test p_values; {}\n{} mouse {} dir'.format(feature, mouse, dirname))
                plt.xticks([2 * i for i in range(len(p_values))], [(start_time + timedelta(hours=2 * i)).strftime("%H:%M") for i in range(len(p_values))], rotation=60)
                plt.axis([0, len(p_values) - 1, 0, np.max(p_values)])
                plt.savefig(os.path.join(data_path, 'p_value_plots', mouse + '_' + feature + '.png'))
                plt.close()

        send_mail('p-value and feature plots, {} mouse {} record'.format(mouse, dirname), img_dir=os.path.join(data_path, 'p_value_plots'))
        send_mail('mean table for {} mouse, {} record'.format(mouse, dirname), img_dir=os.path.join(data_path, 'xlses', 'mean'))


if __name__ == '__main__':
    test(sys.argv[1], sys.argv[2], ks=False)

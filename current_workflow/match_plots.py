import os, sys
import csv
import xlrd
from datetime import datetime, timedelta

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw, dtw

import numpy as np
from numpy.linalg import norm

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def move_day(day, to_day):
    dist = metric(day[:len(to_day)], to_day[:len(day)])
    best = 0

    is_sec = False
    for i in range(1, 30):#len(to_day) - len(day)):
        cur_to_compare = to_day[i: i + len(day)]

        # if i < 15:
        #     plt.figure()
        #     plt.plot(day)
        #     plt.plot(cur_to_compare)
        #     plt.show()
        cur_dist = metric(cur_to_compare, day[:len(cur_to_compare)])
        print(dist, cur_dist)
        if cur_dist < dist:
            dist = cur_dist
            best = i

    # print('hehe')
    # for i in range(1, 30):#len(day) - len(to_day)):
    #     cur_to_compare = day[i: i + len(to_day)]
    #     #
    #     # if i < 15:
    #     #     plt.figure()
    #     #     plt.plot(to_day)
    #     #     plt.plot(cur_to_compare)
    #     #     plt.show()
    #
    #     cur_dist = metric(cur_to_compare, day[:len(cur_to_compare)])
    #     print(dist, cur_dist)
    #     if cur_dist < dist:
    #         is_sec = True
    #         dist = cur_dist
    #         best = i

    print('best = ', best)
    if is_sec:
        day = day[best:]
    else:
        to_day = to_day[best: ]
    return day, to_day#[:len(to_day)], to_day


def mine_dtw(series_l, series_r):

    print(len(series_l), len(series_r))
    first = series_l[:]
    second = series_r[:]
    # second = ((np.array(series_r[:len(series_l)]) - np.min(series_r[:len(series_l)])) / (
    #             np.max(series_r[:len(series_l)]) - np.min(series_r[:len(series_l)]))) * (
    #                      np.max(series_l) - np.min(series_l)) + np.min(series_l)

    dist, path = dtw(first, second, dist=euclidean)

    res = [0 for i in range(len(series_l))]
    for i in range(len(path)):
        if res[path[i][1]] == 0:
            res[path[i][1]] = (path[i][0], path[i][1])

    new_plt = np.array([first[val[0]] for val in res])
   #  new_plt = ((np.array([first[val[0]] for val in res]) - np.min(series_l)) / (
   #               np.max(series_l) - np.min(series_l))) * (
   #                         np.max(series_r[:len(series_l)]) - np.min(
   #                     series_r[:len(series_l)])) + np.min(series_r[:len(series_l)])

    return new_plt, path, dist


def find_peaks(time_series, first=True, last=True):
    bias = 90

    first_peak = 0
    last_peak = len(time_series) - 1

    if first:
        for i in range(len(time_series) - bias):
            if i < bias:
                peak = True
                for k in range(bias + i):
                    if time_series[i + bias - k] < time_series[i]:
                        peak = False
                        break

                if peak:
                    first_peak = i
                    break
            else:
                peak = True
                for k in range(2 * bias):
                    if time_series[i + bias - k] < time_series[i]:
                        peak = False
                        break

                if peak:
                    first_peak = i
                    break

        print(first_peak)

    if last:
        for i in reversed(range(bias, len(time_series))):
            if i > len(time_series) - bias:
                peak = True
                for k in range(bias + len(time_series) - i):
                    if time_series[i - bias + k] < time_series[i]:
                        peak = False
                        break

                if peak:
                    last_peak = i
                    break
            else:
                peak = True
                for k in range(2 * bias):
                    if time_series[i - bias + k] < time_series[i]:
                        peak = False
                        break

                if peak:
                    last_peak = i
                    break

#        print(last_peak, len(time_series))

    return time_series[first_peak: last_peak + 1]


def metric(plot_1, plot_2):
    return np.mean((np.array(plot_1) - np.array(plot_2)) ** 2)


features = ['head_body_dist', 'body_speed_x',
            'head_rotation', 'body_speed',
            'temp',
            ]

feature_inexes = [1, 2, 7, 8, 16]

# 1 STEP: 3 days healthy mouse from 2 x 3 + 1 x 8 days


for _, ind in enumerate(feature_inexes):
    feature = features[_]

    sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
    sheet = sheet.sheet_by_index(0)
    data = np.array(sheet.col_values(ind)[2:])
    data_r = np.array(sheet.col_values(ind + 16)[2:])
    indexes = np.where(data == '')[0].tolist()
    indexes_r = np.where(data_r == '')[0].tolist()
    indexes.insert(0, 0)
    indexes_r.insert(0, 0)
    #print(indexes, indexes_r)
    start_time = datetime.strptime('1/1/2018 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')

    l_3_days = data[1:407]
    r_3_days = data_r[1:407]
    r_8_days = data_r[407: 1464]
    l_1_day = data[1465:1598]
    r_1_day = data_r[1465:1598]
    l_2_day = data[1599:]
    r_2_day = data_r[1599:]

    days = [l_3_days, r_3_days, r_8_days[:max(len(l_3_days), len(r_3_days))], l_1_day]

    period = 4
    mean_days = [[] for i in range(len(days))]

    #plt.figure(figsize=(14, 8))
    for i in range(len(days)):
        days[i] = days[i][np.where(days[i] != '')]
        days[i] = days[i][np.where(days[i] != 'PV1')]

        days[i] = days[i].astype(float).tolist()

        for j in range(period, len(days[i]) - period):
            mean_days[i].append(np.mean(days[i][j - period: j + period]))

        #plt.plot(mean_days[i])

    mean_healthy_temp = [np.mean([mean_days[0][i], mean_days[1][i], mean_days[2][i]]) for i in range(len(mean_days[0]))]

   # plt.figure()
   # plt.plot(mean_healthy_temp)
    mean_healthy_temp = find_peaks(mean_healthy_temp)

    to_csv = [np.array(mean_healthy_temp).tolist()]
    print(to_csv)
    with open('mean_healthy_{}.csv'.format(features[_]), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(to_csv)
   # plt.show()
   # plt.plot(mean_healthy_temp)
   # plt.show()
    # 2 STEP: all healthy by mouse and experiment
    all_days = [l_3_days, r_3_days, r_8_days, l_1_day, r_1_day,] #l_2_day, r_2_day]

    mean_all_days = [[] for i in range(len(all_days))]

    for i in range(len(all_days)):
        all_days[i] = all_days[i][np.where(all_days[i] != '')]
        all_days[i] = all_days[i][np.where(all_days[i] != 'PV1')]

        all_days[i] = all_days[i].astype(float).tolist()

        for j in range(period, len(all_days[i]) - period):
            mean_all_days[i].append(np.mean(all_days[i][j - period: j + period]))


    # 3 STEP: all sick by mouse and experiment

    sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
    sheet = sheet.sheet_by_index(0)

    data = np.array(sheet.col_values(ind)[2:])
    data_r = np.array(sheet.col_values(ind + 16)[2:])
    indexes = np.where(data == '')[0].tolist()
    indexes_r = np.where(data_r == '')[0].tolist()
    #print(indexes, indexes_r)
    start_time = datetime.strptime('1/1/2018 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')

    l_8_days = data[1:1057]
    l_8_1_days = data[1058:2119]
    r_8_days = data_r[1058:2119]
    l_3_days = data[2120:]
    r_3_days = data_r[2120:]

    days_PV1 = [l_8_days, l_8_1_days,] #l_3_days]
    days_DIP = [r_8_days,] #r_3_days]

    period = 20
    mean_days_PV1 = [[] for i in range(len(days_PV1))]

    #plt.figure(figsize=(14, 8))
    for i in range(len(days_PV1)):
        days_PV1[i] = days_PV1[i][np.where(days_PV1[i] != '')]
        days_PV1[i] = days_PV1[i][np.where(days_PV1[i] != 'PV1')]

        days_PV1[i] = days_PV1[i].astype(float).tolist()

        for j in range(period, len(days_PV1[i]) - period):
            mean_days_PV1[i].append(np.mean(days_PV1[i][j - period: j + period]))

    #     plt.plot(mean_days[i])
    #
    # plt.show()

    mean_days_DIP = [[] for i in range(len(days_DIP))]

    #plt.figure(figsize=(14, 8))
    for i in range(len(days_DIP)):
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != '')]
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != 'DIP')]
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != 'PV1')]

        days_DIP[i] = days_DIP[i].astype(float).tolist()

        for j in range(period, len(days_DIP[i]) - period):
            mean_days_DIP[i].append(np.mean(days_DIP[i][j - period: j + period]))

    #     plt.plot(mean_days_DIP[i])
    #
    # plt.show()
    #
    mean_PV1_temp = [np.mean([mean_days_PV1[0][i], mean_days_PV1[1][i]]) for i in range(len(mean_days_PV1[0]))]
    mean_PV1_temp = find_peaks(mean_PV1_temp)
    mean_DIP_temp = mean_days_DIP[0]#[np.mean([mean_days[0][i], mean_days[1][i], mean_days[2][i]]) for i in range(len(mean_days[0]))]
    mean_DIP_temp = find_peaks(mean_DIP_temp)
    #
    # plt.figure()
    # plt.plot(mean_PV1_temp)
    # plt.show()

    # 4 STEP: linear regression for healthy mouses relative to average healthy mouse

    for d, day in enumerate([mean_all_days[-1]]):#, mean_all_days[3], mean_all_days[4]]):# +mean_days_PV1 + mean_days_DIP)):

        if d == 0 or d == 1:
            pv_day = mean_days_PV1[0]
        if d == 0:
            pv_day = mean_days_DIP[0]

        day = pv_day[:len(pv_day) // 3]

        pv_day = find_peaks(pv_day, last=False)

        # plt.figure()
        # plt.plot(day)
        # plt.plot(mean_healthy_temp)
        # plt.show()
        if True:#ind == 16:#mean_all_days.index(day) == 2 and ind == 16:
         #   plt.figure()
         #   plt.plot(day)

            day = find_peaks(day, last=False)
            # plt.plot(day)
            # plt.plot(mean_healthy_temp)
            #
            # plt.show()

            day, mean_day = move_day(day, mean_healthy_temp)

            print(len(day))

            # plt.plot(day)
            # plt.plot(mean_day)
            #
            # plt.show()

            mean_day = np.array(mean_day[:len(day)]).reshape(-1, 1)
            reg_day = np.array(day[:len(mean_day)]).reshape(-1, 1)

            reg = LinearRegression(copy_X=True).fit(mean_day, reg_day)

            mean_day = mean_day.reshape(1, -1).tolist()[0]

         #  plt.show()
            a, b = reg.coef_[0][0], reg.intercept_[0]
            print(a, b)

            #if #ind == 16:#mean_all_days.index(day) == 2 and ind == 16:
            # first_9_days = mean_days_PV1[0]#(mean_day * (len(day) // len(mean_healthy_temp) + 1))[:len(day)]
            first_9_days = (mean_day * (len(day + pv_day) // len(mean_day) + 1))[:len(day + pv_day)]

        #    first_9_days = find_peaks(first_9_days)

            # second_9_days = mean_days_PV1[-1]
            # second_9_days = find_peaks(second_9_days)
            # first_9_days, second_9_days = first_9_days[:len(second_9_days)], second_9_days[:len(first_9_days)]
            #
            # plt.figure()
            # plt.plot(first_9_days)
            # plt.plot(second_9_days)


            res = a * np.array(first_9_days) + b

      #       plt.figure()
      #       plt.plot(first_9_days, 'C0')
      # #      plt.plot(day + pv_day, 'C1')
      # #      plt.plot(second_9_days, 'C1')
      #       plt.plot(res, 'C2')

            window = len(day + pv_day) // (64 * 3)

            dists = []
            for i in range(len(day + pv_day) - window):
                cur_res_path = res[i: i + window]
                cur_res_ex = (day + pv_day)[i: i + window]

                new_res_path, __, dist = mine_dtw(cur_res_path, cur_res_ex)

                dists.append(dist)

            # plt.figure()
            #
            # plt.plot(dists)
            # plt.show()

            to_csv = [['dists'] + dists, ['avg_H'] + first_9_days, ['H+PV1_ex'] + day + pv_day, ['reg_res_y=%.2f*x+%.2f'%(a, b)] + res.tolist()]

            to_csv = np.array(to_csv).T.tolist()
      #      to_csv.insert(0, ['dists', 'average_H', 'H+PV1_ex'])


            with open('{}_DTW_DIP_1.csv'.format(features[_]), 'w') as csvfile:
                 writer = csv.writer(csvfile)
                 writer.writerows(to_csv)

                # if i == 0:
                #     plt.figure()
                #     plt.plot(cur_res_first, 'b')
                #     plt.plot(cur_res_path, 'r')
                #     for [map_x, map_y] in _:
                #
                #         plt.plot([map_x, map_y], [cur_res_path[map_x], cur_res_first[map_y]], 'r')
                #
                #     plt.show()

            # new_res, path, dist = mine_dtw(res, first_9_days)
            # path = path

            # for [map_x, map_y] in path:
            #     # print map_x, x[map_x], ":", map_y, y[map_y]
            #
            #     plt.plot([map_x, map_y], [res[map_x], first_9_days[map_y]], 'r')

            # plt.plot(new_res, 'C3')
            # plt.axis([0, len(day + pv_day), 0.95 * np.min([first_9_days, day + pv_day, res]), 1.05 * np.max([first_9_days, day + pv_day, res])])
            # plt.xticks([])
            # plt.show()

            # if d == 0:
            #     to_csv = [['average_healthy', 'R2_example_h+PV1', 'regression', 'TW']]
            #     to_csv += np.array([first_9_days, day + pv_day, res, new_res]).T.tolist()
            #
            #     with open('{}_all_methods_PV1_example.csv'.format(features[ind]), 'w') as csvfile:
            #          writer = csv.writer(csvfile)
            #          writer.writerows(to_csv)

        #
        #
        # # #
        # plt.figure(figsize=(14, 8))
        #
        # plt.plot(mean_healthy_temp[:len(day)])
        # plt.plot(day[:len(mean_healthy_temp)])
        # plt.xticks([])
        # plt.title('temperatures for y = {} * x + {}'.format())
        # plt.plot(res)
        #
        # plt.savefig('{}.png')
        # plt.show()

#        to_csv_avg_temps = [['avg_healthy', 'to_match_healthy_experiment', 'regression_result, a = {}, b = {}'.format(a, b),]]

        # cur_avg_temps = np.array([mean_healthy_temp[:len(day)], day[:len(mean_healthy_temp)], res]).T.tolist()
        #
        # to_csv_avg_temps += cur_avg_temps
#        print(len(to_csv_avg_temps))

        # with open('{}_healthy.csv'.format(mean_all_days.index(day)), 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(to_csv_avg_temps)
    #
    # first_9_days = (mean_healthy_temp * 3)[:len(mean_all_days[2])]# + mean_healthy_temp[:(2 * len(mean_healthy_temp)) // 3]
    # res = a * np.array(first_9_days) + b
    #
    # # plt.figure()
    # plt.plot(mean_healthy_temp)
    # plt.plot(mean_PV1_temp[:len(mean_healthy_temp)])
    # plt.plot(mean_DIP_temp[:len(mean_healthy_temp)])
    # # plt.axis([0, len(mean_healthy_temp), np.min(mean_healthy_temp) - 5, np.max(mean_healthy_temp) + 5])
    # #
    # plt.show()

    # 5 STEP: moved average temp for healthy, PV1, PV1 + DIP

  #   dist = metric(mean_healthy_temp, mean_PV1_temp[:len(mean_healthy_temp)])
  #   best = 0
  #
  #   for i in range(1, (len(mean_PV1_temp) - len(mean_healthy_temp)) // 5):
  #       cur_to_compare = mean_PV1_temp[i: i + len(mean_healthy_temp)]
  #
  #       cur_dist = metric(cur_to_compare, mean_healthy_temp)
  #       if cur_dist < dist:
  #           dist = cur_dist
  #           best = i
  #
  # #  print(best)
  #   mean_PV1_temp_new = mean_PV1_temp[best: best + len(mean_healthy_temp)]
  #
  #   dist = metric(mean_healthy_temp, mean_DIP_temp[:len(mean_healthy_temp)])
  #   best = 0
  #
  #   for i in range(1, (len(mean_DIP_temp) - len(mean_healthy_temp)) // 5):
  #       cur_to_compare = mean_DIP_temp[i: i + len(mean_healthy_temp)]
  #       cur_dist = metric(cur_to_compare, mean_healthy_temp)
  #       if cur_dist < dist:
  #           dist = cur_dist
  #           best = i
  #
  # #  print(best)
  #   mean_DIP_temp_new = mean_DIP_temp[best: best + len(mean_healthy_temp)]
    #
    # if ind == 7:
    #     plt.figure()
    #
    #     plt.subplot(211)
    #     plt.plot(mean_healthy_temp)
    #     plt.plot(mean_PV1_temp[:len(mean_healthy_temp)])
    #     plt.plot(mean_DIP_temp[:len(mean_healthy_temp)])
    #
    #     plt.subplot(212)
    #     plt.plot(mean_healthy_temp)
    #     plt.plot(mean_PV1_temp_new)
    #     plt.plot(mean_DIP_temp_new)
    #
    #     plt.show()

    # to_csv_avg_temps = [['avg_healthy', 'avg_PV', 'avg_PV1+DIP']]
    #
    # avg_temps = np.array([mean_healthy_temp, mean_PV1_temp_new, mean_DIP_temp_new]).T.tolist()
    #
    # to_csv_avg_temps += avg_temps
  #  print(len(to_csv_avg_temps))

    # with open('avg_{}_by_virus.csv'.format(feature), 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(to_csv_avg_temps)
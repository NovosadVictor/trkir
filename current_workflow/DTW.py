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


def metric(plot_1, plot_2):
    return norm(np.array(plot_1) - np.array(plot_2))

features = ['head_body_dist', 'body_speed_x',
            'head_speed_x',
            'head_rotation', 'body_speed', 'head_speed',
            'temp',
            ]

feature_inexes = [1, 2, 16]#4, 7, 8, 9, 16]

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

    days = [l_3_days, r_3_days, r_8_days[:max(len(l_3_days), len(r_3_days))]]

    period = 20
    mean_days = [[] for i in range(len(days))]

    #plt.figure(figsize=(14, 8))
    for i in range(len(days)):
        days[i] = days[i][np.where(days[i] != '')]
        days[i] = days[i][np.where(days[i] != 'PV1')]

        days[i] = days[i].astype(float).tolist()

        for j in range(period, len(days[i]) - period):
            mean_days[i].append(np.mean(days[i][j - period: j + period]))

    mean_healthy_temp = [np.mean([mean_days[0][i], mean_days[1][i], mean_days[2][i]]) for i in range(len(mean_days[0]))]

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


    mean_days_DIP = [[] for i in range(len(days_DIP))]

    #plt.figure(figsize=(14, 8))
    for i in range(len(days_DIP)):
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != '')]
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != 'DIP')]
        days_DIP[i] = days_DIP[i][np.where(days_DIP[i] != 'PV1')]

        days_DIP[i] = days_DIP[i].astype(float).tolist()

        for j in range(period, len(days_DIP[i]) - period):
            mean_days_DIP[i].append(np.mean(days_DIP[i][j - period: j + period]))

    mean_PV1_temp = [np.mean([mean_days_PV1[0][i], mean_days_PV1[1][i]]) for i in range(len(mean_days_PV1[0]))]
    mean_DIP_temp = mean_days_DIP[0]#[np.mean([mean_days[0][i], mean_days[1][i], mean_days[2][i]]) for i in range(len(mean_days[0]))]

    # 5 STEP: moved average temp for healthy, PV1, PV1 + DIP

    #first = [(i, mean_healthy_temp[i]) for i in range(len(mean_healthy_temp))]
    first = mean_healthy_temp[:]
    #second = [(i, mean_PV1_temp[i]) for i in range(len(mean_healthy_temp))]
    second = ((np.array(mean_PV1_temp[:len(mean_healthy_temp)]) - np.min(mean_PV1_temp[:len(mean_healthy_temp)])) / (np.max(mean_PV1_temp[:len(mean_healthy_temp)]) - np.min(mean_PV1_temp[:len(mean_healthy_temp)]))) * (np.max(mean_healthy_temp) - np.min(mean_healthy_temp)) + np.min(mean_healthy_temp)

    dist, path = dtw(first, second, dist=euclidean)
    print(path)

    res = [0 for i in range(len(mean_healthy_temp))]
    for i in range(len(path)):
        if res[path[i][1]] == 0:
            res[path[i][1]] = (path[i][0], path[i][1])

    new_plt = ((np.array([first[val[0]] for val in res]) - np.min(mean_healthy_temp)) / (np.max(mean_healthy_temp) - np.min(mean_healthy_temp))) * (np.max(mean_PV1_temp[:len(mean_healthy_temp)]) - np.min(mean_PV1_temp[:len(mean_healthy_temp)])) + np.min(mean_PV1_temp[:len(mean_healthy_temp)])

    plt.figure(figsize=(14, 8))
    plt.plot(first, 'C0')
    plt.plot(mean_PV1_temp[:len(mean_healthy_temp)], 'C1')
    plt.plot(new_plt, 'C2')
    plt.show()


    # dist = metric(mean_healthy_temp, mean_PV1_temp[:len(mean_healthy_temp)])
    # best = 0
    #
    # for i in range(1, (len(mean_PV1_temp) - len(mean_healthy_temp)) // 5):
    #     cur_to_compare = mean_PV1_temp[i: i + len(mean_healthy_temp)]
    #
    #     cur_dist = metric(cur_to_compare, mean_healthy_temp)
    #     if cur_dist < dist:
    #         dist = cur_dist
    #         best = i
    #
    # print(best)
    # mean_PV1_temp_new = mean_PV1_temp[best: best + len(mean_healthy_temp)]

    # dist = metric(mean_healthy_temp, mean_DIP_temp[:len(mean_healthy_temp)])
    # best = 0
    #
    # for i in range(1, (len(mean_DIP_temp) - len(mean_healthy_temp)) // 5):
    #     cur_to_compare = mean_DIP_temp[i: i + len(mean_healthy_temp)]
    #     cur_dist = metric(cur_to_compare, mean_healthy_temp)
    #     if cur_dist < dist:
    #         dist = cur_dist
    #         best = i
    #
    # mean_DIP_temp_new = mean_DIP_temp[best: best + len(mean_healthy_temp)]
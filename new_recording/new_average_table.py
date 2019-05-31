import os
import csv
import sys
import numpy as np

FEATURES = ['date', 'time', 'head_body_distance',
            'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y',
            'head_body_dist_change', 'head_rotation',
            'body_speed', 'head_speed',
            'body_acceleration',
                               'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

healthy_dir = '{}_{}.csv'.format(sys.argv[1], sys.argv[2])
s_dir = 'new_{}_{}.csv'.format(sys.argv[1], sys.argv[2])
print(healthy_dir, s_dir)

with open(healthy_dir, newline='') as csvfile:
    data = list(csv.reader(csvfile))
print('H loaded')

data = np.array(data[:])
float_data = data[:, 2:].astype(float).T

with open(s_dir, newline='') as csvfile:
    s_data = list(csv.reader(csvfile))
print('S loaded')
s_data = np.array(s_data[:])
s_float_data = s_data[:, 2:].astype(float).T


def moving_avg(chart, N, M):
    moving_aves = []
    for i in range(0, len(chart), M):
        before = i if i < N else N
        after = len(chart) - i if i > len(chart) - N - 1 else N
        moving_aves.append(np.mean(chart[i - before: i + after]))
    return moving_aves


N = int(sys.argv[3])
M = int(sys.argv[4])

mean_data = [[] for i in range(len(float_data))]
for i in range(len(float_data)):
    mean_data[i] = moving_avg(float_data[i], N, M)

len_mean = len(mean_data[0])


mean_data = np.array(mean_data)

s_mean_data = [[] for i in range(len(s_float_data))]
for i in range(len(s_float_data)):
    s_mean_data[i] = moving_avg(s_float_data[i], N, M)

s_len_mean = len(s_mean_data[0])


s_mean_data = np.array(s_mean_data)

dates = [[], []]
dates[0] = data[:, 0][:: M][:len_mean]
dates[1] = data[:, 1][:: M][:len_mean]

s_dates = [[], []]
s_dates[0] = s_data[:, 0][:: M][:s_len_mean]
s_dates[1] = s_data[:, 1][:: M][:s_len_mean]

to_csv = [dates[0], dates[1]]
for i in range(len(s_float_data)):
    if 'norm' in sys.argv:
        print(np.min(mean_data[i]), np.max(mean_data[i]))
        s_mean_data[i] = list((np.array(s_mean_data[i]) - np.min(mean_data[i])) / (np.max(mean_data[i]) - np.min(mean_data[i])))
    to_csv.append(s_mean_data[i])

to_csv = np.array(to_csv).T.tolist()
to_csv = [FEATURES] + to_csv

norm = 'norm' if 'norm' in sys.argv else ''
with open('avg_' + '{}_{}_{}_new_{}_{}.csv'.format(N, M, norm, sys.argv[1], sys.argv[2]), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

to_csv = [dates[0], dates[1]]
for i in range(len(float_data)):
    if 'norm' in sys.argv:
        print(np.min(mean_data[i]), np.max(mean_data[i]))
        mean_data[i] = list((np.array(mean_data[i]) - np.min(mean_data[i])) / (np.max(mean_data[i]) - np.min(mean_data[i])))
    to_csv.append(mean_data[i])

to_csv = np.array(to_csv).T.tolist()
to_csv = [FEATURES] + to_csv

norm = 'norm' if 'norm' in sys.argv else ''
with open('avg_' + '{}_{}_{}_{}_{}.csv'.format(N, M, norm, sys.argv[1], sys.argv[2]), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

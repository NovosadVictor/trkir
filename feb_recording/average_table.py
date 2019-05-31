import os
import csv
import sys
import numpy as np

with open(sys.argv[1], newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = np.array(data[:])
float_data = data[:, 2:].astype(float).T


def moving_avg(chart, N):
    moving_aves = []
    for i in range(0, len(chart), 100):
        before = i if i < N else N
        after = len(chart) - i if i > len(chart) - N - 1 else N
        moving_aves.append(np.mean(chart[i - before: i + after]))
    return moving_aves


mean_data = [[] for i in range(len(float_data))]
for i in range(len(float_data)):
    mean_data[i] = moving_avg(float_data[i], 600)

len_mean = len(mean_data[0])


mean_data = np.array(mean_data)

dates = [[], []]
dates[0] = data[:, 0][:: 100][:len_mean]
dates[1] = data[:, 1][:: 100][:len_mean]

to_csv = [dates[0], dates[1]]
for i in range(len(float_data)):
    to_csv.append(mean_data[i])

to_csv = np.array(to_csv).T

with open('avg_' + sys.argv[1], 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

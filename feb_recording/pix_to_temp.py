import os
import ast
import csv

import numpy as np

dir = '/storage/Data/ucsf_processed'

csvs = os.listdir(dir)
csvs.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[0]),
                         int(x.split('_')[1]), x.split('_')[5],
                         int(x.split('_')[3]) % 12, int(x.split('_')[4])))


conc_csv = []
for csv_f in csvs[:]:
    print(os.path.join(dir, csv_f))
    stat_path = '/storage/Data/ucsf/02_18_19_9_16_AM/{}/Stats/thermal.txt'.format(csv_f[:-4]) if os.path.isfile('/storage/Data/ucsf/02_18_19_9_16_AM/{}/Stats/thermal.txt'.format(csv_f[:-4])) else '/storage/Data/ucsf/02_09_19_3_01_AM/{}/Stats/thermal.txt'.format(csv_f[:-4])

    with open(stat_path, 'r') as t_f:
        t_data = t_f.readlines()

    with open(os.path.join(dir, csv_f), newline='') as csvfile:
        data = list(csv.reader(csvfile))

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = ast.literal_eval(data[i][j])

    for i in range(1, len(data)):
        if [] in data[i]:
            data[i] = data[i - 1]

    for i in range(len(data)):
        temps = t_data[i].split()
        min_temp = float(temps[-2])
        max_temp = float(temps[-1])

        for m in range(4):
            data[i][m][-1] = (min_temp * (255 - data[i][m][-1]) + max_temp * data[i][m][-1]) / 255

        data[i] = temps[:4] + data[i]

    conc_csv += data

with open('/storage/Data/ucsf_concatenated.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(conc_csv)


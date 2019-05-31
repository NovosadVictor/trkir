import numpy as np
from matplotlib import pyplot as plt
import sys
import csv


features = ['head_body_dist', 'body_speed_x',
            'head_speed_x' , 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

data = []
for status in ['Healthy', 'Sick']:
    with open('new{}{}_table.csv'.format(sys.argv[2], status), newline='') as csvfile:
        data += list(csv.reader(csvfile))[1:]

data = np.array(data)

time = int(sys.argv[1])

common_label = data[:, :4].tolist()
for c, line in enumerate(common_label):
    common_label[c] = '_'.join(line)

common_label = np.array(common_label).T.reshape(-1, 1)

uniq = np.unique(common_label).tolist()
uniq.sort(key=lambda x: x.split('_')[1])

data = np.concatenate([common_label, data], axis=1)

split_by_all = [data[np.where(data[:, 0] == ext)] for ext in uniq]

to_csv = []
for d in range(len(split_by_all)):
    day = np.array_split(split_by_all[d], time)
    day = [np.mean(sub_day[:, 5:].astype(float), axis=0) for sub_day in day]
    for sub in day:
        to_csv.append([uniq[d]] + uniq[d].split('_') + sub.tolist())

to_csv.insert(0, ['com_id', 'exp_id', 'virus', 'mouse', 'day'] + features)

with open('avg{}{}_hours_table.csv'.format(sys.argv[2], 24 // time), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)




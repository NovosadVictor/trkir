import csv
import os
import numpy as np


csvs = os.listdir('csvs')

for c_c in csvs:
    print(c_c)
    with open(os.path.join('csvs', c_c), newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))[1:]

    name = c_c.split('.')[0]
    with open(os.path.join('stats', name, 'Stats', 'thermal.txt'), 'r') as s_f:
        info = s_f.readlines()

    temps = [data[:, 4], data[:, 9], data[:, 14], data[:, 19], data[:, 24], data[:, 29], data[:, 34], data[:, 39]]
    temps = np.array(temps).T.astype(float).tolist()

    csv_data = [data[0].tolist()]
    for i in range(len(temps)):
        min_temp, max_temp = float(info[i].split()[-2]), float(info[i].split()[-1])
      #  print(' '.join(info[i].split()[1:4]))

        for t in range(len(temps[i])):
            temps[i][t] = (min_temp * (255 - temps[i][t]) + max_temp * temps[i][t]) / 255

        cur_line = data[i]

        new_line = []
        for _, j in enumerate([0, 5, 10, 15, 20, 25, 30, 35]):
            new_line += [float(cur_line[j + 2][1:-1].split(',')[0]), float(cur_line[j + 2][1:-1].split(',')[1]),
                         float(cur_line[j + 1][1:-1].split(',')[0]), float(cur_line[j + 1][1:-1].split(',')[1]),
                         float(cur_line[j + 3][1:-1].split(',')[0]), float(cur_line[j + 3][1:-1].split(',')[1]),
                         temps[i][_],
                         float(cur_line[j])]

        new_line += ['%d.jpg' % i, ' '.join(info[i].split()[1:4])]

        csv_data.append(new_line)

    with open(os.path.join('csvs', 'new_' + name + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


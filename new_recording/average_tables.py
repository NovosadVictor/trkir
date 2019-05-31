import os
import csv
import numpy as np

xlses = os.listdir('data_csv')
xlses.sort(key=lambda x: (int(x.split('_')[4]), int(x.split('_')[2]), int(x.split('_')[3]),
                          x.split('_')[7].split('.')[0], int(x.split('_')[5]) % 12, int(x.split('_')[6])))
print(xlses)

names = set([xls[1:] for xls in xlses])


for name in names:
    print(name)
    for j in range(4):
        xls_l = str(2 * j) + name
        xls_r = str(2 * j + 1) + name

        with open(os.path.join('data_csv', xls_l), newline='') as csvfile:
            left_data = list(csv.reader(csvfile))

        with open(os.path.join('data_csv', xls_r), newline='') as csvfile:
            right_data = list(csv.reader(csvfile))

        average_data = []

        for i in range(len(left_data)):
            if '0.0' == right_data[i][-4]:
                average_data.append(left_data[i])
            else:
                to_app = np.mean(np.array([left_data[i][:-2], right_data[i][:-2]]).astype(float), axis=0)
                average_data.append(to_app.tolist() + left_data[i][-2:])

        with open(os.path.join('caged_csv', str(j) + name), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(average_data)

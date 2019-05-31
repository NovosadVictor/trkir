import csv
import sys, os
import numpy as np

data_path = sys.argv[2]
with open(os.path.join(data_path, '{}_09_27_18_12_52_AM.csv'.format(sys.argv[1])), newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile))).T.tolist()

with open(os.path.join(data_path, '{}_09_27_18_11_07_PM.csv'.format(sys.argv[1])), newline='') as csvfile:
    new_data = np.array(list(csv.reader(csvfile))).T.tolist()

dates = new_data[-1]

for i in range(len(dates)):
    if '11:55:00 PM' in dates[i]:
        ind = i
        break


for i in range(len(dates)):
    if '09/28/2018 12:52:25 AM' in dates[i]:
        ind_end = i
        break


to_app = dates[ind: ind_end + 2]

for i in range(len(to_app)):
    to_app[i] = to_app[i].replace('/27/', '/26/')
    to_app[i] = to_app[i].replace('/28/', '/27/')

for i in range(len(data) - 1):
    data[i] = new_data[i][ind: ind_end + 2] + data[i]

data[-1] = to_app + data[-1]

with open(os.path.join(data_path, '{}_09_27_18_12_52_AM.csv'.format(sys.argv[1])), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)


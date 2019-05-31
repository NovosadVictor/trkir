import os
import csv
import ast
import numpy as np
from matplotlib import pyplot as plt


temps = []
with open('/storage/Data/ucsf_concatenated.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(data[0])

print('loaded')
for i in range(len(data)):
    cur_temps = []
    print(i)
    for m in range(4):
        data[i][m + 4] = ast.literal_eval(data[i][m + 4])
        cur_temps.append(float(data[i][m + 4][-1]))

    temps.append(cur_temps)

temps = np.array(temps)

plt.figure(figsize=(14, 8))
plt.plot(temps[:, 0])
plt.plot(temps[:, 1])
plt.plot(temps[:, 2])
plt.plot(temps[:, 3])

plt.savefig('./temps.png')

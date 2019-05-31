import os
import sys
import csv
from datetime import datetime, timedelta
import numpy as np

with open(sys.argv[1], newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile))).T
print(data.shape)

dates = data[:2]

correct_time = []

for i in range(len(dates[0])):
    datetime_object = datetime.strptime(dates[0][i] + ' ' + dates[1][i], '%m/%d/%Y %I:%M:%S%p')
    d = datetime_object - timedelta(hours=8)
    correct_time.append(d.strftime('%m/%d/%Y %I:%M %p'))

to_csv = [correct_time]
to_csv += list(data[2:])
to_csv = np.array(to_csv).T

with open('changed_' + sys.argv[1], 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)




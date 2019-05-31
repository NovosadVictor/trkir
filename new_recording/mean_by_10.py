import os
import csv
import numpy as np
from datetime import datetime, timedelta


for j in range(4):
    with open(os.path.join('conc', '%d.csv' % j), newline='') as csvfile:
        data = list(csv.reader(csvfile))

    start_date = datetime.strptime(data[0][-1], '%m/%d/%Y %I:%M:%S %p')
    last_date = datetime.strptime(data[-1][-1], '%m/%d/%Y %I:%M:%S %p')
    print(start_date, last_date)

    total_minutes = last_date - start_date
    total_minutes = total_minutes.days * 24 * 60 + total_minutes.seconds // 60

    by_10 = total_minutes // 10

    print(total_minutes)

    splitted_data = np.array_split(np.array(data), by_10)

    new_data = [np.mean(split[:, :-2].astype(float), axis=0).tolist() + split[0, -2:].tolist() for split in splitted_data]

    with open(os.path.join('by_10_minutes', str(j) + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_data)
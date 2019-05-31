import os
import csv
import numpy as np

xlses = os.listdir('caged_csv')

print(xlses)

names = set([xls[1:] for xls in xlses])

for j in range(4):
    cxlses = [xls for xls in xlses if xls[0] == str(j)]
    cxlses.sort(key=lambda x: (int(x.split('_')[4]), int(x.split('_')[2]), int(x.split('_')[3]),
                              x.split('_')[7].split('.')[0], int(x.split('_')[5]) % 12, int(x.split('_')[6])))

    print(cxlses)
    data = []
    for xls in cxlses:
        with open(os.path.join('caged_csv', xls), newline='') as csvfile:
            data += list(csv.reader(csvfile))

    with open(os.path.join('conc', str(j) + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
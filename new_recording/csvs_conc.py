import csv
import os

csvs = os.listdir('csvs')

subdirnames = set(['_'.join(csv.split('_')[8:14]) for csv in csvs])

print(subdirnames)

for name in subdirnames:
    cur_csvs = [csv for csv in csvs if name in csv[20:]]
    cur_csvs.sort(key=lambda x: int(x.split('_')[-2]))
    print(cur_csvs)

    cur_data = []
    for c_c in cur_csvs:
        with open(os.path.join('csvs', c_c), newline='') as csvfile:
            if cur_data != []:
                cur_data += list(csv.reader(csvfile))[1:]
            else:
                cur_data += list(csv.reader(csvfile))

    with open(os.path.join('csvs', name + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cur_data)
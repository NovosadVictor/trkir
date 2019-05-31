import os, sys
import csv
import numpy as np
import xlrd


os.chdir('/root/Novosad/mouses/')


def concatenate(mouse_n, dirname, subdirname):
    print('Concatenating ', mouse_n, dirname, subdirname)
    if not os.path.isdir(os.path.join('check', dirname, 'xlses', 'concatenated')):
        os.mkdir(os.path.join('check', dirname, 'xlses', 'concatenated'))

    already_csv = [f for f in os.listdir(os.path.join('check', dirname, 'xlses', 'concatenated')) if f[0] == mouse_n]

    data = None

    if len(already_csv) > 0:
        with open(os.path.join('check', dirname, 'xlses', 'concatenated', already_csv[0]), newline='') as csvfile:
            data = list(csv.reader(csvfile))
            if len(data) > 20:
                data = np.array(data).T.tolist()

    csv_name = already_csv[0] if len(already_csv) > 0 else '{}_concatenated.csv'.format(mouse_n)

    with open(os.path.join('check', dirname, 'xlses', 'data', '{}_{}.csv'.format(mouse_n, subdirname)), newline='') as csvfile:
        new_data = list(csv.reader(csvfile))
        if len(new_data) > 20:
            new_data = np.array(new_data).T.tolist()

    # xls = os.path.join('check', dirname, 'xlses', 'data', '{}_{}.xls'.format(mouse_n, subdirname))
    # sheet = xlrd.open_workbook(xls)
    # sheet = sheet.sheet_by_index(0)

 #   print(len(data), len(new_data))
    if data is None:
        print('None')
        data = [[] for i in range(len(new_data))]
        for i in range(len(new_data)):
            data[i] += new_data[i]
    else:
        for i in range(len(new_data)):
            data[i] += new_data[i]

    with open(os.path.join('check', dirname, 'xlses', 'concatenated', csv_name), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    print('Finish concatenating')


if __name__ == '__main__':
    concatenate(sys.argv[1], sys.argv[2], sys.argv[3])






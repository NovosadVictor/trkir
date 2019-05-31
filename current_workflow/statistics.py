import xlrd
import numpy as np

import csv


sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
sheet = sheet.sheet_by_index(0)

info = []
print(sheet.ncols)
for i in range(1, 16 * 2 + 1):
    info.append(sheet.col_values(i)[1:])

dates = sheet.col_values(0)[1:]

splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i + 1] == '']
splits.insert(0, 0)
splits.append(len(dates) - 1)


splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]

to_forget = splitted[1]

splitted = [splitted[0], splitted[2], splitted[3]]

for k in range(len(splitted)):
    splitted[k] = np.array(splitted[k]).T.tolist()
#    splitted[k] = splitted[k]['' not in splitted[k]]
    true_splitted_k = []
    for i in range(len(splitted[k])):
        if '' not in splitted[k][i][:]:
            true_splitted_k.append(splitted[k][i])

    splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()


avgs = [[[0,0] for i in range(len(info))] for j in range(len(splitted))]
for k in range(len(splitted)):

    for j in range(len(info)):
        avgs[k][j][0] = np.mean(splitted[k][j][:len(splitted[k][j]) // 2])
        avgs[k][j][1] = np.mean(splitted[k][j][len(splitted[k][j]) // 2:])

features = ['head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

features = ['experiment_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]

to_forget = np.array(to_forget).T.tolist()
to_forget = to_forget[1:]

true_splitted_k = []
for i in range(len(to_forget)):
    if '' not in to_forget[i][len(to_forget[i]) // 2:]:
        true_splitted_k.append(to_forget[i][len(to_forget[i]) // 2 :])

to_forget = np.array(true_splitted_k).astype(float).T.tolist()

avg_forget = [[0,0] for i in range(len(info) // 2)] + [[np.mean(to_forget[i][:len(to_forget[i]) // 2]), np.mean(to_forget[len(to_forget[i]) // 2:])] for i in range(len(to_forget))]
print(avg_forget)

avgs.insert(1, avg_forget)

to_avgs = [[] for i in range((len(splitted) + 1) * 2)]
for i in range(len(avgs)):
    for j in range(len(avgs[i])):
        to_avgs[2 * i].append(avgs[i][j][0])
        to_avgs[2 * i + 1].append(avgs[i][j][1])

data_to_csv = [features] + [['%d' % (i + 1)] + to_avgs[i] for i in range(len(to_avgs))]

with open('healthy_avgs.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_csv)



sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
sheet = sheet.sheet_by_index(0)

info = []
print(sheet.ncols)
for i in range(1, 16 * 2 + 1):
    info.append(sheet.col_values(i)[2:])

dates = sheet.col_values(0)[2:]

splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i + 1] == '']
splits.insert(0, 0)
splits.append(len(dates) - 2)
print(len(dates))


splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]

to_forget = splitted[0]

splitted = splitted[1:]

for k in range(len(splitted)):
    splitted[k] = np.array(splitted[k]).T.tolist()
#    splitted[k] = splitted[k]['' not in splitted[k]]
    true_splitted_k = []
    for i in range(len(splitted[k])):
        if '' not in splitted[k][i][:]:
            true_splitted_k.append(splitted[k][i])

    splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()


avgs = [[[0, 0] for i in range(len(info))] for j in range(len(splitted))]
for k in range(len(splitted)):

    for j in range(len(info)):
        avgs[k][j][0] = np.mean(splitted[k][j][:len(splitted[k][j]) // 2])
        avgs[k][j][1] = np.mean(splitted[k][j][len(splitted[k][j]) // 2:])

features = ['head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

features = ['experiment_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]

to_forget = np.array(to_forget).T.tolist()

true_splitted_k = []
for i in range(len(to_forget)):
    if '' not in to_forget[i][:len(to_forget[i]) // 2]:
        true_splitted_k.append(to_forget[i][:len(to_forget[i]) // 2 ])

to_forget = np.array(true_splitted_k).astype(float).T.tolist()

avg_forget = [[np.mean(to_forget[i][:len(to_forget[i]) // 2]), np.mean(to_forget[i][len(to_forget[i]) // 2:])] for i in range(len(to_forget))] + [[0, 0] for i in range(len(info) // 2)]
print(avg_forget)

avgs.insert(0, avg_forget)

to_avgs = [[] for i in range((len(splitted) + 1) * 2)]
for i in range(len(avgs)):
    for j in range(len(avgs[i])):
        to_avgs[2 * i].append(avgs[i][j][0])
        to_avgs[2 * i + 1].append(avgs[i][j][1])

data_to_csv = [features] + [['%d' % (i + 1)] + to_avgs[i] for i in range(len(to_avgs))]

with open('sick_avgs.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_csv)
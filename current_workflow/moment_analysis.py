import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
from scipy.stats import pearsonr


features = ['head_body_dist', 'body_speed_x',
            'head_speed_x' , 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

time = int(sys.argv[1])

with open('avg{}{}_hours_table.csv'.format(sys.argv[2], 24 // time), newline='') as csvfile:
    data = list(csv.reader(csvfile))[1:]

data = np.array(data)

common_label = data[:, 2]

uniq = np.unique(common_label).tolist()

split_by_status = [data[np.where(data[:, 2] == ext)] for ext in uniq]


plot_mean = []
plot_std = []
plot_75_q = []
plot_25_q = []

healthy_days = np.unique(split_by_status[0][:, 4])
healthy_by_day = [split_by_status[0][np.where(split_by_status[0][:, 4] == day)]
                  for day in healthy_days]

virus = np.unique(split_by_status[1][:, 3])
sick_by_virus = [split_by_status[1][np.where(split_by_status[1][:, 3] == v)]
                  for v in virus]

for day in healthy_by_day:
    plot_mean.append(np.mean(day[:, 5:].astype(float), axis=0))
    plot_std.append(np.std(day[:, 5:].astype(float), axis=0))
    plot_75_q.append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
    plot_25_q.append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))

for v_ind, v in enumerate(sick_by_virus):
    sick_days = np.unique(v[:, 4])
    virus_by_day = [v[np.where(v[:, 4] == day)] for day in sick_days]
    #
    # plot_mean.append([0] * len(plot_mean[-1]))
    # plot_std.append([0] * len(plot_mean[-1]))
    # plot_75_q.append([0] * len(plot_mean[-1]))
    # plot_25_q.append([0] * len(plot_mean[-1]))
    for day in virus_by_day:
        plot_mean.append(np.mean(day[:, 5:].astype(float), axis=0))
        plot_std.append(np.std(day[:, 5:].astype(float), axis=0))
        plot_75_q.append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
        plot_25_q.append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))

plot_mean = np.array(plot_mean).T
plot_std = np.array(plot_std).T
plot_75_q = np.array(plot_75_q).T
plot_25_q = np.array(plot_25_q).T

to_csv = [['healthy_{}'.format(i + 1) for i in range(8)] + ['PV_{}'.format(i + 1) for i in range(8)] + ['DIP_{}'.format(i + 1) for i in range(8)]]
to_csv += plot_mean.tolist() + plot_std.tolist() + plot_75_q.tolist() + plot_25_q.tolist() + (plot_75_q - plot_25_q).tolist()

to_csv = np.array(to_csv).T.tolist()
to_csv.insert(0, ['status_day'] + ['mean_{}'.format(feature) for feature in features] + ['std_{}'.format(feature) for feature in features] + \
['75q_{}'.format(feature) for feature in features] + ['25q_{}'.format(feature) for feature in features] + ['75-25_{}'.format(feature) for feature in features])

with open('moments{}by_{}_hours.csv'.format(sys.argv[2], 24 // time), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

#
# print(plot_mean[0][-1])
# print(pearsonr(plot_mean[0][-1], plot_mean[1][-1])[0])
# print(pearsonr(plot_mean[1][-1], plot_mean[2][-1])[0])
# print(pearsonr(plot_mean[0][-1], plot_mean[2][-1])[0])

# plt.figure(figsize=(14, 8))
# plt.plot(plot_mean[-1], 'r')
# plt.plot(plot_mean[-1] + 3 * plot_std[-1], 'r')
# #plt.fill_between([i for i in range(len(plot_mean[-1]))], plot_mean[0], plot_mean[0] + 3 * plot_std[0])
# #plt.axis([0, len(plot_mean[-1]) - 1, 5, 25])
# plt.show()





#
# plot_mean = [[], [], []]
# plot_std = [[], [], []]
# plot_75_q = [[], [], []]
# plot_25_q = [[], [], []]
#
# healthy_days = np.unique(split_by_status[0][:, 4])
# healthy_by_day = [split_by_status[0][np.where(split_by_status[0][:, 4] == day)]
#                   for day in healthy_days]
#
# virus = np.unique(split_by_status[1][:, 3])
# sick_by_virus = [split_by_status[1][np.where(split_by_status[1][:, 3] == v)]
#                   for v in virus]
#
# for day in healthy_by_day:
#     plot_mean[0].append(np.mean(day[:, 5:].astype(float), axis=0))
#     plot_std[0].append(np.std(day[:, 5:].astype(float), axis=0))
#     plot_75_q[0].append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
#     plot_25_q[0].append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))
#
# for v_ind, v in enumerate(sick_by_virus):
#     sick_days = np.unique(v[:, 4])
#     virus_by_day = [v[np.where(v[:, 4] == day)] for day in sick_days]
#
#     # plot_mean.append([0] * len(plot_mean[-1]))
#     # plot_std.append([0] * len(plot_mean[-1]))
#     # plot_75_q.append([0] * len(plot_mean[-1]))
#     # plot_25_q.append([0] * len(plot_mean[-1]))
#     for day in virus_by_day:
#         plot_mean[v_ind + 1].append(np.mean(day[:, 5:].astype(float), axis=0))
#         plot_std[v_ind + 1].append(np.std(day[:, 5:].astype(float), axis=0))
#         plot_75_q[v_ind + 1].append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
#         plot_25_q[v_ind + 1].append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))
#
#
#
# for i in range(3):
#     plot_mean[i] = np.array(plot_mean[i]).T
#     plot_std[i] = np.array(plot_std[i]).T
#     plot_75_q[i] = np.array(plot_75_q[i]).T
#     plot_25_q[i] = np.array(plot_25_q[i]).T
#
# print(plot_mean[0][-1])
# print(pearsonr(plot_mean[0][-1], plot_mean[1][-1])[0])
# print(pearsonr(plot_mean[1][-1], plot_mean[2][-1])[0])
# print(pearsonr(plot_mean[0][-1], plot_mean[2][-1])[0])
#
# plt.figure(figsize=(14, 8))
# plt.plot(plot_mean[0][-1].tolist() + plot_mean[1][-1].tolist() + plot_mean[2][-1].tolist(), 'r')
# #plt.plot(plot_mean[1][-1], 'g')
# #plt.plot(plot_mean[2][-1], 'b')
# # plt.plot(plot_mean[-1], 'r')
# # plt.plot(plot_75_q[-1], 'g')
# # plt.plot(plot_25_q[-1], 'b')
# #plt.plot(plot_75_q[-1] - plot_25_q[-1])
# #plt.plot(plot_mean[-1] + 3 * plot_std[-1], 'C3')
# #plt.plot(plot_std[-1], 'C4')
# #plt.fill_between([i for i in range(len(plot_mean[-1]))], plot_mean[0], plot_mean[0] + 3 * plot_std[0])
# #plt.axis([0, len(plot_mean[-1]) - 1, 5, 25])
# plt.show()







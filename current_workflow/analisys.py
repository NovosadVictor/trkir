import csv

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA

features = ['date', 'head_body_dist', 'body_speed_x', 'body_speed_y',
            'head_speed_x', 'head_speed_y', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

with open('all_by_10_minutes.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = data[139:1200]
data = np.array(data)

data = data.T.tolist()
dates = data[0][:]

data[3] = np.array(data[3])
data[5] = np.array(data[5])
data[19] = np.array(data[19])
data[21] = np.array(data[21])

data[3][np.where(data[3] == 'nan')] = np.max(data[3][np.where(data[3] != 'nan')][1:].astype(float))
data[5][np.where(data[5] == 'nan')] = np.max(data[5][np.where(data[5] != 'nan')][1:].astype(float))
data[19][np.where(data[19] == 'nan')] = np.max(data[5][np.where(data[19] != 'nan')][1:].astype(float))
data[21][np.where(data[21] == 'nan')] = np.max(data[5][np.where(data[21] != 'nan')][1:].astype(float))

to_del = [3, 5, 6, 10, 11, 12, 13, 14, 15]
to_del += [i + 16 for i in to_del]

print(to_del)
data = [data[i] for i in range(len(data)) if i not in to_del]

for i in range(1, len(data)):
    data[i] = np.array(data[i]).astype(float).tolist()
    data[i] = (np.array(data[i]) - np.mean(data[i])) / (np.max(data[i]) - np.min(data[i]))

data = np.array(data).T

days = ['27', '28', '29', '30', '01', '02', '03', '04', '05']

colors = [['b', 'r'],
          ['g', 'k'],
          ['c', 'y'],
          ['aqua', 'coral'],
          ['cadetblue', 'grey'],
          ['indigo', 'brown'],
          ['slateblue', 'lightgray'],
          ['darkcyan', 'yellow'],
          ['navy', 'pink']]

to_csv = []

left_mouse = data[:, 1: 8].astype(float).tolist()
right_mouse = data[:, 8:].astype(float).tolist()
print(left_mouse[0], right_mouse[0])

# pca_1 = PCA(n_components=3)
# pca_1.fit(left_mouse)
#
# Xt_1 = pca_1.transform(left_mouse)
# ev_1 = pca_1.explained_variance_ratio_
# print(ev_1)
#
#
# pca_2 = PCA(n_components=3)
# pca_2.fit(right_mouse)
#
# Xt_2 = pca_2.transform(right_mouse)
# ev_2 = pca_2.explained_variance_ratio_
# print(ev_2)
#
# to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100), 'L_PC3 %d'  % int(ev_1[2] * 100),
#                'R_PC1 %d' % int(ev_2[0] * 100), 'R_PC2 %d' % int(ev_2[1] * 100), 'R_PC3 %d' % int(ev_2[2] * 100)])
#
#
# for k in range(Xt_1.shape[0]):
#     to_csv.append([dates[k], Xt_1[k, 0], Xt_1[k, 1], Xt_1[k, 2], Xt_2[k, 0], Xt_2[k, 1], Xt_2[k, 2]])
# #
pca_1 = PCA(n_components=3)
pca_1.fit(left_mouse + right_mouse)

Xt_1 = pca_1.transform(left_mouse + right_mouse)
ev_1 = pca_1.explained_variance_ratio_
print(ev_1)

to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100), 'L_PC3 %d'  % int(ev_1[2] * 100)])

for k in range(Xt_1.shape[0] // 2):
    to_csv.append([dates[k], Xt_1[k, 0], Xt_1[k, 1], Xt_1[k, 2]])
for k in range(Xt_1.shape[0] // 2, Xt_1.shape[0]):
    to_csv.append([dates[k - Xt_1.shape[0] // 2], Xt_1[k, 0], Xt_1[k, 1], Xt_1[k, 2]])
#
# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# patches = [mpatches.Patch(color=colors[i][0], label='l {}'.format(days[i])) for i in range(len(days))]
# patches += [mpatches.Patch(color=colors[i][1], label='r {}'.format(days[i])) for i in range(len(days))]
#
# for k in range(Xt.shape[0] // 2):
#
#     to_csv.append([dates[k], Xt[k, 0], Xt[k, 1], Xt[k, 2]])
#     for day in days:
#         if day == dates[k].split()[0].split('/')[1]:
#             text_day = day
#             break
#
#     ax.scatter(Xt[k, 0], Xt[k, 1], Xt[k, 2],
#                 color=colors[days.index(day)][0])
#
# for k in range(Xt.shape[0] // 2, Xt.shape[0]):
#     to_csv.append([dates[k - Xt.shape[0]], Xt[k, 0], Xt[k, 1], Xt[k, 2]])
#     for day in days:
#         if day == dates[k - Xt.shape[0] // 2].split()[0].split('/')[1]:
#             text_day = day
#             break
#
#     ax.scatter(Xt[k, 0], Xt[k, 1], Xt[k, 2],
#                color=colors[days.index(day)][1])
#
# ax.legend(fontsize=12, loc="lower right", handles=patches)
# plt.title('PCA for all days,\nleft vs right mouse')
#
# #plt.show()

with open('pca_table_wo_some_features.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

# for k in range(Xt.shape[0] // 2):
#   #  print(k)
#
#
#     for day in days:
#       #  print(dates[k].split()[0].split('/')[1])
#         if day == dates[k].split()[0].split('/')[1]:
#             text_day = day
#         #    print(0, text_day)
#             break
#
#     #plt.scatter(Xt[k, 0], Xt[k, 1], c=colors[days.index(day)][0])
#     ax.scatter(Xt[k, 0], Xt[k, 1], Xt[k, 2],
#   color=colors[days.index(day)][0])
#   #  if k % (24 * 6 + 1) == 0:
#   #      plt.text(1.01 * Xt[k, 0], 1.01 * Xt[k, 1], str(day))
# patches = [mpatches.Patch(color=colors[i][0], label='l {}'.format(days[i])) for i in range(len(days))]
# ax.legend(fontsize=12, loc="lower right", handles=patches)
# plt.title('PCA for all days,\nleft mouse')
#
# plt.show()
#
#
# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')
# for k in range(Xt.shape[0] // 2, Xt.shape[0]):
#     for day in days:
#         if day == dates[k - Xt.shape[0] // 2].split()[0].split('/')[1]:
#             text_day = day
#             break
#  #   plt.scatter(Xt[k, 0], Xt[k, 1],
#  #                   c=colors[days.index(day)][1])
#
#     ax.scatter(Xt[k, 0], Xt[k, 1], Xt[k, 2],
#                color=colors[days.index(day)][1])
#
# patches = [mpatches.Patch(color=colors[i][1], label='r {}'.format(days[i])) for i in range(len(days))]
# ax.legend(fontsize=12, loc="lower right", handles=patches)
#    # plt.text(1.01 * Xt[k, 0], 1.01 * Xt[k, 1], str(k - Xt.shape[0] // 2 + day_bias))
#
# plt.title('PCA for all days,\nright mouse')
#
# print('show')
# plt.show()


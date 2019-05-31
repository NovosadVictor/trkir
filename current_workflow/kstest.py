from scipy.stats import ks_2samp
import numpy as np
import sys
import csv
from scipy.stats import pearsonr

from sklearn.decomposition import PCA


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
#
# for day in healthy_by_day:
#     plot_mean.append(np.mean(day[:, 5:].astype(float), axis=0))
#     plot_std.append(np.std(day[:, 5:].astype(float), axis=0))
#     plot_75_q.append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
#     plot_25_q.append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))

kstest = [[], []]
for v_ind, v in enumerate(sick_by_virus):
    sick_days = np.unique(v[:, 4])
    virus_by_day = [v[np.where(v[:, 4] == day)] for day in sick_days]


    for v_1, v_day in enumerate(virus_by_day):
        kstest[v_ind].append([])
        v_day = v_day[:, 5:].astype(float).T.tolist()
        for h_1, h_day in enumerate(healthy_by_day):
            h_day = h_day[:, 5:].astype(float).T.tolist()

            kstest[v_ind][-1].append(1 - list(ks_2samp(h_day[-1], v_day[-1]))[-1])

print(kstest)

          #  for i in range(len(v_day)):
          #      print(list(ks_2samp(h_day[i], v_day[i]))[-1])


    # for day in virus_by_day:
    #     plot_mean.append(np.mean(day[:, 5:].astype(float), axis=0))
    #     plot_std.append(np.std(day[:, 5:].astype(float), axis=0))
    #     plot_75_q.append(np.quantile(day[:, 5:].astype(float), 0.75, axis=0))
    #     plot_25_q.append(np.quantile(day[:, 5:].astype(float), 0.25, axis=0))
#
# plot_mean = np.array(plot_mean)
# plot_std = np.array(plot_std)
# plot_75_q = np.array(plot_75_q)
# plot_25_q = np.array(plot_25_q)
#
# Xt = {}
# ev = {}
#
# ff = {'mean': plot_mean, 'std': plot_std, '75q-25q': plot_75_q - plot_25_q}
#
# for f in ['mean', 'std', '75q-25q']:
#     print(f)
#     pca = PCA(n_components=2)
#     pca.fit(ff[f])
#
#     Xt[f] = pca.transform(ff[f]).tolist()
#     ev[f] = pca.explained_variance_ratio_
#
#
# Xt = np.concatenate([Xt['mean'], Xt['std'], Xt['75q-25q']], axis=1).tolist()
# print(Xt)
# to_csv = [['{}_PC1_{:.2f}'.format('mean', ev['mean'][0]), '{}_PC2_{:.2f}'.format('mean', ev['mean'][1]),
#            '{}_PC2_{:.2f}'.format('std', ev['std'][0]), '{}_PC2_{:.2f}'.format('std', ev['std'][1]),
#            '{}_PC2_{:.2f}'.format('75q-25q', ev['75q-25q'][0]), '{}_PC2_{:.2f}'.format('75q-25q', ev['75q-25q'][1])]]
# to_csv += Xt
# to_csv = np.array(to_csv).T.tolist()
# to_csv.insert(0, [''] + ['healthy_{}'.format(i + 1) for i in range(8)] + ['PV_{}'.format(i + 1) for i in range(8)] + ['DIP_{}'.format(i + 1) for i in range(8)])
#
# to_csv = np.array(to_csv).T.tolist()
#
# with open('pca{}.csv'.format(sys.argv[2]), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)

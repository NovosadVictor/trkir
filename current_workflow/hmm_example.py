import csv
import numpy as np
import sys
import itertools
from matplotlib import pyplot as plt
from hmmlearn.hmm import GaussianHMM as HMM
import matplotlib.cm as cm
from datetime import datetime, timedelta

np.random.seed(1)
features = ['head_body_dist', 'body_speed_x',
            'head_speed_x', 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

needed_features = ['temp']
#needed_features = ['head_body_dist', 'body_speed_x', 'head_rotation', 'body_speed', 'temp']
#needed_features = ['head_body_dist', 'head_rotation', 'temp']
feature_indexes = [features.index(feature) for feature in needed_features]

np.random.seed(1)


def load_mean_healthy(f):
    with open('mean_healthy_{}.csv'.format(f), newline='') as csvfile:
        data_h = np.array(list(csv.reader(csvfile))).astype(float).tolist()

    return data_h[0]


def moving_avg(chart, N):
    moving_aves = []

    for i in range(len(chart)):
        before = i if i < N else N
        after = len(chart) - i if i > len(chart) - N - 1 else N
        moving_aves.append(np.mean(chart[i - before: i + after]))

    return moving_aves


def match_range(to_match, part, plot):
    max = np.max(plot[:int(part * len(to_match))])
    min = np.min(plot[:int(part * len(to_match))])
    mean = np.mean(plot[:int(part * len(to_match))])
    max_2 = np.max(to_match[:int(part * len(to_match))])
    min_2 = np.min(to_match[:int(part * len(to_match))])
    mean_2 = np.mean(to_match[:int(part * len(to_match))])

    print(min, max, mean)
    #   print(min_2, max_2, mean_2)
    #    max_to_match = np.max(to_match[:int(part * len(to_match))])

    before_scale = (np.array(to_match) - min_2) / (max_2 - min_2)
    print(np.max(before_scale[:int(part * len(to_match))]), np.min(before_scale[:int(part * len(to_match))]))
    after_scale = (np.array(before_scale) * (max - min) + min).tolist()

    print(np.min(after_scale[:int(part * len(to_match))]), np.max(after_scale[:int(part * len(to_match))]),
          np.mean(after_scale[:int(part * len(to_match))]))
    return after_scale  # np.array(to_match) + (max - max_to_match)


def scale_mean(to_scale):
    return ((np.array(to_scale) - np.mean(to_scale)) / (np.max(to_scale) - np.min(to_scale))).tolist()


with open('new_Healthy_table.csv', newline='') as csvfile:
    data_h = list(csv.reader(csvfile))[1:]

data_h = np.array(data_h)

left_first_h = data_h[np.where(data_h[:, 0] == '0')]
left_first_h = left_first_h[np.where(left_first_h[:, 2] == 'L')]

left_first_h = left_first_h[:, 4:].astype(float)

right_first_h = data_h[np.where(data_h[:, 0] == '1')]
right_first_h = right_first_h[np.where(right_first_h[:, 2] == 'R')]

right_first_h = right_first_h[:, 4:].astype(float)

with open('new_Sick_table.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))[1:]

data = np.array(data)

# LEFT MOUSE FIRST EXPERIMENT

left_first = data[np.where(data[:, 0] == '0')]
right_first = data[np.where(data[:, 0] == '1')]
left_second = data[np.where(data[:, 0] == '1')]

left_first = left_first[np.where(left_first[:, 2] == 'L')]
right_first = right_first[np.where(right_first[:, 2] == 'R')]
left_second = left_second[np.where(left_second[:, 2] == 'L')]

left_first = left_first[:, 4:].astype(float)
right_first = right_first[:, 4:].astype(float)
left_second = left_second[:, 4:].astype(float)

to_hmm_l_f = [0] * len(needed_features)
to_hmm_l_s = [0] * len(needed_features)
to_hmm_r_f = [0] * len(needed_features)

to_hmm_h = [0] * len(needed_features)
to_hmm_r_h = [0] * len(needed_features)
mean_healthy = [0] * len(needed_features)

windowed_data = [[], [], [], [], [], []]

fft_already_exists = False

for f_ind, f in enumerate(feature_indexes):
    smooth = 4

    to_hmm_l_f[f_ind] = moving_avg(left_first[:, f].tolist(), smooth)
    to_hmm_l_s[f_ind] = moving_avg(left_second[:, f].tolist(), smooth)
    to_hmm_r_f[f_ind] = moving_avg(right_first[:, f].tolist(), smooth)

    to_hmm_h[f_ind] = moving_avg(left_first_h[:, f].tolist(), smooth)
    to_hmm_r_h[f_ind] = moving_avg(right_first_h[:, f].tolist(), smooth)

    mean_healthy[f_ind] = (load_mean_healthy(needed_features[f_ind]) * 10)[
                          :len(to_hmm_l_f[f_ind]) + len(to_hmm_h[f_ind])]

    to_hmm_r_f[f_ind][150:175] = np.linspace(to_hmm_r_f[f_ind][150], to_hmm_r_f[f_ind][175], 25).tolist()

    to_hmm_l_f[f_ind] = match_range(to_hmm_l_f[f_ind], 0.4, to_hmm_h[f_ind])
    to_hmm_l_s[f_ind] = match_range(to_hmm_l_s[f_ind], 0.4, to_hmm_l_f[f_ind])
    to_hmm_r_f[f_ind] = match_range(to_hmm_r_f[f_ind], 0.4, to_hmm_l_f[f_ind])
    mean_healthy[f_ind] = match_range(mean_healthy[f_ind], 0.4, to_hmm_l_f[f_ind])

    # if f_ind == 0:
    #     plt.figure()
    #     plt.plot(to_hmm_l_f[0])
    #     plt.plot(to_hmm_r_f[0])
    #     plt.show()

    # plt.figure()
    # plt.plot(to_hmm_l_f)
    # plt.plot(to_hmm_h)
    # # to_hmm_l_f = moving_avg(to_hmm_l_f, 10)
    # # to_hmm_l_s = moving_avg(to_hmm_l_s, 10)
    # # to_hmm_r_f = moving_avg(to_hmm_r_f, 10)
    # # to_hmm_r_f = moving_avg(to_hmm_r_f, 10)
    # # to_hmm_h = moving_avg(to_hmm_h, 10)
    # #
    # # plt.plot(to_hmm_l_f)
    # # plt.plot(to_hmm_h)
    #
    # plt.show()

    # to_hmm_l_f[0] = scale_mean(to_hmm_l_f[0])
    # to_hmm_l_s[0] = scale_mean(to_hmm_l_s[0])
    # to_hmm_r_f[0] = scale_mean(to_hmm_r_f[0])
    #
    # to_hmm_h[0] = scale_mean(to_hmm_h[0][1])

    #
    # to_h_fft = []
    # to_pv_fft = []
    # to_dip_fft = []
    #
    # if 'h' in sys.argv:
    #     for i in range(3):
    #         cur_part = to_hmm_h[f_ind][i * (len(to_hmm_h[f_ind]) // 3): (i + 1) * (len(to_hmm_h[f_ind]) // 3)]
    #         wo_abs = np.fft.fft(cur_part)
    #         min = np.argmin(wo_abs[:])
    #         min = wo_abs[min]
    #
    #         cur_fft = np.abs(wo_abs)[0: 20]
    #
    #         freqs = np.sort(np.argsort(cur_fft)[-7:])
    #         # for sf in range(len(wo_abs)):
    #         #     if sf not in freqs:
    #         #         wo_abs[sf] = min
    #         # print(freqs)
    #
    #
    #         iyf = np.fft.ifft(wo_abs[:freqs[-1]], n=len(wo_abs))
    #         to_h_fft += iyf[1:].tolist()
    #
    #         # plt.figure()
    #         # plt.plot(cur_part[1:])
    #         # plt.plot(iyf[1:])
    #         #
    #         # plt.show()
    #
    # if 'l_f' in sys.argv:
    #     for i in range(8):
    #         cur_part = to_hmm_l_f[f_ind][i * (len(to_hmm_l_f[f_ind]) // 8): (i + 1) * (len(to_hmm_l_f[f_ind]) // 8)]
    #         wo_abs = np.fft.fft(cur_part)
    #         min = np.argmin(wo_abs[:])
    #         min = wo_abs[min]
    #
    #         cur_fft = np.abs(wo_abs)[0: 20]
    #
    #         freqs = np.sort(np.argsort(cur_fft)[-7:])
    #         # for sf in range(len(wo_abs)):
    #         #     if sf not in freqs:
    #         #         wo_abs[sf] = min
    #         # print(freqs)
    #
    #
    #         iyf = np.fft.ifft(wo_abs[:freqs[-1]], n=len(wo_abs))
    #
    #         to_pv_fft += iyf[1:].tolist()
    #
    #         # plt.figure()
    #         # plt.plot(cur_part[1:])
    #         # plt.plot(iyf[1:])
    #         #
    #         # plt.show()
    #
    # if 'r_f' in sys.argv:
    #     for i in range(8):
    #         cur_part = to_hmm_r_f[f_ind][i * (len(to_hmm_r_f[f_ind]) // 8): (i + 1) * (len(to_hmm_r_f[f_ind]) // 8)]
    #         wo_abs = np.fft.fft(cur_part)
    #         min = np.argmin(wo_abs[:])
    #         min = wo_abs[min]
    #
    #         cur_fft = np.abs(wo_abs)[0: 20]
    #
    #         freqs = np.sort(np.argsort(cur_fft)[-7:])
    #         # for sf in range(len(wo_abs)):
    #         #     if sf not in freqs:
    #         #         wo_abs[sf] = min
    #         # print(freqs)
    #
    #         iyf = np.fft.ifft(wo_abs[:freqs[-1]], n=len(wo_abs))
    #         to_dip_fft += iyf[1:].tolist()
    #
    # # show = 0
    # # if 'l_f' in sys.argv:
    # #     show = 1
    # # if 'r_f' in sys.argv:
    # #     show = 2
    # #
    # # show_ifft = [to_h_fft, to_pv_fft, to_dip_fft][show]
    # # show_plot = [to_hmm_h[f_ind], to_hmm_l_f[f_ind], to_hmm_r_f[f_ind]][show]
    # # plt.figure()
    # # plt.plot(show_plot)
    # # plt.plot(show_ifft)
    # #
    # # plt.show()
    #
    # to_pv_fft_b = (np.array(to_h_fft) + 7).tolist()
    # to_pv_fft = (np.array(to_pv_fft) + 7).tolist()
    # to_pv_fft = to_pv_fft_b + to_pv_fft
    # to_h_plot = (np.array(to_hmm_h[f_ind]) + 7).tolist()
    # to_pv_plot = (np.array(to_hmm_l_f[f_ind]) + 7).tolist()
    # to_pv_plot = to_h_plot + to_pv_plot
    #
    # to_dip_fft = to_h_fft + to_dip_fft
    # to_dip_plot = to_hmm_h[f_ind] + to_hmm_r_f[f_ind]
    #
    # to_pv_fft = np.real(to_pv_fft).tolist()
    # to_dip_fft = np.real(to_dip_fft).tolist()

    for fft_num, to_fft in enumerate(
            [to_hmm_l_f[f_ind], to_hmm_l_s[f_ind], to_hmm_r_f[f_ind], to_hmm_h[f_ind], to_hmm_r_h[f_ind],
             mean_healthy[f_ind]]):
        window_size = 140

        for i in range(len(to_fft)):

            part = float(i) / (len(to_fft) - 1)
            to_left = int(window_size * part)
            to_right = window_size - to_left

          #  if 300 < i < 600:
                #     plt.figure()
            #     plt.plot(to_fft[i - to_left: i + to_right])
            #     plt.savefig('january_hmm_plots/trash/{}.png'.format(i))
            #     plt.close()

            cur_fft = np.abs(np.fft.fft(to_fft[i - to_left: i + to_right]))[1: window_size // 2]

            if not fft_already_exists:
                windowed_data[fft_num].append([  # np.mean(to_fft[i: i + window_size]),
                                                  np.subtract(
                                                      *np.percentile(to_fft[i - to_left: i + to_right], [75, 25]))] +
                                              np.sort(np.argsort(cur_fft[:])[-4:-1]).tolist())
            else:
                windowed_data[fft_num][i] += [  # np.mean(to_fft[i: i + window_size]),
                                                  np.subtract(
                                                      *np.percentile(to_fft[i - to_left: i + to_right], [75, 25]))] +\
                                              np.sort(np.argsort(cur_fft[:])[-4:-1]).tolist()

    fft_already_exists = True

for f in range(len(needed_features)):
    to_hmm_l_f[f] = moving_avg(to_hmm_l_f[f], 10)
    to_hmm_l_s[f] = moving_avg(to_hmm_l_s[f], 10)
    to_hmm_r_f[f] = moving_avg(to_hmm_r_f[f], 10)
    to_hmm_h[f] = moving_avg(to_hmm_h[f], 10)
    to_hmm_r_h[f] = moving_avg(to_hmm_r_h[f], 10)
    mean_healthy[f] = moving_avg(mean_healthy[f], 10)

h_pv = [to_hmm_h[f] + to_hmm_l_f[f] for f in range(len(needed_features))]
h_dip = [to_hmm_h[f] + to_hmm_r_f[f] for f in range(len(needed_features))]

for n_comp in range(5, 10):
#    n_comp = int(sys.argv[-1])
    hmm = HMM(n_components=n_comp)
    hmm.fit(windowed_data[3] + windowed_data[0])

    means = np.around(hmm.means_[:], 3)
    means_2 = np.around(means[:])
    means_2[:, ::6] = means[:, ::6]
    #print(means_2.tolist())
    # print(np.around(hmm.transmat_, 3))

    preds_l_f = hmm.predict(windowed_data[0])
    preds_l_s = hmm.predict(windowed_data[1])
    preds_r_f = hmm.predict(windowed_data[2])
    preds_h = hmm.predict(windowed_data[3])
    preds_r_h = hmm.predict(windowed_data[4])
    preds_m_h = hmm.predict(windowed_data[5])

    conc = hmm.predict(windowed_data[3] + windowed_data[0])
    conc_dip = hmm.predict(windowed_data[3] + windowed_data[2])

    colors = cm.rainbow(np.linspace(0, 1, n_comp))

    start_date = datetime.strptime('12:00 AM', '%I:%M %p')

    # print(len(windowed_data[3]) + len(windowed_data[0]), len(to_pv_fft))
    for f in range(len(needed_features)):
        if f < len(needed_features) - 1:
            continue
        # for i in range(7):
        #     plt.plot(i, 1, color=colors[i], marker='.', markersize=10)
        # plt.show()
    #     pv_h_times = [(start_date + timedelta(minutes=i * (1440 / (len(windowed_data[3]) / 3)))).strftime('%I:%M %p') for i
    #                   in range(len(windowed_data[3]))]
    #     pv_h_times += [(start_date + timedelta(minutes=i * (1440 / (len(windowed_data[0]) / 8)))).strftime('%I:%M %p') for i
    #                    in range(len(windowed_data[0]))]
    #
    #     plt.figure(figsize=(14, 8))
    #
    #     for i in range(len(preds_m_h)):
    #         plt.plot(i, mean_healthy[f][i], color=colors[preds_m_h[i]], marker='.', markersize=10)
    #
    #     plt.xticks([i for i in range(len(pv_h_times))][::len(pv_h_times) // 22], pv_h_times[::len(pv_h_times) // 22], rotation=60)
    #
    #     plt.title('average healthy {}'.format(needed_features[f]))
    #     plt.ylim([0.9 * np.min(mean_healthy[f]), 1.1 * np.max(mean_healthy[f])])
    #     plt.show()
    # #    plt.savefig('hmm_last/hmm_by_{}_ma_avg_healthy.png'.format(needed_features[f]))
    #
        # plt.figure(figsize=(14, 8))
        # for i in range(len(h_pv[f])):#len(windowed_data[3]) + len(windowed_data[0])):
        #     plt.plot(i, h_pv[f][i] + 1.1 * (np.max(h_dip[f]) - np.min(h_pv[f])), color=colors[conc[i]], marker='.', markersize=10)
    #
    #     plt.figure(figsize=(14, 8))
    #     plt.subplot(211)
    #     for i in range(len(windowed_data[3] + windowed_data[0])):#len(windowed_data[3]) + len(windowed_data[0])):
    #         plt.plot(i, (windowed_data[3] + windowed_data[0])[i][0], color=colors[conc[i]], marker='.', markersize=10)
    #
    #     plt.title('h + pv / dip {} intq'.format(needed_features[f]))
    #
    # #    plt.annotate('H+PV1', xy=[20, 1], xytext=[25, 1.2])
    # #
    # #    plt.xticks([i for i in range(len(pv_h_times))][::len(pv_h_times) // 22], pv_h_times[::len(pv_h_times) // 22], rotation=60)
    #   #  plt.savefig('hmm/h_pv_{}.png'.format(needed_features[f]))
    #
    # #    dip_times = [(start_date + timedelta(minutes = i * (1440 / (len(windowed_data[2]) / 8)))).strftime('%I:%M %p') for i in range(len(windowed_data[2]))]
    #
    # #   plt.figure(figsize=(14, 8))
    # #     for i in range(len(h_dip[f])):#len(windowed_data[3]) + len(windowed_data[2])):
    # #         plt.plot(i, h_dip[f][i], color=colors[conc_dip[i]], marker='.', markersize=10)
    #     plt.subplot(212)
    #
    #     for i in range(len(windowed_data[3] + windowed_data[2])):#len(windowed_data[3]) + len(windowed_data[2])):
    #         plt.plot(i, (windowed_data[3] + windowed_data[2])[i][0], color=colors[conc_dip[i]], marker='.', markersize=10)

     #   plt.show()
  #      plt.savefig('january_hmm_plots/5_freqs/hmm_{}_intq_temp.png'.format(n_comp))
  #      plt.close()
  #

        all_possible_outcomes = list(itertools.product(np.linspace(0.2, 2, 18),
                                              np.linspace(0, 9, 10),
                                              np.linspace(0, 9, 10),
                                              np.linspace(0, 9, 10),
                                              ))

        most_probable_outcomes = []

        for i in range(140):
            print(i)
            previous_data = (windowed_data[3] + windowed_data[0])[200:][i - 70: i]

            outcome_score = []
            for possible_outcome in all_possible_outcomes:
                outcome_score.append(np.exp(hmm.score(previous_data + [possible_outcome])))

            most_probable_outcome = all_possible_outcomes[np.argmax(
                outcome_score)]

            print(most_probable_outcome)
            most_probable_outcomes.append(most_probable_outcome)

        cur_pred_states = hmm.predict(most_probable_outcomes)

        plt.figure()
        for i in range(len(most_probable_outcomes)):  # len(windowed_data[3]) + len(windowed_data[2])):
            plt.plot(i, most_probable_outcomes[i][0], color=colors[cur_pred_states[i]], marker='.', markersize=10)
        for i in range(len(windowed_data[3] + windowed_data[0])):  # len(windowed_data[3]) + len(windowed_data[2])):
            plt.plot(i, (windowed_data[3] + windowed_data[0])[i][0], color=colors[conc[i]], marker='.', markersize=10)
        plt.show()


# plt.figure(figsize=(14, 8))
# for i in range(len(h_pv[f])):  # len(windowed_data[3]) + len(windowed_data[0])):
#     plt.plot(i, h_pv[f][i] + 5, color=colors[conc[i]], marker='.', markersize=10)
# plt.title('h + pv + dip {}'.format(needed_features[f]))
#
# #   plt.annotate('H+PV1', xy=[20, 1], xytext=[25, 1.2])
#
# plt.xticks([i for i in range(len(pv_h_times))][::len(pv_h_times) // 22], pv_h_times[::len(pv_h_times) // 22],
#            rotation=60)
# #  plt.savefig('hmm/h_pv_{}.png'.format(needed_features[f]))
#
# dip_times = [(start_date + timedelta(minutes=i * (1440 / (len(windowed_data[2]) / 8)))).strftime('%I:%M %p') for i
#              in range(len(windowed_data[2]))]
#
# #    plt.figure(figsize=(14, 8))
# for i in range(len(h_dip[f])):  # len(windowed_data[3]) + len(windowed_data[2])):
#     plt.plot(i, h_dip[f][i], color=colors[conc_dip[i]], marker='.', markersize=10)
#
# plt.savefig('hmm/{}_all.png'.format(needed_features[f]))
#    plt.annotate('H+DIP', xy=[20, 0.8], xytext=[25, 0.6])

#    plt.title('DIP {}'.format(needed_features[f]))

#    plt.xticks([i for i in range(len(windowed_data[2]))][::len(windowed_data[2]) // 16], dip_times[::len(windowed_data[2]) // 16], rotation=60)

# plt.ylim([0.4, 1.3])

# print(len(windowed_data[2]), len(windowed_data[0]))
# plt.savefig('hmm/DIP_{}.png'.format(needed_features[f]))

# freqs_labels = []
# for fea in needed_features:
#     freqs_labels += ['{}_inq'.format(fea), '{}_freq_0'.format(fea), '{}_freq_1'.format(fea), '{}_freq_2'.format(fea),
#                      '{}_freq_3'.format(fea), '{}_freq_4'.format(fea)]
#
#
# to_csv = [['id', 'H+PV1_{}'.format(needed_features[f]), 'state'] + freqs_labels]
#
# for i in range(len(windowed_data[3]) + len(windowed_data[0])):
#     to_csv.append([pv_h_times[i], h_pv[f][i], conc[i]] + (windowed_data[3] + windowed_data[0])[i])
#
# with open('hmm_last/HMM_state_fft_H_PV1_{}.csv'.format(needed_features[f]), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#
# to_csv = [['id', 'DIP_{}'.format(needed_features[f]), 'state'] + freqs_labels]
#
# for i in range(len(windowed_data[2])):
#     to_csv.append([pv_h_times[i], h_dip[f][i], conc_dip[i]] + windowed_data[2][i])
#
# with open('hmm_last/HMM_state_fft_DIP_{}.csv'.format(needed_features[f]), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)

# for i in range(len(windowed_data[3]) + len(windowed_data[0])):
#     to_csv.append([h_pv[f][i], conc[i]] + (windowed_data[3] + windowed_data[0])[i])
#
# with open('hmm/HMM_state_H_PV1_{}.csv'.format(needed_features[f]), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#
# to_csv = [['DIP_{}'.format(needed_features[f]), 'state'] + freqs_labels]
#
# for i in range(len(windowed_data[2])):
#     to_csv.append([to_hmm_r_f[f][i], preds_r_f[i]] + windowed_data[2][i])
#
# with open('hmm/HMM_state_DIP_{}.csv'.format(needed_features[f]), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)

# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[4])):
#     plt.plot(i, to_hmm_r_h[f][i], color=colors[preds_r_h[i]], marker='.', markersize=15)
#
# plt.title('Healthy {}'.format(needed_features[f]))

# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[1])):
#     plt.plot(i, to_hmm_l_s[i], color=colors[preds_l_s[i]], marker='.', markersize=15)
# #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])

# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[2])):
#     plt.plot(i, to_hmm_r_f[1][i], color=colors[preds_r_f[i]], marker='.', markersize=15)
# #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])
#
# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[3]) + len(windowed_data[0])):
#     plt.plot(i, h_pv[0][i], color=colors[conc[i]], marker='.', markersize=15)
#
# # plt.figure(figsize=(14, 8))
# # for i in range(len(windowed_data[1])):
# #     plt.plot(i, to_hmm_l_s[i], color=colors[preds_l_s[i]], marker='.', markersize=15)
# # #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])
#
# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[2])):
#     plt.plot(i, to_hmm_r_f[0][i], color=colors[preds_r_f[i]], marker='.', markersize=15)
# #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])
#
# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[0])):
#     plt.plot(i, to_hmm_l_f[i], color=colors[preds_l_f[i]], marker='.', markersize=15)
# #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])
#
# plt.figure(figsize=(14, 8))
# for i in range(len(windowed_data[3])):
#     plt.plot(i, to_hmm_h[i], color=colors[preds_h[i]], marker='.', markersize=15)
# #        plt.plot(to_hmm_l_s[:len(windowed_data[1])])

# plt.show()


# for f in range(13, 14):
#     to_hmm_l_f[1] = moving_avg(left_first[:, f].tolist(), 4)
#     to_hmm_l_s[1] = moving_avg(left_second[:, f].tolist(), 4)
#     to_hmm_r_f[1] = moving_avg(right_first[:, f].tolist(), 4)
#
#     to_hmm_h[1] = moving_avg(left_first_h[:, f].tolist(), 4)
#
#     to_hmm_l_f[1] = match_range(to_hmm_l_f[1], 0.4, to_hmm_h[1])
#     to_hmm_l_s[1] = match_range(to_hmm_l_s[1], 0.4, to_hmm_l_f[1])
#     to_hmm_r_f[1] = match_range(to_hmm_r_f[1], 0.4, to_hmm_l_f[1])
#
#     # plt.figure()
#     # plt.plot(to_hmm_l_f[1])
#     # plt.plot(to_hmm_h[1])
#     #
#     # # to_hmm_l_f = moving_avg(to_hmm_l_f, 10)
#     # # to_hmm_l_s = moving_avg(to_hmm_l_s, 10)
#     # # to_hmm_r_f = moving_avg(to_hmm_r_f, 10)
#     # # to_hmm_r_f = moving_avg(to_hmm_r_f, 10)
#     # # to_hmm_h = moving_avg(to_hmm_h, 10)
#     # #
#     # # plt.plot(to_hmm_l_f)
#     # # plt.plot(to_hmm_h)
#     #
#     # plt.show()
#     #
#     # to_hmm_l_f[1] = scale_mean(to_hmm_l_f[1])
#     # to_hmm_l_s[1] = scale_mean(to_hmm_l_s[1])
#     # to_hmm_r_f[1] = scale_mean(to_hmm_r_f[1])
#     #
#     # to_hmm_h[1] = scale_mean(to_hmm_h[1])
#
#     for fft_num , to_fft in enumerate([to_hmm_l_f[1], to_hmm_l_s[1], to_hmm_r_f[1], to_hmm_h[1]]):
#         window_size = 140
#         print(window_size)
#         for i in range(len(to_fft) - window_size):
#
#             cur_fft = np.abs(np.fft.fft(to_fft[i: i + window_size]))[1: window_size // 2]
#         #    print(cur_fft, np.argsort(cur_fft)[-4:])
#
#             windowed_data[fft_num][i] += [#np.mean(to_fft[i: i + window_size]),
#                                            np.subtract(*np.percentile(to_fft[i: i + window_size], [75, 25]))] + np.sort(np.argsort(cur_fft[:])[-6:-1]).tolist()


# to_hmm_l_s = match_maxes(to_hmm_l_s, 0.2, to_hmm_l_f)
# to_hmm_r_f = match_maxes(to_hmm_r_f, 0.3, to_hmm_l_f)

# plt.figure(figsize=(14, 8))
# plt.plot(to_hmm_l_f)
# plt.plot(to_hmm_r_f)
# plt.plot(to_hmm_l_s)
# plt.show()

# hmm = HMM(n_components=2)
# hmm.fit(np.array(to_hmm_l_f[:]).reshape(-1, 1))
#
# preds_l_s = hmm.predict(np.array(to_hmm_l_s[:]).reshape(-1, 1))
# preds_r_f = hmm.predict(np.array(to_hmm_r_f[:]).reshape(-1, 1))
#
# colors_l_s = ['r', 'b']
# colors_r_f = ['g', 'c']
#
# plt.figure(figsize=(14, 8))
# for i in range(len(to_hmm_l_s)):
#     plt.plot([i], [to_hmm_l_s[i]], colors_l_s[preds_l_s[i]] + '.')
#
# for i in range(len(to_hmm_r_f)):
#     plt.plot([i], [to_hmm_r_f[i]], colors_r_f[preds_r_f[i]] + '.')


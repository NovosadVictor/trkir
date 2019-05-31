import csv
import numpy as np
from scipy.stats import iqr
from matplotlib import pyplot as plt
from hmmlearn.hmm import GaussianHMM as HMM
import matplotlib.cm as cm

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
#needed_features = ['head_body_dist', 'head_rotation', 'temp']
#needed_features = ['head_body_dist', 'temp']
needed_features = ['head_rotation', 'temp']

feature_indexes = [features.index(feature) for feature in needed_features]


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
    max_2 = np.max(to_match[:int(part * len(to_match))])
    min_2 = np.min(to_match[:int(part * len(to_match))])

    before_scale = (np.array(to_match) - min_2) / (max_2 - min_2)
    after_scale = (np.array(before_scale) * (max - min) + min).tolist()

    return after_scale


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
to_hmm_h_l_f = [0] * len(needed_features)
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

    for fft_num, to_fft in enumerate(
            [to_hmm_h[f_ind] + to_hmm_l_f[f_ind], to_hmm_h[f_ind] + to_hmm_r_f[f_ind], mean_healthy[f_ind]]):
        window_size = 140

        i = 0
        while i < len(to_fft):
            part = float(i) / (len(to_fft) - 1)
            to_left = int(window_size * part)
            to_right = window_size - to_left

            cur_fft = np.abs(np.fft.fft(to_fft[i - to_left: i + to_right]))[1: window_size // 2]

            if not fft_already_exists:
                windowed_data[fft_num] += [[np.exp(iqr(to_fft[i - to_left: i + to_right]))] +
                                           cur_fft[1:13].tolist()]
            else:
                windowed_data[fft_num][i // 12] += ([np.exp(iqr(to_fft[i - to_left: i + to_right]))] +
                                              cur_fft[1:13].tolist())

            i += 12

    fft_already_exists = True

for f in range(len(needed_features)):
    to_hmm_h_l_f[f] = moving_avg(to_hmm_h[f] + to_hmm_l_f[f], 10)
    to_hmm_l_f[f] = moving_avg(to_hmm_l_f[f], 10)
    to_hmm_l_s[f] = moving_avg(to_hmm_l_s[f], 10)
    to_hmm_r_f[f] = moving_avg(to_hmm_r_f[f], 10)
    to_hmm_h[f] = moving_avg(to_hmm_h[f], 10)
    to_hmm_r_h[f] = moving_avg(to_hmm_r_h[f], 10)
    mean_healthy[f] = moving_avg(mean_healthy[f], 10)

h_pv = [to_hmm_h[f] + to_hmm_l_f[f] for f in range(len(needed_features))]
h_dip = [to_hmm_h[f] + to_hmm_r_f[f] for f in range(len(needed_features))]

for n_comp in range(8, 9):
    hmm = HMM(n_components=n_comp)
    hmm.fit(windowed_data[0])

    conc_pv = np.repeat(hmm.predict(windowed_data[0]), 12)
    conc_dip = np.repeat(hmm.predict(windowed_data[1]), 12)
    avg_h_preds = np.repeat(hmm.predict(windowed_data[2]), 12)

    with open('7_states_hmm_res.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([conc_pv, conc_dip, avg_h_preds])

    colors = cm.rainbow(np.linspace(0, 1, n_comp))

    for f in range(len(needed_features)):
        if f < len(needed_features) - 1:
            continue

        print(len(windowed_data[0]), len(conc_dip))
        plt.figure(figsize=(14, 8))

        plt.subplot(211)
        plt.title('temp intq and top 5 freqs, pv and dip')

        for i in range(len(windowed_data[0])):
            print(i)
            plt.plot(i, windowed_data[0][i][0] - 10, color=colors[conc_pv[i * 12]], marker='.', markersize=10)
        for i in range(len(windowed_data[1])):
            plt.plot(i, windowed_data[1][i][0] + 10, color=colors[conc_dip[i * 12]], marker='.', markersize=10)
        # for j in range(1, len(windowed_data[2][0])):
        #     plt.subplot(510 + j)
        #     if j == 1:
        #         plt.title('Avg. Healthy frequencies')
        #     plt.plot(np.array(windowed_data[2])[:, j], color=colors[j], label='%d frq' % j, linewidth=3)
        #     plt.ylim([0, 13])
        #     plt.legend(loc='upper right')

        plt.subplot(212)

        for j in range(1, 13):  # len(windowed_data[0][0])):
            plt.plot(np.array(windowed_data[0])[:, j] - 10, label='%d frq' % j, linewidth=3)
        plt.legend(loc='upper right')
        for j in range(1, 13):#len(windowed_data[1][0])):
            plt.plot(np.array(windowed_data[1])[:, j], label='%d frq' % j, linewidth=3)
        plt.legend(loc='upper right')

        plt.show()
#        plt.savefig('january_hmm_plots/5_freqs/hmm_{}_intq_temp.png'.format(n_comp))
        plt.close()

        for j in range(26):
            print(j, np.min(np.array(windowed_data[0])[:, j]), np.max(np.array(windowed_data[0])[:, j]))

        for j in range(26):
            print(j, np.min(np.array(windowed_data[1])[:, j]), np.max(np.array(windowed_data[1])[:, j]))

        plt.figure(figsize=(14, 8))
        plt.title('hmm with 6 states, only temp temp used, pv, dip, average healthy')

        for i in range(len(h_pv[f])):
            plt.plot(i, h_pv[f][i] + 1.1 * (np.max(h_dip[f]) - np.min(h_pv[f])), color=colors[conc_pv[i]], marker='.', markersize=10)

        for i in range(len(h_dip[f])):
            plt.plot(i, h_dip[f][i], color=colors[conc_dip[i]], marker='.', markersize=10)

        print(len(avg_h_preds), len(mean_healthy[f]))
        for i in range(len(mean_healthy[f])):
            plt.plot(i, mean_healthy[f][i] - 5, color=colors[avg_h_preds[i]], marker='.', markersize=10)

        plt.show()
        #        plt.savefig('january_hmm_plots/5_freqs/hmm_{}_intq_temp.png'.format(n_comp))
        plt.close()
 #
 #        all_possible_outcomes = list(itertools.product(np.linspace(0.8, 2, 7),
 #                                                       np.linspace(0, 5, 6),
 #                                                       np.linspace(1, 7, 7),
 #                                                       np.linspace(3, 10, 8),
 #                                                       np.linspace(4, 11, 8),
 #                                                       np.linspace(5, 12, 8)
 #                                                       ))
 #
 #        for num_data, data in enumerate(windowed_data):
 #            most_probable_outcomes = []
 #
 #            t = 0
 #            times = 1400
 #            while t < times:
 #                print(t)
 #                previous_data = data[t - 48: t]
 #                print(previous_data)
 #
 #                outcome_score = []
 #                for possible_outcome in all_possible_outcomes:
 #                    outcome_score.append(hmm.score(previous_data + [possible_outcome]))
 #
 #                most_probable_outcome = all_possible_outcomes[np.argmax(
 #                    outcome_score)]
 #
 #                print(most_probable_outcome)
 #                most_probable_outcomes += [most_probable_outcome] * 12
 #
 #                t += 12
 #
 #            with open('{}.csv'.format(num_data), 'w') as csvfile:
 #                writer = csv.writer(csvfile)
 #                writer.writerows(most_probable_outcomes)
 #
 #            cur_pred_states = hmm.predict(most_probable_outcomes)
 #
 #            plt.figure(figsize=(14, 8))
 #
 #            plt.subplot(211)
 #            for i in range(len(most_probable_outcomes)):
 #                plt.plot(i, most_probable_outcomes[i][0] - 2, color=colors[cur_pred_states[i]], marker='.', markersize=10)
 #            for j in range(1, len(most_probable_outcomes[0])):
 #                plt.plot(np.array(most_probable_outcomes)[:, j], color=colors[j], label='%d frq' % j, linewidth=3)
 #            plt.legend()
 #
 #            plt.subplot(212)
 #            for i in range(0, times):
 #                plt.plot(i, data[i][0] - 2, color=colors[conc_pv[i]], marker='.', markersize=10)
 #            for j in range(1, len(data[0])):
 #                plt.plot(np.array(data)[: times, j], color=colors[j], label='%d frq' % j, linewidth=3)
 #            plt.legend()
 #
 #            plt.savefig('%d.png' % num_data)


#         splitted_pv_preds = np.array_split(conc_pv, 10)
#         splitted_dip_preds = np.array_split(conc_dip, 10)
#         splitted_healthy_preds = np.array_split(avg_h_preds, 10)
#
#         pv_days_state_counts = [[] for r in range(10)]
#         for day in range(len(pv_days_state_counts)):
#             for state in range(n_comp):
#                 pv_days_state_counts[day].append(len(list(np.where(np.array(splitted_pv_preds[day]) == state))[0]))
#
#         dip_days_state_counts = [[] for r in range(10)]
#         for day in range(len(dip_days_state_counts)):
#             for state in range(n_comp):
#                 dip_days_state_counts[day].append(len(list(np.where(np.array(splitted_dip_preds[day]) == state))[0]))
#
#         avg_h_days_state_counts = [[] for r in range(10)]
#         for day in range(len(avg_h_days_state_counts)):
#             for state in range(n_comp):
#                 avg_h_days_state_counts[day].append(len(list(np.where(np.array(splitted_healthy_preds[day]) == state))[0]))
#
# #        print(pv_days_state_counts, dip_days_state_counts, avg_h_days_state_counts)
#
#         plt.figure(figsize=(14, 8))
#
#         ax = plt.subplot(311)
#         ax.set_title('time in each state for PV - DIP - Avg. Healthy')
#         for x in range(1, 11):
#             for state in range(n_comp):
#                 ax.bar(x - 0.3 + 0.1 * state, pv_days_state_counts[x - 1][state], width=0.1, color=colors[state], align='center')
#         ax.set_ylim([0, 150])
#         ax.set_xticks([i for i in range(1, 11)])
#         ax.set_xticklabels([i for i in range(1, 11)])
#
#         ax = plt.subplot(312)
#         for x in range(1, 11):
#             for state in range(n_comp):
#                 ax.bar(x - 0.3 + 0.1 * state, dip_days_state_counts[x - 1][state], width=0.1, color=colors[state], align='center')
#         ax.set_ylim([0, 150])
#         ax.set_xticks([i for i in range(1, 11)])
#         ax.set_xticklabels([i for i in range(1, 11)])
#
#         ax = plt.subplot(313)
#         for x in range(1, 11):
#             for state in range(n_comp):
#                 ax.bar(x - 0.3 + 0.1 * state, avg_h_days_state_counts[x - 1][state], width=0.1, color=colors[state],
#                        align='center')
#         ax.set_ylim([0, 150])
#         ax.set_xticks([i for i in range(1, 11)])
#         ax.set_xticklabels([i for i in range(1, 11)])
#
#         plt.show()

        # plt.figure()
        # for i in range(n_comp):
        #     plt.plot(i, 1, color=colors[i],  label='%d state' % (i + 1))
        #
        # plt.legend()
        # plt.show()
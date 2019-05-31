import csv
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

features = ['head_body_dist', 'body_speed_x',
            'head_speed_x' , 'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
            'body_coord_x', 'body_coord_y',
            'head_coord_x', 'head_coord_y',
            'temp',
            ]

def moving_avg(chart, N):
    moving_aves = []

    for i in range(len(chart)):
        before = i if i < N else N
        after = len(chart) - i if i > len(chart) - N - 1 else N
        moving_aves.append(np.mean(chart[i - before: i + after]))

    return moving_aves

# with open('new_{}_table.csv'.format(sys.argv[1]), newline='') as csvfile:
#     data_h = list(csv.reader(csvfile))[1:]
#
# data_h = np.array(data_h)
#
# # LEFT MOUSE FIRST EXPERIMENT
#
# left_first = data_h[np.where(data_h[:, 0] == '0')]
# left_first = left_first[np.where(left_first[:, 2] == 'L')]
#
# right_second = data_h[np.where(data_h[:, 0] == '1')]
# right_second = right_second[np.where(right_second[:, 2] == 'R')]
#
#
#
# left_first = left_first[:, 4:].astype(float)
# right_second = right_second[:, 4:].astype(float)
#
# for f in range(13, 14):#len(features)):
#     to_fft_l = moving_avg(left_first[:, f].tolist(), 10)
#     to_fft_r = moving_avg(right_second[:, f].tolist(), 10)
#  #   to_fft = left_first[:, f].tolist()
#  #   print(len(to_fft))
#     # if True:#f == 0:
#     #     plt.figure()
#     #     plt.plot(to_fft)
#     #     plt.show()
#
#     for to_fft in [to_fft_l]:#, to_fft_r]:
#
#         N = len(to_fft)
#         T = 1 / N
#
#         yf = np.fft.fft(to_fft)
#         xf = np.fft.fftfreq(N)
#
#         max_freq = xf[1:][np.argmax(np.abs(yf[1: N // 2]))]
#
#         print(max_freq)
#
#         print(len(yf))
#         abs = 2 / N * np.abs(yf[: N // 2])[:10]
#         print(abs)
#
#      #   yf[1] = 0
#
#       #   for ff in range(1, 15):
#       #       iyf = np.fft.ifft(yf[:ff].tolist(), n=N)
#       #       plt.figure(figsize=(7, 4))
#       #       plt.plot(iyf)
#       # #      plt.plot(np.abs(yf)[1: N // 2][:30])
#       #       plt.savefig('freqs/fft_{}_freqs_wo_1.png'.format(ff))
#       # #      plt.plot()
#
#
#         x = np.linspace(0, N, N)
#
#         cos = abs[0] / 2 + abs[3] * np.cos(2 * np.pi * 3 / N * x)
#         cos_2 = abs[0] / 2 + abs[6] * np.cos(2 * np.pi * 6 / N * x)
#         iyf = np.fft.ifft(yf[:7].tolist(), n=N)
#         plt.figure(figsize=(7, 4))
#     #  plt.plot(np.abs(yf[1: N // 2])[:30])
#
#     #    iyf = np.fft.ifft(yf[1:2].tolist(), n=N)
#
#         plt.plot(cos)
#         plt.plot(cos_2)
#         plt.plot(iyf)
#
#         plt.show()
#     #    plt.plot(to_fft)
#     #    plt.plot(2 / N * np.abs(yf)[1: N // 2][:20])
#      #  plt.plot(to_fft)
#    #     plt.savefig('freqs/fft_ampl.png')
#
#   #      plt.show()
#   #
#   # #   q_days = int(sys.argv[2])
#   #   splitted_to_fft = np.array_split(to_fft, q_days)
#   #   ises = np.array_split([i for i in range(len(to_fft))], q_days)
#   #
#   #   ffts = []
#   #
#   #   result_plot = []
#   #   result_ifft = []
#   #   for split in splitted_to_fft:
#   #       cur_fft = np.fft.fft(split)
#   #       print(len(cur_fft))
#   #
#   #       freqs = np.fft.fftfreq(len(split))
#   #
#   #       ffts.append(np.abs(2 / len(split) * cur_fft[1: 30]))
#   #       result_plot += split.tolist()
#   #       result_ifft += np.fft.ifft(cur_fft[:10], n=len(split)).tolist()
#   #
#   #
#   #   # fig, ax = plt.subplots()
#   #
#   #   print(len(ffts))
#   #   # for _, fft in enumerate(ffts):
#   #   #     plt.plot(fft, label=str(_))
#   #
#   #   to_csv = [['freqs'] + ['fft_day_%d' % i for i in range(q_days)]]
#   #
#   # #  freqs = np.fft.fftfreq(len(ffts[0]))
#   #
#   #   ffts.insert(0, freqs[1: 30])
#   #   ffts = np.array(ffts).T.tolist()
#   #
#   #   for fft in ffts:
#   #       to_csv.append(fft)
#   #
#   #   with open('{}_fft_example_by_day.csv'.format(sys.argv[1]), 'w') as csvfile:
#   #       writer = csv.writer(csvfile)
#   #       writer.writerows(to_csv)
#
#     # for day in range(len(ffts)):
#     #     ax.plot(ffts[day], label=str(day) + ' day')
#     #ax.plot(result_plot)
#     #ax.plot(result_ifft)
#     # sp = np.linspace(0, N, N)
#   #  freq = xf[] * (N / 2)
#
#        # ax.plot(2.0 / N * np.abs(yf[1: N // 2])[:50])
#        # ax.plot(iyf)
#        # ax.plot(to_fft)
# #     plt.title(features[f] + ' average fft')
# #
# #     plt.legend(loc='upper right')
# #
# # #  plt.ylim([0, 0.1])
# #     plt.show()
#
#     # fig, ax = plt.subplots()
#     #
#     # print(len(ffts))
#     # for _, fft in enumerate(splitted_to_fft):
#     #     plt.plot(ises[_], fft, label=str(_))
#     #
#     #
#     # plt.title(features[f] + ' colored by 6 hours')
#     # plt.show()


with open('new_Sick_table.csv', newline='') as csvfile:
    data_s = list(csv.reader(csvfile))[1:]

data_s = np.array(data_s)

# LEFT MOUSE FIRST EXPERIMENT

left_first = data_s[np.where(data_s[:, 0] == '0')]
left_first = left_first[np.where(left_first[:, 2] == 'L')]

left_first = left_first[:, 4:].astype(float)

for f in range(13, 14):#len(features)):
    to_fft = moving_avg(left_first[:, f].tolist(), 1)
    to_fft = ((np.array(to_fft) - np.mean(to_fft)) / (np.max(to_fft) - np.min(to_fft))).tolist()
 #   print(len(to_fft))
    if f == 0:
        plt.figure()
        plt.plot(to_fft)
        plt.show()

    N = len(to_fft)
    T = 1 / N

    q_days = 8
    splitted_to_fft = np.array_split(to_fft, q_days)
    ises = np.array_split([i for i in range(len(to_fft))], q_days)

    ffts_s = []

    result_plot = []
    result_ifft = []
    for split in splitted_to_fft:
        cur_fft = np.fft.fft(split)
        print(len(cur_fft))

        freqs = np.fft.fftfreq(len(split))

        ffts_s.append(np.abs(2 / len(split) * cur_fft[1: 50]))
        result_plot += split.tolist()
        result_ifft += np.fft.ifft(cur_fft[:10], n=len(split)).tolist()



  #   yf = np.fft.fft(to_fft)
  #   xf = np.fft.fftfreq(N)
  #
  #   max_freq = np.argmax(np.abs(yf[1:])) + 1
  #   # print(max_freq)
  #   print(len(yf))
  #
  #   iyf = np.fft.ifft(yf[: 200].tolist() + yf[-200:].tolist(), n=N)
  #
  #   sp = np.linspace(0, N, N)
  # #  freq = xf[] * (N / 2)
  #
  #
  #   fig, ax = plt.subplots()
  #
  # #  ax.plot(xf[:50], 2.0 / N * np.abs(yf[: N // 2])[1:51])
  #   ax.plot(iyf)
  #   ax.plot(to_fft)
  #   plt.title(features[f])
  #   plt.show()




with open('new_Healthy_table.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))[1:]

data = np.array(data)

# LEFT MOUSE FIRST EXPERIMENT

left_first = data[np.where(data[:, 0] == '0')]
left_first = left_first[np.where(left_first[:, 2] == 'L')]

days = sorted(set(left_first[:, 3]))
left_f_by_day = [left_first[np.where(left_first[:, 3] == day)] for day in days]

left_first = [left[:, 4:].astype(float) for left in left_first]

for f in range(13, 14):#len(features)):
    ffts_h = []

    for d in range(len(left_first)):
        to_fft = moving_avg(left_first[d][:, f].tolist(), 1)
        to_fft = ((np.array(to_fft) - np.mean(to_fft)) / (np.max(to_fft) - np.min(to_fft))).tolist()
        print(len(to_fft))
        if f == 0:
            plt.figure()
            plt.plot(to_fft)
            plt.show()

        cur_fft = np.fft.fft(to_fft)

        freqs = np.fft.fftfreq(len(to_fft))

        ffts_h.append(np.abs(2 / len(to_fft) * cur_fft[1: 40]))


# plt.figure(figsize=(14, 8))
#
# for fft in ffts_h + ffts_s:
#     plt.plot(fft)
#
# plt.show()

labels = [0, 1, 2]#[0 for i in range(len(ffts_h))]# + [1 for i in range(len(ffts_s))]

plt.figure()
for d in range(3):
    for fft in ffts_d[d]:
        plt.plot(fft, 'C%d' % d)

plt.show()
# pairs = [[i, j] for i in range(len(ffts_s)) for j in range(len(ffts_s)) if i != j]
# for pair in pairs:
#     fft_1 = ffts_s[pair[0]]
#     fft_2 = ffts_s[pair[1]]
#
#     rf = RandomForestClassifier()
#
#     labels = [0, 1]
#
#     rf.fit([fft_1 , fft_2], labels)
#
#     print(pair, rf.feature_importances_)
#
#     plt.figure(figsize=(14, 8))
#
#     plt.plot(fft_1, 'C0')
#     plt.plot(fft_2, 'C1')
#
#     plt.show()



  #   yf = np.fft.fft(to_fft)
  #   xf = np.fft.fftfreq(N)
  #
  #   max_freq = np.argmax(np.abs(yf[1:])) + 1
  #   print(max_freq)
  #
  #   iyf = np.fft.ifft(yf[max_freq - 1: max_freq + 2].tolist() + yf[-max_freq - 2: - max_freq + 1].tolist(), n=N)
  #
  #   sp = np.linspace(0, N, N)
  # #  freq = xf[] * (N / 2)
  #
  #
  #   fig, ax = plt.subplots()
  #
  # #  ax.plot(xf[:50], 2.0 / N * np.abs(yf[: N // 2])[1:51])
  #   ax.plot(iyf)
  #   ax.plot(to_fft)
  #   plt.title(features[f])
  #   plt.show()
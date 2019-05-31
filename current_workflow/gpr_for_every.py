import csv
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

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

with open('new_Healthy_table.csv', newline='') as csvfile:
    data_h = list(csv.reader(csvfile))[1:]
with open('new_Sick_table.csv', newline='') as csvfile:
    data_s = list(csv.reader(csvfile))[1:]
# with open('new_{}_table.csv'.format(sys.argv[1]), newline='') as csvfile:
#     data = list(csv.reader(csvfile))[1:]

#data = np.array(data)
data_h = np.array(data_h)
data_s = np.array(data_s)

left_first_h = data_h[np.where(data_h[:, 0] == '0')]
left_first_h = left_first_h[np.where(left_first_h[:, 2] == 'L')]
left_first_s = data_s[np.where(data_s[:, 0] == '0')]
left_first_s = left_first_s[np.where(left_first_s[:, 2] == 'L')]

left_first_h = left_first_h[:, 4:].astype(float)
left_first_s = left_first_s[:, 4:].astype(float)

for f in range(13, 14):
    to_gpr_h = moving_avg(left_first_h[:, f].tolist(), 20)
    to_gpr_s = moving_avg(left_first_s[:, f].tolist(), 20)

    to_gpr_h = (to_gpr_h * (len(to_gpr_s) // len(to_gpr_h) + 1))[: len(to_gpr_s)][20:]

    if True:
        plt.figure()
        plt.plot(to_gpr_h)
        plt.plot(to_gpr_s)
        plt.show()

    step_h = 5
    step_s = 5
    x_h = np.array([i for i in range(len(to_gpr_h))]).reshape(-1, 1)
    x_s = np.array([i for i in range(len(to_gpr_s))]).reshape(-1, 1)
    X_h = x_h[::step_h]
    X_s = x_s[::step_s]

    kernel = C(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))
    gp_h = GPR(kernel=kernel, n_restarts_optimizer=9)
    gp_s = GPR(kernel=kernel, n_restarts_optimizer=9)
    gp_h.fit(X_h, np.array(to_gpr_h[::step_h]).reshape(-1, 1))
    gp_s.fit(X_s, np.array(to_gpr_s[::step_s]).reshape(-1, 1))

    y_pred_h, sigma_h = gp_h.predict(x_h, return_std=True)
    y_pred_s, sigma_s = gp_s.predict(x_s, return_std=True)

    plt.figure(figsize=(14, 8))

#    plt.plot(x, to_gpr_h, 'c:')
#    plt.plot(X, to_gpr_h[::step], 'r.', markersize=5)
    plt.plot(x_h, y_pred_h, 'c-')

    plt.fill(np.concatenate([x_h, x_h[::-1]]),
             np.concatenate([y_pred_h - 0.6 * sigma_h, #1.9600
                             (y_pred_h + 0.6 * sigma_h)[::-1]]),
             alpha=0.5, fc='g', ec='None')

#    plt.plot(x, to_gpr_s, 'r:')
#    plt.plot(X, to_gpr_s[::step], 'r.', markersize=5)
    plt.plot(x_s, y_pred_s, 'b-')

    plt.fill(np.concatenate([x_s, x_s[::-1]]),
             np.concatenate([y_pred_s - 0.6 * sigma_s, #1.9600
                             (y_pred_s + 0.6 * sigma_s)[::-1]]),
             alpha=0.5, fc='r', ec='None')

    plt.show()


# LEFT MOUSE FIRST EXPERIMENT

# left_first = data[np.where(data[:, 0] == '0')]
# left_first = left_first[np.where(left_first[:, 2] == 'L')]
#
# left_first = left_first[:, 4:].astype(float)
#
# for f in range(1):
#     to_gpr = moving_avg(left_first[:, f].tolist(), 20)
#     print(len(to_gpr))
#     # if f == 0:
#     #     plt.figure()
#     #     plt.plot(to_gpr)
#     #     plt.show()
#
#     step = 5
#     x = np.array([i for i in range(len(to_gpr))]).reshape(-1, 1)
#     X = x[::step]
#
#     kernel = C(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))
#     gp = GPR(kernel=kernel, n_restarts_optimizer=9)
#     gp.fit(X, np.array(to_gpr[::step]).reshape(-1, 1))
#
#     y_pred, sigma = gp.predict(x, return_std=True)
#
#     plt.figure(figsize=(14, 8))
#     plt.plot(x, to_gpr, 'r:', label='f(x)')
#   #  plt.plot(X, to_gpr[::step], 'r.', markersize=5, label=u'Observations')
#   #  plt.plot(x, y_pred, 'b-', label=u'Prediction')
#   #   samples = gp.sample_y(x, n_samples=5)
#   #   for s in range(samples.shape[-1]):
#   #       plt.plot(x, samples[:, :, s], 'g-', label='sample')
#     plt.fill(np.concatenate([x, x[::-1]]),
#              np.concatenate([y_pred - 1 * sigma, #1.9600
#                              (y_pred + 1 * sigma)[::-1]]),
#              alpha=0.5, fc='g', ec='None')
#     plt.xlabel('$x$')
#     plt.ylabel('$f(x)$')
#     plt.legend(loc='upper left')
#
#     plt.show()
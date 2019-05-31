import csv
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from sklearn.metrics import mean_squared_error

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


def match_maxes(to_match, part, plot):
    max = np.max(plot)
    max_to_match = np.max(to_match[:int(part * len(to_match))])

    return np.array(to_match) + (max - max_to_match)


def mean_scaling(list):
    return (np.array(list) - np.mean(list)) / (np.max(list) - np.min(list))


def create_dataset(dataset_x, dataset_y=None, look_back=1, labels=True):
    dataX, dataY = [], []

    for i in range(len(dataset_x)-look_back-1):
        a = dataset_x[i: (i + look_back)]
        dataX.append(a)
        if labels:
            dataY.append(dataset_y[i + look_back])

    if labels:
        return np.array(dataX), np.array(dataY)

    return np.array(dataX)


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

np.random.seed(7)

l_f = []
l_s = []
r_f = []

look_back = 5
n_features = 2

for _, f in enumerate([0, 13]):
    l_f.append(moving_avg(left_first[:, f].tolist(), 20))
    l_s.append(moving_avg(left_second[:, f].tolist(), 20))
    r_f.append(moving_avg(right_first[:, f].tolist(), 20))

    l_s[_] = match_maxes(l_s[_], 0.2, l_f[_])
    r_f[_] = match_maxes(r_f[_], 0.3, l_f[_])

    l_f[_] = mean_scaling(l_f[_])
    l_s[_] = mean_scaling(l_s[_])
    r_f[_] = mean_scaling(r_f[_])

l_f = np.concatenate([np.array(l_f[0]).reshape(-1, 1), np.array(l_f[1]).reshape(-1, 1)], axis=1).tolist()
l_s = np.concatenate([np.array(l_s[0]).reshape(-1, 1), np.array(l_s[1]).reshape(-1, 1)], axis=1).tolist()
r_f = np.concatenate([np.array(r_f[0]).reshape(-1, 1), np.array(r_f[1]).reshape(-1, 1)], axis=1).tolist()

train_labels = [[1, 0] for i in range(552)] + [[0, 1] for i in range(len(l_f) - 552)]
test_labels = [[1, 0] for i in range(len(r_f))]

train_X, train_Y = create_dataset(l_f, train_labels, look_back)
test_X, test_Y = create_dataset(r_f, test_labels, look_back)
pv_test_X = create_dataset(l_s, look_back=look_back, labels=False)

print(train_X.shape)
train_X = np.reshape(train_X, (train_X.shape[0], look_back, n_features))
test_X = np.reshape(test_X, (test_X.shape[0], look_back, n_features))
pv_test_X = np.reshape(pv_test_X, (pv_test_X.shape[0], pv_test_X.shape[1], n_features))

model = Sequential()
model.add(LSTM(6, input_shape=(look_back, n_features)))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=5, batch_size=5, verbose=2)

print(train_X.shape, pv_test_X.shape)
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)
pv_predict = model.predict(pv_test_X)

trainPredict = np.around(trainPredict)
testPredict = np.around(testPredict)
pv_predict = np.around(pv_predict)

print(testPredict)

trainScore = 1 - mean_squared_error(train_Y, trainPredict[:])
testScore = 1 - mean_squared_error(test_Y, testPredict[:])
print('Train Score: %.2f accuracy' % (trainScore))
print('Test Score: %.2f accuracy' % (testScore))

colors = [['b', 'r'], ['g', 'c']]

for f in range(n_features):
    plt.figure(figsize=(14, 8))

    # for i in range(min(len(r_f), len(testPredict))):
    #     plt.plot([i], [r_f[i]], colors[np.argmax(testPredict[i])] + '.')

    for i in range(min(len(l_s), len(pv_predict))):
        plt.plot([i], [l_s[i][f]], colors[f][np.argmax(pv_predict[i])] + '.')

    plt.show()

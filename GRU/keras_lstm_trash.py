from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import pandas as pd
from random import random
import numpy as np
import csv


with open('7_states_hmm_res.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(len(data[0]), len(data[1]), len(data[2]))

for i in range(len(data)):
    data[i] = data[i][::12][:120]

data = np.array(data).astype(float).tolist()

print(len(data[0]), len(data[1]), len(data[2]))
train_data = data[:]
for i in range(len(train_data)):
    train_data[i] = np.array(np.array_split(train_data[i], 24))

print(np.array(train_data).shape)

train_data = np.reshape(train_data, (3, 24, 5))
print(train_data)

y_train = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

in_neurons = 5
out_neurons = 3
hidden_neurons = 124

model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(24, 5), return_sequences=True))
model.add(Dense(out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(train_data, y_train, batch_size=3, epochs=10)

predicted = model.predict(train_data)
rmse = np.sqrt(((predicted - y_train) ** 2).mean(axis=0))

print(predicted, rmse)


#
# flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
# pdata = pd.DataFrame({"a":flow, "b":flow})
# pdata.b = pdata.b.shift(9)
# data = pdata.iloc[10:] * random()  # some noise
# print(data)
#
#
# def _load_data(data, n_prev = 100):
#     """
#     data should be pd.DataFrame()
#     """
#
#     docX, docY = [], []
#     for i in range(len(data)-n_prev):
#         docX.append(data.iloc[i:i+n_prev].as_matrix())
#         docY.append(data.iloc[i+n_prev].as_matrix())
#     alsX = np.array(docX)
#     alsY = np.array(docY)
#
#     return alsX, alsY
#
#
# def train_test_split(df, test_size=0.1):
#     """
#     This just splits data to training and testing parts
#     """
#     ntrn = round(len(df) * (1 - test_size))
#
#     X_train, y_train = _load_data(df.iloc[0:ntrn])
#     X_test, y_test = _load_data(df.iloc[ntrn:])
#
#     return (X_train, y_train), (X_test, y_test)
#
# (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
# print(X_train[0], X_train[1], y_train[0])
#
# # and now train the model
# # batch_size should be appropriate to your memory size
# # number of epochs should be higher for real world problems
# #model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)
#
# # in_out_neurons = 2
# # hidden_neurons = 300
# #
# # model = Sequential()
# # model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
# # model.add(Dense(hidden_neurons, in_out_neurons))
# # model.add(Activation("linear"))
# # model.compile(loss="mean_squared_error", optimizer="rmsprop")
#

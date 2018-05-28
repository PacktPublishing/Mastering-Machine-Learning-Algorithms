import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler


# Set random seed for reproducibility
np.random.seed(1000)

# Download the dataset from: https://datamarket.com/data/set/22ti/zuerich-monthly-sunspot-numbers-1749-1983#!ds=22ti&display=line﻿
dataset_filename = '<YOUR_PATH>\dataset.csv'

n_samples = 2820
data = np.zeros(shape=(n_samples, ), dtype=np.float32)

sequence_length = 15


if __name__ == '__main__':
    # Load the dataset
    with open(dataset_filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 0:
            continue

        if i == n_samples + 1:
            break

        _, value = line.split(',')
        data[i - 1] = float(value)

    # Scale the dataset between -1 and 1
    mmscaler = MinMaxScaler((-1.0, 1.0))
    data = mmscaler.fit_transform(data.reshape(-1, 1))

    # Show the dataset
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(data)
    ax.grid()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Monthly sunspots numbers')
    plt.show()

    # Create the train and test sets (rounding to 2800 samples)
    X_ts = np.zeros(shape=(n_samples - sequence_length, sequence_length, 1), dtype=np.float32)
    Y_ts = np.zeros(shape=(n_samples - sequence_length, 1), dtype=np.float32)

    for i in range(0, data.shape[0] - sequence_length):
        X_ts[i] = data[i:i + sequence_length]
        Y_ts[i] = data[i + sequence_length]

    X_ts_train = X_ts[0:2300, :]
    Y_ts_train = Y_ts[0:2300]

    X_ts_test = X_ts[2300:2800, :]
    Y_ts_test = Y_ts[2300:2800]

    # Create the model
    model = Sequential()

    model.add(LSTM(4, stateful=True, batch_input_shape=(20, sequence_length, 1)))

    model.add(Dense(1))
    model.add(Activation('tanh'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001, decay=0.0001),
                  loss='mse',
                  metrics=['mse'])

    # Train the model
    model.fit(X_ts_train, Y_ts_train,
              batch_size=20,
              epochs=100,
              shuffle=False,
              validation_data=(X_ts_test, Y_ts_test))

    # Show the result
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(Y_ts_test, label='True values')
    ax.plot(model.predict(X_ts_test, batch_size=20), label='Predicted values')
    ax.grid()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Monthly sunspots numbers')
    ax.legend()
    plt.show()






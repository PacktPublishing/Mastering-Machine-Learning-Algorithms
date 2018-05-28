import matplotlib.pyplot as plt
import numpy as np

# To install Keras: pip install -U keras
# Further information: https://keras.io
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000
nsb = int(nb_samples / 4)


if __name__ == '__main__':
    # Create dataset
    X = np.zeros((nb_samples, 2))
    Y = np.zeros((nb_samples,))

    X[0:nsb, :] = np.random.multivariate_normal([1.0, -1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[0:nsb] = 0.0

    X[nsb:(2 * nsb), :] = np.random.multivariate_normal([1.0, 1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[nsb:(2 * nsb)] = 1.0

    X[(2 * nsb):(3 * nsb), :] = np.random.multivariate_normal([-1.0, 1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[(2 * nsb):(3 * nsb)] = 0.0

    X[(3 * nsb):, :] = np.random.multivariate_normal([-1.0, -1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[(3 * nsb):] = 1.0

    ss = StandardScaler()
    X = ss.fit_transform(X)

    X, Y = shuffle(X, Y, random_state=1000)

    # Create an MLP
    model = Sequential()

    model.add(Dense(4, input_dim=2))
    model.add(Activation('tanh'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1000)

    # Train the model
    model.fit(X_train,
              to_categorical(Y_train, num_classes=2),
              epochs=100,
              batch_size=32,
              validation_data=(X_test, to_categorical(Y_test, num_classes=2)))

    # Plot the classification result
    Y_pred = model.predict(X)
    Y_pred_mlp = np.argmax(Y_pred, axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y_pred_mlp == 0, 0], X[Y_pred_mlp == 0, 1])
    ax.scatter(X[Y_pred_mlp == 1, 0], X[Y_pred_mlp == 1, 1])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.semi_supervised import LabelSpreading

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 5000
nb_unlabeled = 1000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, random_state=100)
    Y[nb_samples - nb_unlabeled:nb_samples] = -1

    # Create and fit a LabelSpreading instance
    ls = LabelSpreading(kernel='rbf', gamma=10.0, alpha=0.2)
    ls.fit(X, Y)

    Y_final = ls.predict(X)

    # Show the final result
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], color='#88d7f0', marker='s', s=100)
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], color='#55ffec', marker='o', s=100)
    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='x', s=20)

    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].set_title('Dataset')
    ax[0].grid()

    ax[1].scatter(X[Y_final == 0, 0], X[Y_final == 0, 1], color='#88d7f0', marker='s', s=100)
    ax[1].scatter(X[Y_final == 1, 0], X[Y_final == 1, 1], color='#55ffec', marker='o', s=100)

    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].set_title(r'Scikit-Learn Label Spreading ($\alpha$=0.2)')
    ax[1].grid()

    plt.show()
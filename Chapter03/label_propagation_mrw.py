import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 2000
nb_unlabeled = 1950
nb_classes = 2


def rbf(x1, x2, sigma=1.0):
    d = np.linalg.norm(x1 - x2, ord=1)
    return np.exp(-np.power(d, 2.0) / (2 * np.power(sigma, 2)))


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples,
                  n_features=2,
                  centers=nb_classes,
                  cluster_std=2.5,
                  random_state=500)

    Y[nb_samples - nb_unlabeled:] = -1

    # Show the original dataset
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='x', s=50)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=100)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='g', marker='s', s=100)

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()

    # Create the weight matrix
    W = kneighbors_graph(X, n_neighbors=15, mode='connectivity', include_self=True).toarray()

    for i in range(nb_samples):
        for j in range(nb_samples):
            if W[i, j] != 0.0:
                W[i, j] = rbf(X[i], X[j])

    # Compute the Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    Luu = L[nb_samples - nb_unlabeled:, nb_samples - nb_unlabeled:]
    Wul = W[nb_samples - nb_unlabeled:, 0:nb_samples - nb_unlabeled,]
    Yl = Y[0:nb_samples - nb_unlabeled]

    # Perform the random walk
    Yu = np.round(np.linalg.solve(Luu, np.dot(Wul, Yl)))
    Y[nb_samples - nb_unlabeled:] = Yu.copy()

    # Show the final dataset
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=100)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='g', marker='s', s=100)

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()

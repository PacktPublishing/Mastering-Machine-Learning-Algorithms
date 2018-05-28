import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5, random_state=1000)

    # Show the unclustered dataset
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[:, 0], X[:, 1], s=40)
    ax.grid()
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')

    plt.show()

    # Create and fit a Gaussian Mixture
    gm = GaussianMixture(n_components=3)
    gm.fit(X)

    # Show the Gaussian parameters
    print('Weights:\n')
    print(gm.weights_)

    print('\nMeans:\n')
    print(gm.means_)

    print('\nCovariances:\n')
    print(gm.covariances_)

    # Show the clustered dataset
    Yp = gm.predict(X)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[Yp == 0, 0], X[Yp == 0, 1], c='red', marker='o', s=50)
    ax.scatter(X[Yp == 1, 0], X[Yp == 1, 1], c='blue', marker='x', s=50)
    ax.scatter(X[Yp == 2, 0], X[Yp == 2, 1], c='green', marker='s', s=50)
    ax.grid()
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')

    plt.show()
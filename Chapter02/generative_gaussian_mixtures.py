import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 1000
nb_unlabeled = 750
nb_iterations = 5

# First Gaussian
m1 = np.random.uniform(-7.5, 10.0, size=2)
c1 = np.random.uniform(5.0, 15.0, size=(2, 2))
c1 = np.dot(c1, c1.T)
q1 = 0.5

# Second Gaussian
m2 = np.random.uniform(-7.5, 10.0, size=2)
c2 = np.random.uniform(5.0, 15.0, size=(2, 2))
c2 = np.dot(c2, c2.T)
q2 = 0.5


def show_dataset():
    fig, ax = plt.subplots(figsize=(20, 15))

    g1 = Ellipse(xy=m1, width=3 * np.sqrt(c1[0, 0]), height=3 * np.sqrt(c1[1, 1]), fill=False, linestyle='dashed',
                 linewidth=1)
    g1_1 = Ellipse(xy=m1, width=2 * np.sqrt(c1[0, 0]), height=2 * np.sqrt(c1[1, 1]), fill=False, linestyle='dashed',
                   linewidth=2)
    g1_2 = Ellipse(xy=m1, width=1.4 * np.sqrt(c1[0, 0]), height=1.4 * np.sqrt(c1[1, 1]), fill=False, linestyle='dashed',
                   linewidth=3)

    g2 = Ellipse(xy=m2, width=3 * np.sqrt(c2[0, 0]), height=3 * np.sqrt(c2[1, 1]), fill=False, linestyle='dashed',
                 linewidth=1)
    g2_1 = Ellipse(xy=m2, width=2 * np.sqrt(c2[0, 0]), height=2 * np.sqrt(c2[1, 1]), fill=False, linestyle='dashed',
                   linewidth=2)
    g2_2 = Ellipse(xy=m2, width=1.4 * np.sqrt(c2[0, 0]), height=1.4 * np.sqrt(c2[1, 1]), fill=False, linestyle='dashed',
                   linewidth=3)

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='#88d7f0', s=100)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='#55ffec', s=100)
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='d', s=25)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    # Generate dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=2.5, random_state=100)

    unlabeled_idx = np.random.choice(np.arange(0, nb_samples, 1), replace=False, size=nb_unlabeled)
    Y[unlabeled_idx] = -1

    # Show the dataset with the initial Gaussians
    show_dataset()

    # Training process
    for i in range(nb_iterations):
        Pij = np.zeros((nb_samples, 2))

        # E Step
        for i in range(nb_samples):
            if Y[i] == -1:
                p1 = multivariate_normal.pdf(X[i], m1, c1, allow_singular=True) * q1
                p2 = multivariate_normal.pdf(X[i], m2, c2, allow_singular=True) * q2
                Pij[i] = [p1, p2] / (p1 + p2)

            else:
                Pij[i, :] = [1.0, 0.0] if Y[i] == 0 else [0.0, 1.0]

        # M Step
        n = np.sum(Pij, axis=0)
        m = np.sum(np.dot(Pij.T, X), axis=0)

        m1 = np.dot(Pij[:, 0], X) / n[0]
        m2 = np.dot(Pij[:, 1], X) / n[1]

        q1 = n[0] / float(nb_samples)
        q2 = n[1] / float(nb_samples)

        c1 = np.zeros((2, 2))
        c2 = np.zeros((2, 2))

        for t in range(nb_samples):
            c1 += Pij[t, 0] * np.outer(X[t] - m1, X[t] - m1)
            c2 += Pij[t, 1] * np.outer(X[t] - m2, X[t] - m2)

        c1 /= n[0]
        c2 /= n[1]

    # Show the final Gaussians
    show_dataset()

    # Check some points
    print('The first 10 unlabeled samples:')
    print(np.round(X[Y == -1][0:10], 3))

    print('\nCorresponding Gaussian assigments:')
    print(np.round(Pij[Y == -1][0:10], 3))
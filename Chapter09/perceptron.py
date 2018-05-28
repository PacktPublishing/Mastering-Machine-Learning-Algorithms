import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import cpu_count

from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
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

    # Show the dataset
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()

    # Perform a classification based on a Perceptron
    # Starting from Scikit-Learn 0.19 it's helpful to include a max_iter or tol parameter to avoid a warning
    pc = Perceptron(penalty='l2', alpha=0.1, n_jobs=cpu_count(), random_state=1000)
    print('Perceptron CV score: {}'.format(np.mean(cross_val_score(pc, X, Y, cv=10))))

    # Show a classification result
    pc.fit(X, Y)
    Y_pred_perceptron = pc.predict(X)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y_pred_perceptron == 0, 0], X[Y_pred_perceptron == 0, 1])
    ax.scatter(X[Y_pred_perceptron == 1, 0], X[Y_pred_perceptron == 1, 1])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()




import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(1000)


n_components = 2
learning_rate = 0.01
nb_iterations = 5000
t = 0.0


def zero_center(Xd):
    return Xd - np.mean(Xd, axis=0)


if __name__ == '__main__':
    # Create the dataset
    X, _ = make_blobs(n_samples=500, centers=2, cluster_std=5.0, random_state=1000)
    Xs = zero_center(X)

    Q = np.cov(Xs.T)
    eigu, eigv = np.linalg.eig(Q)

    print('Eigenvalues: {}'.format(eigu))
    print('Eigenvectors: {}'.format(eigv))

    # Initialize the weights
    W_sanger = np.random.normal(scale=0.5, size=(n_components, Xs.shape[1]))
    W_sanger /= np.linalg.norm(W_sanger, axis=1).reshape((n_components, 1))

    # Perform the training cycle
    for i in range(nb_iterations):
        dw = np.zeros((n_components, Xs.shape[1]))
        t += 1.0

        for j in range(Xs.shape[0]):
            Ysj = np.dot(W_sanger, Xs[j]).reshape((n_components, 1))
            QYd = np.tril(np.dot(Ysj, Ysj.T))
            dw += np.dot(Ysj, Xs[j].reshape((1, X.shape[1]))) - np.dot(QYd, W_sanger)

        W_sanger += (learning_rate / t) * dw
        W_sanger /= np.linalg.norm(W_sanger, axis=1).reshape((n_components, 1))

    print('Final weights: {}'.format(W_sanger.T))

    # Plot the final configuration
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(Xs[:, 0], Xs[:, 1], c='blue')
    ax.set_xlabel(r'$x_0$')
    ax.set_xlabel(r'$x_1$')
    W = W_sanger * 15

    ax.arrow(0, 0, W[0, 0], W[0, 1], head_width=1.0, head_length=2.0, fc='k', ec='k')
    ax.annotate(r'$w_0$', xy=(1.0, 1.0), xycoords='data', xytext=(W[0, 0] + 0.5, W[0, 1] + 0.5), textcoords='data',
                size=20)

    ax.arrow(0, 0, W[1, 0], W[1, 1], head_width=1.0, head_length=2.0, fc='k', ec='k')
    ax.annotate(r'$w_1$', xy=(1.0, 1.0), xycoords='data', xytext=(W[1, 0] + 0.5, W[1, 1] + 0.5), textcoords='data',
                size=20)

    ax.grid()
    plt.show()
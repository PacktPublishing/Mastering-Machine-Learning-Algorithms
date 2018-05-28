import numpy as np

from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(1000)


n_components = 2
learning_rate = 0.0001
max_iterations = 1000
stabilization_cycles = 5
threshold = 0.00001


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

    # Initialize the variables
    W = np.random.normal(0.0, 0.5, size=(Xs.shape[1], n_components))
    V = np.tril(np.random.normal(0.0, 0.01, size=(n_components, n_components)))
    np.fill_diagonal(V, 0.0)

    prev_W = np.zeros((Xs.shape[1], n_components))
    t = 0

    # Perform the training cycle
    while (np.linalg.norm(W - prev_W, ord='fro') > threshold and t < max_iterations):
        prev_W = W.copy()
        t += 1

        for i in range(Xs.shape[0]):
            y_p = np.zeros((n_components, 1))
            xi = np.expand_dims(Xs[i], 1)
            y = None

            for _ in range(stabilization_cycles):
                y = np.dot(W.T, xi) + np.dot(V, y_p)
                y_p = y.copy()

            dW = np.zeros((Xs.shape[1], n_components))
            dV = np.zeros((n_components, n_components))

            for t in range(n_components):
                y2 = np.power(y[t], 2)
                dW[:, t] = np.squeeze((y[t] * xi) + (y2 * np.expand_dims(W[:, t], 1)))
                dV[t, :] = -np.squeeze((y[t] * y) + (y2 * np.expand_dims(V[t, :], 1)))

            W += (learning_rate * dW)
            V += (learning_rate * dV)

            V = np.tril(V)
            np.fill_diagonal(V, 0.0)

            W /= np.linalg.norm(W, axis=0).reshape((1, n_components))

    print('Final w: {}'.format(W))

    # Compute the covariance matrix
    Y_comp = np.zeros((Xs.shape[0], n_components))

    for i in range(Xs.shape[0]):
        y_p = np.zeros((n_components, 1))
        xi = np.expand_dims(Xs[i], 1)

        for _ in range(stabilization_cycles):
            Y_comp[i] = np.squeeze(np.dot(W.T, xi) + np.dot(V.T, y_p))
            y_p = y.copy()

    print('Final covariance matrix: {}'.format(np.cov(Y_comp.T)))
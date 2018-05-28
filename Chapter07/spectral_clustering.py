import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X = np.zeros(shape=(nb_samples, 2))

    for i in range(nb_samples):
        X[i, 0] = float(i)

        if i % 2 == 0:
            X[i, 1] = 1.0 + (np.random.uniform(0.65, 1.0) * np.sin(float(i) / 100.0))
        else:
            X[i, 1] = 0.1 + (np.random.uniform(0.5, 0.85) * np.sin(float(i) / 100.0))

    ss = StandardScaler()
    Xs = ss.fit_transform(X)

    # Test K-Means
    km = KMeans(n_clusters=2, random_state=1000)
    Y_km = km.fit_predict(Xs)

    # Plot the result
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(Xs[Y_km == 0, 0], Xs[Y_km == 0, 1])
    ax.scatter(Xs[Y_km == 1, 0], Xs[Y_km == 1, 1])

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()

    # Apply Spectral clustering
    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=20, random_state=1000)
    Y_sc = sc.fit_predict(Xs)

    # Plot the result
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(Xs[Y_sc == 0, 0], Xs[Y_sc == 0, 1])
    ax.scatter(Xs[Y_sc == 1, 0], Xs[Y_sc == 1, 1])

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()





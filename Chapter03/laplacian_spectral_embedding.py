import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.manifold import SpectralEmbedding

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    faces = fetch_olivetti_faces()

    # Train Laplacian Spectral Embedding
    se = SpectralEmbedding(n_components=2, n_neighbors=15)
    X_se = se.fit_transform(faces['data'])

    # Plot the result
    fig, ax = plt.subplots(figsize=(18, 10))

    for i in range(400):
        ax.scatter(X_se[:, 0], X_se[:, 1], color=cm.rainbow(faces['target'] * 10), marker='o', s=30)
        ax.annotate('%d' % faces['target'][i], xy=(X_se[i, 0] + 0.001, X_se[i, 1] + 0.001))

    ax.set_xlim([-0.15, 0.0])
    ax.set_ylim([-0.2, 0.4])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.manifold import LocallyLinearEmbedding

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    faces = fetch_olivetti_faces()

    # Train LLE
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
    X_lle = lle.fit_transform(faces['data'])

    # Plot the result
    fig, ax = plt.subplots(figsize=(18, 10))

    for i in range(100):
        ax.scatter(X_lle[i, 0], X_lle[i, 1], marker='o', s=100)
        ax.annotate('%d' % faces['target'][i], xy=(X_lle[i, 0] + 0.0015, X_lle[i, 1] + 0.0015))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()
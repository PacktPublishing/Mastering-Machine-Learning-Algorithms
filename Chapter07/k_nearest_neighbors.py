import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X_train = digits['data'] / np.max(digits['data'])

    # Perform kNN
    knn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
    knn.fit(X_train)

    # Query the model
    distances, neighbors = knn.kneighbors(X_train[100].reshape(1, -1), return_distance=True)

    print('Distances: {}'.format(distances[0]))

    # Plot the neighbors
    fig, ax = plt.subplots(5, 10, figsize=(8, 8))

    for y in range(5):
        for x in range(10):
            idx = neighbors[0][(x + (y * 10))]
            ax[y, x].matshow(digits['images'][idx], cmap='gray')
            ax[y, x].set_xticks([])
            ax[y, x].set_yticks([])

    plt.show()


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(1000)


def zero_center(Xd):
    return Xd - np.mean(Xd, axis=0)


if __name__ == '__main__':
    # Load the dataset
    digits = fetch_mldata('MNIST original')
    X = zero_center(digits['data'].astype(np.float64))
    np.random.shuffle(X)

    # Create dataset + heteroscedastic noise
    Omega = np.random.uniform(0.0, 0.75, size=X.shape[1])
    Xh = X + np.random.normal(0.0, Omega, size=X.shape)

    # Show dataset + heteroscedastic noise plot
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(Xh[(i * 10) + j].reshape((28, 28)), cmap='gray')
            ax[i, j].axis('off')

    plt.show()

    # Perform a PCA on the dataset + heteroscedastic noise
    pca = PCA(n_components=64, svd_solver='full', random_state=1000)
    Xpca = pca.fit_transform(Xh)

    print('PCA score: {}'.format(pca.score(Xh)))
    print('Explained variance ratio: {}'.format(np.sum(pca.explained_variance_ratio_)))

    # Show the explained variance plot
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(np.arange(0, len(ev), 1), ev)
    ax.set_xlabel('Components')
    ax.set_ylabel('Explained variance ratio')

    plt.show()



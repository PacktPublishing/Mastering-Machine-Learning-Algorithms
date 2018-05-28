import matplotlib.pyplot as plt
import numpy as np

from sklearn.covariance import LedoitWolf
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import FactorAnalysis

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

    # Perform Factor analysis
    fa = FactorAnalysis(n_components=64, random_state=1000)
    fah = FactorAnalysis(n_components=64, random_state=1000)

    Xfa = fa.fit_transform(X)
    Xfah = fah.fit_transform(Xh)

    print('Factor analysis score X: {}'.format(fa.score(X)))
    print('Factor analysis score Xh: {}'.format(fah.score(Xh)))

    # Perform Lodoit-Wolf shrinkage
    ldw = LedoitWolf()
    ldwh = LedoitWolf()

    ldw.fit(X)
    ldwh.fit(Xh)

    print('Ledoit-Wolf score X: {}'.format(ldw.score(X)))
    print('Ledoit-Wolf score Xh: {}'.format(ldwh.score(Xh)))

    # Show the components
    fig, ax = plt.subplots(8, 8, figsize=(10, 10))

    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(fah.components_[(i * 8) + j].reshape((28, 28)), cmap='gray')
            ax[i, j].axis('off')

    plt.show()

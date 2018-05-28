import matplotlib.pyplot as plt
import numpy as np

# To install Scikit-Fuzzy (if not already installed): pip install -U scikit-fuzzy
# Further instructions: https://pythonhosted.org/scikit-fuzzy/
from skfuzzy.cluster import cmeans, cmeans_predict

from sklearn.datasets import load_digits


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X_train = digits['data'] / np.max(digits['data'])

    # Perform Fuzzy C-Means
    fc, W, _, _, _, _, pc = cmeans(X_train.T, c=10, m=1.25, error=1e-6, maxiter=10000, seed=1000)

    print('Partition coeffiecient: {}'.format(pc))

    # Plot the centroids
    fig, ax = plt.subplots(1, 10, figsize=(10, 10))

    for i in range(10):
        c = fc[i]
        ax[i].matshow(c.reshape(8, 8) * 255.0, cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

    # Membership degrees of a sample representing the digit '7'
    print('Membership degrees: {}'.format(W[:, 7]))

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(np.arange(10), W[:, 7])
    ax.set_xlabel('Cluster index')
    ax.set_ylabel('Fuzzy membership')
    ax.grid()

    plt.show()

    # Perform a prediction
    new_sample = np.expand_dims(X_train[7], axis=1)
    Wn, _, _, _, _, _ = cmeans_predict(new_sample, cntr_trained=fc, m=1.25, error=1e-6, maxiter=10000, seed=1000)

    print('Membership degrees: {}'.format(Wn.T))


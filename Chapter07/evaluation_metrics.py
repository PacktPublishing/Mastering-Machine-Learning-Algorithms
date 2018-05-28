import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, silhouette_score, silhouette_samples


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X_train = digits['data'] / np.max(digits['data'])

    # Perform K-Means with 10 clusters
    km = KMeans(n_clusters=10, random_state=1000)
    Y = km.fit_predict(X_train)

    print('Homogeneity score: {}'.format(homogeneity_score(digits['target'], Y)))
    print('Completeness score: {}'.format(completeness_score(digits['target'], Y)))
    print('Adjusted Rand score: {}'.format(adjusted_rand_score(digits['target'], Y)))
    print('Silhouette score: {}'.format(silhouette_score(X_train, Y, metric='euclidean')))

    # Plot silhouette plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    nb_clusters = [3, 5, 10, 12]
    mapping = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, n in enumerate(nb_clusters):
        km = KMeans(n_clusters=n, random_state=1000)
        Y = km.fit_predict(X_train)

        silhouette_values = silhouette_samples(X_train, Y)

        ax[mapping[i]].set_xticks([-0.15, 0.0, 0.25, 0.5, 0.75, 1.0])
        ax[mapping[i]].set_yticks([])
        ax[mapping[i]].set_title('%d clusters' % n)
        ax[mapping[i]].set_xlim([-0.15, 1])
        ax[mapping[i]].grid()
        y_lower = 20

        for t in range(n):
            ct_values = silhouette_values[Y == t]
            ct_values.sort()

            y_upper = y_lower + ct_values.shape[0]

            color = cm.Accent(float(t) / n)
            ax[mapping[i]].fill_betweenx(np.arange(y_lower, y_upper), 0, ct_values, facecolor=color, edgecolor=color)

            y_lower = y_upper + 20

    plt.show()
import numpy as np

from scipy.spatial.distance import pdist

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 100
nb_bins = 100


def max_min_mean(p=1.0, d=2):
    Xs = np.random.uniform(0.0, 1.0, size=(nb_bins, nb_samples, d))

    pd_max = np.zeros(shape=(nb_bins,))
    pd_min = np.zeros(shape=(nb_bins,))

    for i in range(nb_bins):
        pd = pdist(Xs[i], metric='minkowski', p=p)
        pd_max[i] = np.max(pd)
        pd_min[i] = np.min(pd)

    return np.mean(pd_max - pd_min)


if __name__ == '__main__':
    print('P=1 -> {}'.format(max_min_mean(p=1.0)))
    print('P=2 -> {}'.format(max_min_mean(p=2.0)))
    print('P=10 -> {}'.format(max_min_mean(p=10.0)))
    print('P=100 -> {}'.format(max_min_mean(p=100.0)))


    print('P=1 -> {}'.format(max_min_mean(p=1.0, d=100)))
    print('P=2 -> {}'.format(max_min_mean(p=2.0, d=100)))
    print('P=10 -> {}'.format(max_min_mean(p=10.0, d=100)))
    print('P=100 -> {}'.format(max_min_mean(p=100.0, d=100)))


    print('P=1 -> {}'.format(max_min_mean(p=1.0, d=1000)))
    print('P=2 -> {}'.format(max_min_mean(p=2.0, d=1000)))
    print('P=10 -> {}'.format(max_min_mean(p=10.0, d=1000)))
    print('P=100 -> {}'.format(max_min_mean(p=100.0, d=1000)))

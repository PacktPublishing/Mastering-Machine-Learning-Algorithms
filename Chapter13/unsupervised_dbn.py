import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# To install the DBN package: pip install git+git://github.com/albertbup/deep-belief-network.git
# Further information: https://github.com/albertbup/deep-belief-network
from dbn.tensorflow import UnsupervisedDBN

from keras.datasets import mnist

from sklearn.manifold import TSNE
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 400


if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (_, _) = mnist.load_data()
    X_train, Y_train = shuffle(X_train, Y_train, random_state=1000)

    width = X_train.shape[1]
    height = X_train.shape[2]

    X = X_train[0:nb_samples].reshape((nb_samples, width * height)).astype(np.float32) / 255.0
    Y = Y_train[0:nb_samples]

    # Train the unsupervised DBN
    unsupervised_dbn = UnsupervisedDBN(hidden_layers_structure=[512, 256, 64],
                                       learning_rate_rbm=0.05,
                                       n_epochs_rbm=100,
                                       batch_size=64,
                                       activation_function='sigmoid')

    X_dbn = unsupervised_dbn.fit_transform(X)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=1000)
    X_tsne = tsne.fit_transform(X_dbn)

    # Show the result
    fig, ax = plt.subplots(figsize=(18, 10))

    colors = [cm.tab10(i) for i in Y]

    for i in range(nb_samples):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, marker='o', s=30)
        ax.annotate('%d' % Y[i], xy=(X_tsne[i, 0] + 1, X_tsne[i, 1] + 1))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()


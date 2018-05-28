import numpy as np

# To install the DBN package: pip install git+git://github.com/albertbup/deep-belief-network.git
# Further information: https://github.com/albertbup/deep-belief-network
from dbn.tensorflow import SupervisedDBNClassification

from sklearn.datasets import fetch_kddcup99
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load and normalize the dataset
    kddcup = fetch_kddcup99(subset='smtp', shuffle=True, random_state=1000)

    ss = StandardScaler()
    X = ss.fit_transform(kddcup['data']).astype(np.float32)

    le = LabelEncoder()
    Y = le.fit_transform(kddcup['target']).astype(np.float32)

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)

    # Train the model
    classifier = SupervisedDBNClassification(hidden_layers_structure=[64, 64],
                                             learning_rate_rbm=0.001,
                                             learning_rate=0.01,
                                             n_epochs_rbm=20,
                                             n_iter_backprop=150,
                                             batch_size=256,
                                             activation_function='relu',
                                             dropout_p=0.25)

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    print('Accuracy score: {}'.format(accuracy_score(Y_test, Y_pred)))


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    X, Y = load_wine(return_X_y=True)

    # Compute the CV scores for different number of estimators
    scores_ne = []

    for ne in range(10, 201, 10):
        adc = AdaBoostClassifier(n_estimators=ne, learning_rate=0.8, random_state=1000)
        scores_ne.append(np.mean(cross_val_score(adc, X, Y, cv=10)))

    # Plot CV scores
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(list(range(10, 201, 10)), scores_ne)
    ax.set_xlabel('Number of estimators')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.grid()
    plt.show()

    # Compute the CV scores for different learning rates
    scores_eta_adc = []

    for eta in np.linspace(0.01, 1.0, 100):
        adc = AdaBoostClassifier(n_estimators=50, learning_rate=eta, random_state=1000)
        scores_eta_adc.append(np.mean(cross_val_score(adc, X, Y, cv=10)))

    # Plot CV scores
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(list(np.linspace(0.01, 1.0, 100)), scores_eta_adc)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.grid()
    plt.show()

    # Perform PCA and Factor Analysis
    scores_pca = []

    for i in range(13, 1, -1):
        if i < 12:
            pca = PCA(n_components=i, random_state=1000)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X

        adc = AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=1000)
        scores_pca.append(np.mean(cross_val_score(adc, X_pca, Y, cv=10)))

    scores_fa = []

    for i in range(13, 1, -1):
        if i < 12:
            fa = FactorAnalysis(n_components=i, random_state=1000)
            X_fa = fa.fit_transform(X)
        else:
            X_fa = X

        adc = AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=1000)
        scores_fa.append(np.mean(cross_val_score(adc, X_fa, Y, cv=10)))

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot([i for i in range(2, X.shape[1] + 1)], scores_fa[::-1], label='Factor Analysis')
    ax.plot([i for i in range(2, X.shape[1] + 1)], scores_pca[::-1], label='PCA')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.legend()
    ax.grid()
    plt.show()
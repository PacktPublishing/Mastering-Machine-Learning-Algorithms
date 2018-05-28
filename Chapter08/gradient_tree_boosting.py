import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    X, Y = load_wine(return_X_y=True)

    # Perform Gradient Tree Boosting with different tree depths
    scores_md = []
    eta = 0.8

    for md in range(2, 13):
        gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=eta, max_depth=md, random_state=1000)
        scores_md.append(np.mean(cross_val_score(gbc, X, Y, cv=10)))

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(list(range(2, 13)), scores_md)
    ax.set_xlabel('Maximum Tree Depth')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.grid()
    plt.show()

    # Perform Gradient Tree Boosting with different learning rates
    scores_eta = []

    for eta in np.linspace(0.01, 1.0, 100):
        gbr = GradientBoostingClassifier(n_estimators=50, learning_rate=eta, max_depth=2, random_state=1000)
        scores_eta.append(np.mean(cross_val_score(gbr, X, Y, cv=10)))

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(np.linspace(0.01, 1.0, 100), scores_eta)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.grid()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=1000, random_state=1000)

    # Perform a CV with 15 folds and a Logistic Regression
    score = cross_val_score(LogisticRegression(), X, Y, cv=15)

    print('Average CV score: {}'.format(np.mean(score)))
    print('CV score variance: {}'.format(np.var(score)))

    # Plot the scores
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(score)
    ax.set_xlabel('Cross-validation fold')
    ax.set_ylabel('Logistic Regression Accuracy')
    ax.grid()
    plt.show()
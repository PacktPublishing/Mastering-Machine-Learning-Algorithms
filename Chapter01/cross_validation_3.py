import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeavePOut


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    data = load_iris()

    p = 3
    lr = LogisticRegression()

    # Perform Leave-P-Out Cross Validation
    lpo_scores = cross_val_score(lr, data['data'], data['target'], cv=LeavePOut(p))
    print('LPO scores (100): {}'.format(lpo_scores[0:100]))
    print('Average LPO score: {}'.format(lpo_scores.mean()))

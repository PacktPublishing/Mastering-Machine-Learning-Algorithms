import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, LeaveOneOut

from sklearn.svm import SVC


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    data = load_digits()

    # Create a polynomial SVM
    svm = SVC(kernel='poly')

    # Perform k-Fold Cross Validation
    skf_scores = cross_val_score(svm, data['data'], data['target'], cv=10)
    print('CV scores: {}'.format(skf_scores))
    print('Average CV score: {}'.format(skf_scores.mean()))

    # Perform Leave-One-Out Cross Validation
    loo_scores = cross_val_score(svm, data['data'], data['target'], cv=LeaveOneOut())
    print('LOO scores (100): {}'.format(loo_scores[0:100]))
    print('Average LOO score: {}'.format(loo_scores.mean()))
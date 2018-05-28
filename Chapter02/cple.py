import numpy as np

from scipy.optimize import fmin_bfgs

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Set random seed for reproducibility
np.random.seed(1000)

nb_unlabeled = 150

# Create a training Logistic Regression instance
lr = LogisticRegression()

# Initialize soft-variables
q0 = np.random.uniform(0, 1, size=nb_unlabeled)

# Vectorized threshold function
trh = np.vectorize(lambda x: 0.0 if x < 0.5 else 1.0)


def weighted_log_loss(yt, p, w=None, eps=1e-15):
    if w is None:
        w_t = np.ones((yt.shape[0], 2))
    else:
        w_t = np.vstack((w, 1.0 - w)).T

    Y_t = np.vstack((1.0 - yt.squeeze(), yt.squeeze())).T
    L_t = np.sum(w_t * Y_t * np.log(np.clip(p, eps, 1.0 - eps)), axis=1)

    return np.mean(L_t)


def build_dataset(q):
    Y_unlabeled = trh(q)

    X_n = np.zeros((nb_samples, nb_dimensions))
    X_n[0:nb_samples - nb_unlabeled] = X[Y.squeeze() != -1]
    X_n[nb_samples - nb_unlabeled:] = X[Y.squeeze() == -1]

    Y_n = np.zeros((nb_samples, 1))
    Y_n[0:nb_samples - nb_unlabeled] = Y[Y.squeeze() != -1]
    Y_n[nb_samples - nb_unlabeled:] = np.expand_dims(Y_unlabeled, axis=1)

    return X_n, Y_n


def log_likelihood(q):
    X_n, Y_n = build_dataset(q)
    Y_soft = trh(q)

    lr.fit(X_n, Y_n.squeeze())

    p_sup = lr.predict_proba(X[Y.squeeze() != -1])
    p_semi = lr.predict_proba(X[Y.squeeze() == -1])

    l_sup = weighted_log_loss(Y[Y.squeeze() != -1], p_sup)
    l_semi = weighted_log_loss(Y_soft, p_semi, q)

    return l_semi - l_sup


if __name__ == '__main__':
    # Load dataset
    X_a, Y_a = load_digits(return_X_y=True)

    # Select the subset containing all 0s and 1s
    X = np.vstack((X_a[Y_a == 0], X_a[Y_a == 1]))
    Y = np.vstack((np.expand_dims(Y_a, axis=1)[Y_a == 0], np.expand_dims(Y_a, axis=1)[Y_a == 1]))

    nb_samples = X.shape[0]
    nb_dimensions = X.shape[1]
    Y_true = np.zeros((nb_unlabeled,))

    # Select nb_unlabeled samples
    unlabeled_idx = np.random.choice(np.arange(0, nb_samples, 1), replace=False, size=nb_unlabeled)
    Y_true = Y[unlabeled_idx].copy()
    Y[unlabeled_idx] = -1

    # Check the CV scores using only a Logistic Regression
    total_cv_scores = cross_val_score(LogisticRegression(), X, Y.squeeze(), cv=10)

    print('CV scores (only Logistic Regression)')
    print(total_cv_scores)

    # Train CPLE
    print('Training CPLE...')
    q_end = fmin_bfgs(f=log_likelihood, x0=q0, maxiter=5000, disp=False)

    # Build the final dataset
    X_n, Y_n = build_dataset(q_end)

    # Check the CV scores using CPLE
    final_semi_cv_scores = cross_val_score(LogisticRegression(), X_n, Y_n.squeeze(), cv=10)

    print('CV scores (CPLE)')
    print(final_semi_cv_scores)




import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize

from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 500
nb_unlabeled = 400


# Create dataset
X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, random_state=1000)
Y[Y==0] = -1
Y[nb_samples - nb_unlabeled:nb_samples] = 0


# Initialize TSVM variables
w = np.random.uniform(-0.1, 0.1, size=X.shape[1])
eta_labeled = np.random.uniform(0.0, 0.1, size=nb_samples - nb_unlabeled)
eta_unlabeled = np.random.uniform(0.0, 0.1, size=nb_unlabeled)
y_unlabeled = np.random.uniform(-1.0, 1.0, size=nb_unlabeled)
b = np.random.uniform(-0.1, 0.1, size=1)

C_labeled = 1.0
C_unlabeled = 10.0


# Stack all variables into a single vector
theta0 = np.hstack((w, eta_labeled, eta_unlabeled, y_unlabeled, b))


def svm_target(theta, Xd, Yd):
    wt = theta[0:2].reshape((Xd.shape[1], 1))

    s_eta_labeled = np.sum(theta[2:2 + nb_samples - nb_unlabeled])
    s_eta_unlabeled = np.sum(theta[2 + nb_samples - nb_unlabeled:2 + nb_samples])

    return (C_labeled * s_eta_labeled) + (C_unlabeled * s_eta_unlabeled) + (0.5 * np.dot(wt.T, wt))


def labeled_constraint(theta, Xd, Yd, idx):
    wt = theta[0:2].reshape((Xd.shape[1], 1))

    c = Yd[idx] * (np.dot(Xd[idx], wt) + theta[-1]) + \
        theta[2:2 + nb_samples - nb_unlabeled][idx] - 1.0

    return (c >= 0)[0]


def unlabeled_constraint(theta, Xd, idx):
    wt = theta[0:2].reshape((Xd.shape[1], 1))

    c = theta[2 + nb_samples:2 + nb_samples + nb_unlabeled][idx - nb_samples + nb_unlabeled] * \
        (np.dot(Xd[idx], wt) + theta[-1]) + \
        theta[2 + nb_samples - nb_unlabeled:2 + nb_samples][idx - nb_samples + nb_unlabeled] - 1.0

    return (c >= 0)[0]


def eta_labeled_constraint(theta, idx):
    return theta[2:2 + nb_samples - nb_unlabeled][idx] >= 0


def eta_unlabeled_constraint(theta, idx):
    return theta[2 + nb_samples - nb_unlabeled:2 + nb_samples][idx - nb_samples + nb_unlabeled] >= 0


def y_constraint(theta, idx):
    return np.power(theta[2 + nb_samples:2 + nb_samples + nb_unlabeled][idx], 2) == 1.0


if __name__ == '__main__':
    # Show the initial dataset
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='#88d7f0', s=100)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='#55ffec', s=100)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='r', marker='x', s=50)

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()

    # Setup all the constraints
    svm_constraints = []

    for i in range(nb_samples - nb_unlabeled):
        svm_constraints.append({
            'type': 'ineq',
            'fun': labeled_constraint,
            'args': (X, Y, i)
        })
        svm_constraints.append({
            'type': 'ineq',
            'fun': eta_labeled_constraint,
            'args': (i,)
        })

    for i in range(nb_samples - nb_unlabeled, nb_samples):
        svm_constraints.append({
            'type': 'ineq',
            'fun': unlabeled_constraint,
            'args': (X, i)
        })
        svm_constraints.append({
            'type': 'ineq',
            'fun': eta_unlabeled_constraint,
            'args': (i,)
        })

    for i in range(nb_unlabeled):
        svm_constraints.append({
            'type': 'eq',
            'fun': y_constraint,
            'args': (i,)
        })

    # Optimize the objective
    print('Optimizing...')
    result = minimize(fun=svm_target,
                      x0=theta0,
                      constraints=svm_constraints,
                      args=(X, Y),
                      method='SLSQP',
                      tol=0.0001,
                      options={'maxiter': 1000})

    # Extract the last parameters
    theta_end = result['x']
    w = theta_end[0:2]
    b = theta_end[-1]

    Xu = X[nb_samples - nb_unlabeled:nb_samples]
    yu = -np.sign(np.dot(Xu, w) + b)

    # Show the final plots
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], color='#88d7f0', marker='s', s=400)
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], color='#55ffec', marker='o', s=400)
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], color='r', marker='x', s=15)

    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].grid()

    ax[1].scatter(X[Y == -1, 0], X[Y == -1, 1], color='#88d7f0', marker='s', s=50)
    ax[1].scatter(X[Y == 1, 0], X[Y == 1, 1], color='#55ffec', marker='o', s=50)

    ax[1].scatter(Xu[yu == -1, 0], Xu[yu == -1, 1], color='#88d7f0', marker='x', s=150)
    ax[1].scatter(Xu[yu == 1, 0], Xu[yu == 1, 1], color='#55ffec', marker='x', s=150)

    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].grid()

    plt.show()
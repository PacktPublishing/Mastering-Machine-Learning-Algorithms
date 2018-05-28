import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    rs = np.random.RandomState(1000)
    X = rs.normal(loc=1.0, scale=(20.0, 1.0), size=(1000, 2))

    w = np.array([30.0, 3.0])
    w0 = w.copy()

    # Plot the initial w position
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.scatter(X[:, 0], X[:, 1], s=20, color='#cccccc')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    plt.ylim([-10, 10])
    ax.grid()
    ax.arrow(0, 0, 30.0, 3.0, head_width=1.0, head_length=2.0, fc='k', ec='k')
    ax.annotate(r'$w_0$', xy=(10.0, 10.0), xycoords='data', xytext=(25.0, 3.5), textcoords='data', size=18)
    plt.show()

    S = np.cov(X.T)

    for i in range(10):
        w += np.dot(S, w)
        w /= np.linalg.norm(w)

    w *= 50.0

    print('Final w: {}'.format(np.round(w, 1)))

    # Plot the final configuration
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[:, 0], X[:, 1], s=20, color='#cccccc')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    ax.arrow(0, 0, w0[0], w0[1], head_width=1.0, head_length=2.0, fc='k', ec='k')
    ax.annotate(r'$w_0$', xy=(1.0, 1.0), xycoords='data', xytext=(w0[0] - 6.0, w0[1] + 0.5), textcoords='data', size=20)

    ax.arrow(0, 0, w[0], w[1], head_width=1.0, head_length=2.0, fc='k', ec='k')
    ax.annotate(r'$w_\infty$', xy=(1.0, 1.0), xycoords='data', xytext=(w[0] - 6.0, w[1] + 0.5), textcoords='data',
                size=20)

    plt.ylim([-10, 10])
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1000)


nb_iterations = 100000
x = 1.0
samples = []


def prior(x):
    return 0.1 * np.exp(-0.1 * x)


def likelihood(x):
    a = np.sqrt(0.2 / (2.0 * np.pi * np.power(x, 3)))
    b = - (0.2 * np.power(x - 1.0, 2)) / (2.0 * x)
    return a * np.exp(b)


def g(x):
    return likelihood(x) * prior(x)


def q(xp):
    return np.random.normal(xp)


if __name__ == '__main__':
    # Main loop
    for i in range(nb_iterations):
        xc = q(x)

        alpha = g(xc) / g(x)
        if np.isnan(alpha):
            continue

        if alpha >= 1:
            samples.append(xc)
            x = xc
        else:
            if np.random.uniform(0.0, 1.0) < alpha:
                samples.append(xc)
                x = xc

    # Generate the histogram
    hist, _ = np.histogram(samples, bins=100)
    hist_p = hist / len(samples)

    # Show the histogram
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(hist_p)
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')

    plt.show()
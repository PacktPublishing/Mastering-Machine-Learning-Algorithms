import numpy as np

# Set random seed for reproducibility
np.random.seed(1000)


N = 4
Nsamples = 50000


def X1_sample(p=0.35):
    return np.random.binomial(1, p)


def X2_sample(p=0.65):
    return np.random.binomial(1, p)


def X3_sample(x1, x2, p1=0.75, p2=0.4):
    if x1 == 1 and x2 == 1:
        return np.random.binomial(1, p1)
    else:
        return np.random.binomial(1, p2)


def X4_sample(x3, p1=0.65, p2=0.5):
    if x3 == 1:
        return np.random.binomial(1, p1)
    else:
        return np.random.binomial(1, p2)


if __name__ == '__main__':
    # Initialize the sample frequency dictionary
    Fsamples = {}

    # Main loop
    for t in range(Nsamples):
        x1 = X1_sample()
        x2 = X2_sample()
        x3 = X3_sample(x1, x2)
        x4 = X4_sample(x3)

        sample = (x1, x2, x3, x4)

        if sample in Fsamples:
            Fsamples[sample] += 1
        else:
            Fsamples[sample] = 1

    # Compute the probabilities
    samples = np.array(list(Fsamples.keys()), dtype=np.bool_)
    probabilities = np.array(list(Fsamples.values()), dtype=np.float64) / Nsamples

    # Show the probabilities
    for i in range(len(samples)):
        print('P{} = {}'.format(samples[i], probabilities[i]))

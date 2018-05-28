import numpy as np

# Set random seed for reproducibility
np.random.seed(1000)


def theta(theta_prev, z1=50.0, x3=10.0):
    num = (8.0 * z1 * theta_prev) + (4.0 * x3 * (12.0 - theta_prev))
    den = (z1 + x3) * (12.0 - theta_prev)
    return num / den


if __name__ == '__main__':
    theta_v = 0.01

    for i in range(1000):
        theta_v = theta(theta_v)

    # Final theta
    print(theta_v)

    # Probability vector
    p = [theta_v / 6.0, (1 - (theta_v / 4.0)), theta_v / 12.0]

    print(p)
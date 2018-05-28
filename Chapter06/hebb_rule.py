# -*- coding: utf-8 -*-
import numpy as np

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Define initial vectors
    w = np.array([1.0, 0.2])
    x = np.array([0.1, 0.5])
    alpha = 0.0

    print('Initial w: {}'.format(w))

    # Perform the iterations
    for i in range(50):
        y = np.dot(w, x.T)
        w += x * y
        alpha = np.arccos(np.dot(w, x.T) / (np.linalg.norm(w) * np.linalg.norm(x)))

    print('Final w: {}'.format(w))
    print('Final alpha: {}'.format(alpha * 180.0 / np.pi))

    # Repeat the test with alpha greater than 90°
    w = np.array([1.0, -1.0])

    print('Initial w: {}'.format(w))

    # Perform the iterations
    for i in range(50):
        y = np.dot(w, x.T)
        w += x * y
        alpha = np.arccos(np.dot(w, x.T) / (np.linalg.norm(w) * np.linalg.norm(x)))

    print('Final w: {}'.format(w))
    print('Final alpha: {}'.format(alpha * 180.0 / np.pi))
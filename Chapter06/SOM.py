import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(1000)


nb_iterations = 1000
nb_startup_iterations = 500
pattern_length = 64 * 64
pattern_width = pattern_height = 64
eta0 = 1.0
sigma0 = 3.0
tau = 100.0
matrix_side = 5

W = np.random.normal(0, 0.1, size=(matrix_side, matrix_side, pattern_length))
precomputed_distances = np.zeros((matrix_side, matrix_side, matrix_side, matrix_side))


def winning_unit(xt):
    global W
    distances = np.linalg.norm(W - xt, ord=2, axis=2)
    max_activation_unit = np.argmax(distances)
    return int(np.floor(max_activation_unit / matrix_side)), max_activation_unit % matrix_side


def eta(t):
    return eta0 * np.exp(-float(t) / tau)


def sigma(t):
    return float(sigma0) * np.exp(-float(t) / tau)


def distance_matrix(xt, yt, sigmat):
    global precomputed_distances
    dm = precomputed_distances[xt, yt, :, :]
    de = 2.0 * np.power(sigmat, 2)
    return np.exp(-dm / de)


if __name__ == '__main__':
    # Load the dataset
    faces = fetch_olivetti_faces(shuffle=True)
    Xcomplete = faces['data'].astype(np.float64) / np.max(faces['data'])
    np.random.shuffle(Xcomplete)
    X = Xcomplete[0:100]

    # Pre-compute distances
    for i in range(matrix_side):
        for j in range(matrix_side):
            for k in range(matrix_side):
                for t in range(matrix_side):
                    precomputed_distances[i, j, k, t] = np.power(float(i) - float(k), 2) + np.power(float(j) - float(t), 2)

    # Perform training cycle
    sequence = np.arange(0, X.shape[0])
    t = 0

    for e in range(nb_iterations):
        np.random.shuffle(sequence)
        t += 1

        if e < nb_startup_iterations:
            etat = eta(t)
            sigmat = sigma(t)
        else:
            etat = 0.2
            sigmat = 1.0

        for n in sequence:
            x_sample = X[n]

            xw, yw = winning_unit(x_sample)
            dm = distance_matrix(xw, yw, sigmat)

            dW = etat * np.expand_dims(dm, axis=2) * (x_sample - W)
            W += dW

        W /= np.linalg.norm(W, axis=2).reshape((matrix_side, matrix_side, 1))

        if e > 0 and e % 100 == 0:
            print(t)

    # Show the final W matrix
    sc = StandardScaler(with_std=False)
    Ws = sc.fit_transform(W.reshape((matrix_side * matrix_side, pattern_length)))

    matrix_w = np.zeros((matrix_side * pattern_height, matrix_side * pattern_width))

    Ws = Ws.reshape((matrix_side, matrix_side, pattern_length))

    for i in range(matrix_side):
        for j in range(matrix_side):
            matrix_w[i * pattern_height:i * pattern_height + pattern_height,
                     j * pattern_height:j * pattern_height + pattern_width] = W[i, j].reshape((pattern_height, pattern_width)) * 255.0

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.matshow(matrix_w.tolist(), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
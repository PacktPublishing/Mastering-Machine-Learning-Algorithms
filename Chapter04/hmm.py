import numpy as np

from hmmlearn import hmm

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create a Multinomial HMM
    hmm_model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=1000)

    # Define a list of observations
    observations = np.array([[0], [1], [1], [0], [1], [1], [1], [0], [1],
                             [0], [0], [0], [1], [0], [1], [1], [0], [1],
                             [0], [0], [1], [0], [1], [0], [0], [0], [1],
                             [0], [1], [0], [1], [0], [0], [0], [0], [0]], dtype=np.int32)

    # Fit the model using the Forward-Backward algorithm
    hmm_model.fit(observations)

    # Check the convergence
    print('Converged: {}'.format(hmm_model.monitor_.converged))

    # Print the transition probability matrix
    print('\nTransition probability matrix:')
    print(hmm_model.transmat_)

    # Create a test sequence
    sequence = np.array([[1], [1], [1], [0], [1], [1], [1], [0], [1],
                         [0], [1], [0], [1], [0], [1], [1], [0], [1],
                         [1], [0], [1], [0], [1], [0], [1], [0], [1],
                         [1], [1], [0], [0], [1], [1], [0], [1], [1]], dtype=np.int32)

    # Find the the most likely hidden states using the Viterbi algorithm
    lp, hs = hmm_model.decode(sequence)

    print('\nMost likely hidden state sequence:')
    print(hs)

    print('\nLog-propability:')
    print(lp)

    # Compute the posterior probabilities
    pp = hmm_model.predict_proba(sequence)

    print('\nPosterior probabilities:')
    print(pp)

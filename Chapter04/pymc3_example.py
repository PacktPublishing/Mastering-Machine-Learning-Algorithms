import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3.distributions.continuous as pmc
import pymc3.distributions.discrete as pmd
import pymc3.math as pmm

# PyMC 3 Installation instructions (https://github.com/pymc-devs/pymc3)
# Pip: pip install pymc3
# Conda: conda install -c conda-forge pymc3
#
# In case of issues with h5py with an Anaconda distribution, please update the package: conda install h5py

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 500


if __name__ == '__main__':
    # Create a PyMC3 model
    model = pm.Model()

    # Define the model structure
    with model:
        passenger_onboarding = pmc.Wald('Passenger Onboarding', mu=0.5, lam=0.2)
        refueling = pmc.Wald('Refueling', mu=0.25, lam=0.5)
        departure_traffic_delay = pmc.Wald('Departure Traffic Delay', mu=0.1, lam=0.2)

        departure_time = pm.Deterministic('Departure Time',
                                          12.0 + departure_traffic_delay +
                                          pmm.switch(passenger_onboarding >= refueling,
                                                     passenger_onboarding,
                                                     refueling))

        rough_weather = pmd.Bernoulli('Rough Weather', p=0.35)

        flight_time = pmc.Exponential('Flight Time', lam=0.5 - (0.1 * rough_weather))
        arrival_traffic_delay = pmc.Wald('Arrival Traffic Delay', mu=0.1, lam=0.2)

        arrival_time = pm.Deterministic('Arrival time',
                                        departure_time +
                                        flight_time +
                                        arrival_traffic_delay)

    # Sample from the model
    # On Windows with Anaconda 3.5 there can be an issue with joblib, therefore I recommend to set n_jobs=1
    with model:
        samples = pm.sample(draws=nb_samples, njobs=1, random_seed=1000)

    # Plot the summary
    pm.summary(samples)

    # Show the diagrams
    fig, ax = plt.subplots(8, 2, figsize=(14, 18))

    pm.traceplot(samples, ax=ax)

    for i in range(8):
        for j in range(2):
            ax[i, j].grid()

    ax[2, 0].set_xlim([0.05, 1.0])
    ax[3, 0].set_xlim([0.05, 0.4])
    ax[4, 0].set_xlim([12, 16])
    ax[5, 0].set_xlim([0, 10])
    ax[6, 0].set_xlim([0.05, 0.4])
    ax[7, 0].set_xlim([14, 20])

    plt.show()


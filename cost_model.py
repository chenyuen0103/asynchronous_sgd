import numpy as np
import random
from scipy.stats import weibull_min, lognorm

random.seed(0)
np.random.seed(0)

def generate_communication_cost(mu_m, sigma_m, n):
    ''' cost of one-way communication between server and workers'''
    s = np.random.lognormal(mu_m, sigma_m, n)
    return s


def generate_computation_cost(mu, sigma, n):
    ''' cost of gradient computation at workers'''
    s = np.random.lognormal(mu, sigma, n)
    return s



def simulation_cost_model(n, mu_m, sigma_m, mu, sigma):
    ''' simulate the cost of communication and computation for n workers'''
    communication_cost = generate_communication_cost(mu_m, sigma_m, n)
    computation_cost = generate_computation_cost(mu, sigma, n)
    return communication_cost, computation_cost


def simulate_communication_costs(n_workers, distribution="normal", **kwargs):
    """
    Simulates communication costs for a given number of workers.

    Args:
        n_workers (int): Number of workers.
        distribution (str): Type of distribution. Options: 'normal', 'exponential', 'gamma', 'empirical'.
        **kwargs: Distribution parameters (e.g., mean, std for normal, shape and scale for gamma).

    Returns:
        list: Simulated communication costs for each worker.
    """

    if distribution == "normal":
        mean = kwargs.get("mean", 5.0)  # Default mean
        std = kwargs.get("std", 1.0)   # Default standard deviation
        return np.random.normal(mean, std, n_workers).tolist()

    elif distribution == "exponential":
        scale = kwargs.get('scale', 1.0)  # Default scale
        return np.random.exponential(scale, n_workers).tolist()

    elif distribution == "gamma":
        shape = kwargs.get('shape', 2.0)
        scale = kwargs.get('scale', 1.0)
        return np.random.gamma(shape, scale, n_workers).tolist()


    elif distribution == "lognormal":
        sigma = kwargs.get('sigma', 0.5)  # Default sigma
        scale = kwargs.get('scale', np.exp(1.0))  # Default scale, equivalent to mean = 1.0
        return lognorm.rvs(sigma, scale=scale, size=n_workers).tolist()

    elif distribution == "weibull":
        shape = kwargs.get('shape', 2.0)  # Default shape (k)
        scale = kwargs.get('scale', 5.0)  # Default scale (lambda)
        return weibull_min.rvs(shape, loc=0, scale=scale, size=n_workers).tolist()

    elif distribution == "empirical":
        assert "observed_costs" in kwargs, "Must provide a list of observed costs."
        observed_costs = kwargs["observed_costs"]
        # Assuming you have a pre-loaded list 'observed_costs'
        return random.choices(observed_costs, k=n_workers)
    else:
        raise ValueError("Unsupported distribution type.")


def simulate_computation_costs(n_workers, distribution="normal", **kwargs):
    if distribution == "normal":
        mean = kwargs.get("mean", 5.0)  # Default mean
        std = kwargs.get("std", 1.0)   # Default standard deviation
        return np.random.normal(mean, std, n_workers).tolist()

    elif distribution == "exponential":
        scale = kwargs.get('scale', 1.0)  # Default scale
        return np.random.exponential(scale, n_workers).tolist()

    elif distribution == "gamma":
        shape = kwargs.get('shape', 2.0)
        scale = kwargs.get('scale', 1.0)
        return np.random.gamma(shape, scale, n_workers).tolist()


    elif distribution == "lognormal":
        sigma = kwargs.get('sigma', 0.5)  # Default sigma
        scale = kwargs.get('scale', np.exp(1.0))  # Default scale, equivalent to mean = 1.0
        return lognorm.rvs(sigma, scale=scale, size=n_workers).tolist()

    elif distribution == "weibull":
        shape = kwargs.get('shape', 2.0)  # Default shape (k)
        scale = kwargs.get('scale', 5.0)  # Default scale (lambda)
        return weibull_min.rvs(shape, loc=0, scale=scale, size=n_workers).tolist()

    elif distribution == "empirical":
        assert "observed_costs" in kwargs, "Must provide a list of observed costs."
        observed_costs = kwargs["observed_costs"]
        # Assuming you have a pre-loaded list 'observed_costs'
        return random.choices(observed_costs, k=n_workers)
    else:
        raise ValueError("Unsupported distribution type.")
    pass

def patitions(tau, tau_m, S):
    '''partition S data samples into len(tau) workers
    Args:
    tau: list of size len(tau) - 1, the cost of computation at each worker
    tau_m: list of size len(tau) - 1, the cost of communication at each worker
    '''
    b = np.zeros(len(tau))
    inv_sum = sum([1/tau[i] for i in range(len(tau_m))])
    comm_over_comp = sum([tau_m[i]/tau[i] for i in range(len(tau_m))])
    for j in range(len(tau_m)):
        b[j] = (S + 2 * comm_over_comp) / (tau[j] * inv_sum) - 2 * tau_m[j]/tau[j]

    # round b_j to integer, and sum of b_j is equal to S
    b = np.round(b)
    b[-1] = S - sum(b[:-1])
    assert np.allclose(sum(b), S)
    return b

def main():
    n_workers = 10
    # Parameters for communication cost (Log-normal)
    mu_m_comm, sigma_m_comm = 0, 0.25  # Example parameters for communication cost
    # Parameters for computation cost (Log-normal)
    mu_comp, sigma_comp = 0, 0.25  # Example parameters for computation cost

    # Simulate costs
    comm_costs = simulate_communication_costs(n_workers, distribution="lognormal", sigma=sigma_m_comm, scale=np.exp(mu_m_comm))
    comp_costs = simulate_computation_costs(n_workers, distribution="lognormal", sigma=sigma_comp, scale=np.exp(mu_comp))

    print("Communication Costs:", comm_costs)
    print("Computation Costs:", comp_costs)

    # Simulate partitioning
    tau = comp_costs
    tau_m = comm_costs
    S = 1000
    b = patitions(tau, tau_m, S)
    print("Partitions:", b)


if __name__ == "__main__":
    main()
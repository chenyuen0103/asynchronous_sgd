import numpy as np
import random
import pulp
import itertools
import time
from scipy.stats import weibull_min, lognorm

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import queue

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

def naive_partitions(tau, tau_m, S):
    '''partition S data samples into len(tau) workers
    Args:
    tau: list of size len(tau) - 1, the cost of computation at each worker
    tau_m: list of size len(tau) - 1, the cost of communication at each worker
    '''
    start_time = time.time()
    b = np.zeros(len(tau))
    inv_sum = sum([1/tau[i] for i in range(len(tau_m))])
    comm_over_comp = sum([tau_m[i]/tau[i] for i in range(len(tau_m))])
    for j in range(len(tau_m)):
        b[j] = (S + 2 * comm_over_comp) / (tau[j] * inv_sum) - 2 * tau_m[j]/tau[j]

    # round b_j to integer, and sum of b_j is equal to S
    b = np.round(b)
    b[-1] = S - sum(b[:-1])
    assert np.allclose(sum(b), S)
    costs = [tau[i] * b[i] + 2*tau_m[i] for i in range(len(tau_m))]
    end = time.time()
    return b, costs, end - start_time



def exact_partitions_ip(tau, tau_m, S):
    start_time = time.time()
    n = len(tau)

    # Create the optimization model
    model = pulp.LpProblem("Exact Partitions", pulp.LpMinimize)

    # Decision variables
    b = [pulp.LpVariable(f"b[{i}]", lowBound=0, cat='Integer') for i in range(n)]

    # Auxiliary variable
    z = pulp.LpVariable("z", lowBound=0)

    # Objective function
    model += z

    # Constraints
    model += pulp.lpSum(b) == S  # Sum constraint
    for i in range(n):
        for j in range(n):
            model += tau[i] * b[i] + 2 * tau_m[i] - tau[j] * b[j] - 2 * tau_m[j] <= z
            model += -(tau[i] * b[i] + 2 * tau_m[i] - tau[j] * b[j] - 2 * tau_m[j]) <= z

    # Solve
    model.solve()

    if pulp.LpStatus[model.status] == 'Optimal':
        optimal_partition = [int(var.value()) for var in b]
        costs = [tau[i] * optimal_partition[i] + 2 * tau_m[i] for i in range(n)]
        end = time.time() # Put results on the queue
        return optimal_partition, costs, end - start_time  # Then return

    else:
        print("No optimal solution found.") # Indicate failure
        return None, None, None




# Plotting functions
def plot_wait_time(df, n_workers):
    df[df['n_workers'] == n_workers].plot(x='S', y=['naive_max_wait', 'exact_max_wait'], kind='line')
    plt.xlabel('S')
    plt.ylabel('Max Wait Time')
    plt.title(f'Wait Time vs. S (n_workers = {n_workers})')
    plt.show()

def plot_partitioning_time(df, n_workers):
    df[df['n_workers'] == n_workers].plot(x='S', y=['naive_time', 'exact_time'], kind='line')
    plt.xlabel('S')
    plt.ylabel('Partitioning Time')
    plt.title(f'Partitioning Time vs. S (n_workers = {n_workers})')
    plt.show()



def main():
    n_workers = 100
    # Parameters for communication cost (Log-normal)
    mu_m_comm, sigma_m_comm = 0, 0.25  # Example parameters for communication cost
    # Parameters for computation cost (Log-normal)
    mu_comp, sigma_comp = 0, 0.25  # Example parameters for computation cost

    results = pd.DataFrame(columns=['S', 'n_workers', 'naive_max_wait', 'naive_partition_time', 'exact_max_wait', 'exact_partition_time'])

    for S in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:  # Example S values
        for n_workers in [10, 20, 50, 100]:  # Example worker counts
            print(f"Running for S = {S} and n_workers = {n_workers}")
            # Simulate communication and computation costs
            # Simulate costs
            tau_m = simulate_communication_costs(n_workers, distribution="lognormal", sigma=sigma_m_comm,
                                                      scale=np.exp(mu_m_comm))
            tau = simulate_computation_costs(n_workers, distribution="lognormal", sigma=sigma_comp,
                                                    scale=np.exp(mu_comp))


            print("Computation Costs:", tau)
            print("Communication Costs:", tau_m)

            # Naive Partitions
            b, cost, naive_physic_time = naive_partitions(tau, tau_m, S)


            # Exact Partitions
            # terminate the function of waiting for the result after 500 seconds
            b_exact, cost_exact, exact_physic_time = exact_partitions_ip(tau, tau_m, S)
            #append row to the dataframe
            result = {'S': S,
                      'n_workers': n_workers,
                      'naive_max_wait': max(cost) - min(cost),
                      'naive_partition_time': naive_physic_time,
                      'exact_max_wait': max(cost_exact) - min(cost_exact),
                      'exact_partition_time': exact_physic_time}
            # Add this row to the DataFrame


            # Insert the new row using loc
            results.loc[len(results)] = result


            results.to_csv('./partition_results.csv', index=False)
            print("Results:", result)

    # Example plotting usage
    plot_wait_time(results, 100)  # Plot with n_workers = 100
    plot_partitioning_time(results, 50)  # Plot with n_workers = 50

    # Correlations
    for metric in ['naive_max_wait', 'naive_time', 'exact_max_wait', 'exact_time']:
        print(f"Correlation between S and {metric}: {results['S'].corr(results[metric])}")
        print(f"Correlation between n_workers and {metric}: {results['n_workers'].corr(results[metric])}")


if __name__ == "__main__":
    main()
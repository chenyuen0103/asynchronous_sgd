import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns
import time
from configs.exp_config import config
from src.utils.line_reg_generator import DataGenerator
# from server import ParameterServer
# from worker import DataWorker

from distributed_sys import ParameterServer, DataWorker

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (9, 6)


def run(config, data_generator):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Extract configuration variables
    num_workers = config["num_workers"]
    lr = config["lr"]
    asynchronous = config["asynchronous"]
    iterations = config["iterations"]
    it_check = config["it_check"]
    batch_size = config["batch_size"]
    max_seed = config.get("max_seed", 424242)  # Providing a default value if not in configs
    delay_adaptive = config.get("delay_adaptive", False)  # Providing a default value if not in configs
    lr_decay = config.get("lr_decay", 0)  # Providing a default value if not in configs

    # Initialize ParameterServer with corrected parameters
    ps = ParameterServer.remote(config["dim"], lr, asynchronous)
    worker_updates = [0 for i in range(num_workers)]
    # Correction: Use a proper random number generator and extract seeds correctly
    rng = np.random.default_rng(config.get("seed", 42))
    seeds_workers = [rng.integers(max_seed) for _ in range(num_workers)]

    # Initialize workers with corrected reference to data_generator
    workers = [DataWorker.remote(data_generator, lr, batch_size, seeds_workers[i], comp_cost=1e-9, comm_cost=1e-6) for i
               in range(num_workers)]

    # Initialization and training loop remain largely the same
    x = ps.get_x.remote()
    if asynchronous:
        gradients = {}
        worker_last_it = [0 for _ in range(num_workers)]
        worker_id_to_num = {}
        for e, worker in enumerate(workers):
            gradients[worker.compute_gradients.remote(x)] = worker
            worker_id_to_num[worker] = e

    losses = []
    its = []
    ts = []
    delays = []
    t0 = time.perf_counter()
    delay = 0
    trace = []
    grads_per_it = 1 if asynchronous else num_workers

    for it in range(iterations * (num_workers if asynchronous else 1)):
        n_grads = it * grads_per_it
        if asynchronous:
            ready_gradient_list, _ = ray.wait(list(gradients))
            ready_gradient_id = ready_gradient_list[-1]
            worker = gradients.pop(ready_gradient_id)

            # Compute and apply gradients.
            gradients[worker.compute_gradients.remote(x)] = worker
            worker_num = worker_id_to_num[worker]
            delay = it - worker_last_it[worker_num]
            if delay_adaptive:
                lr_new = lr * num_workers / max(num_workers, delay)
                ps.update_lr.remote(lr_new=lr_new)
            x = ps.apply_gradients.remote(grad=ready_gradient_id)
            worker_last_it[worker_num] = it
            worker_updates[worker_num] += 1
        else:
            gradients = [
                worker.compute_gradients.remote(x) for worker in workers
            ]
            # Calculate update after all gradients are available.
            x = ps.apply_gradients.remote(None, *gradients)

        if it % it_check == 0 or (not asynchronous and it % (max(it_check // num_workers, 1)) == 0):
            # Evaluate the current model.
            if not asynchronous:
                print("Save at: ", it)
            x = ray.get(ps.get_x.remote())
            # current_loss = data_generator.evaluate(x)
            # losses.append(current_loss)
            trace.append(x.copy())
            its.append(it)
            ts.append(time.perf_counter() - t0)
            # print(f"Iteration {it}, Loss: {current_loss}")

        lr_new = lr / (1 + lr_decay * n_grads)
        ps.update_lr.remote(lr_new=lr_new)
        t = time.perf_counter()
        if asynchronous:
            delays.append(delay)
    ray.shutdown()
    return np.asarray(its), np.asarray(ts), np.asarray([data_generator.evaluate(x) for x in trace]),  np.asarray(delays)


def run_training(config):
    ray.init(ignore_reinit_error=True)
    num_workers = config["num_workers"]  # Define num_workers here

    # Initialize DataGenerator
    data_generator = DataGenerator(n_data=config["n_data"], dim=config["dim"], noise_scale=config["noise_scale"])

    # Compute least squares solution
    A, b = data_generator.A, data_generator.b
    x_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    f_min = data_generator.evaluate(x_opt)

    # Training configurations for different experiments
    experiments = [
        {"lr": 0.19, "asynchronous": False, "label": "Minibatch SGD"},
        {"lr": 0.43, "asynchronous": True, "label": "Asynchronous SGD"},
        {"lr": 0.43, "asynchronous": True, "delay_adaptive": True, "label": "Delay-Adaptive AsySGD"},
    ]

    results = []  # Store results for plotting

    # Run experiments and collect results
    for exp in experiments:
        updated_config = {**config, **exp}  # Merge dictionaries
        its, ts, losses, delays = run(updated_config, data_generator)  # Ensure 'run' can accept and use updated_config correctly
        results.append((its, losses - f_min, exp["label"]))  # Store results for later plotting

    ray.shutdown()

    # Plot all results
    # plt.figure(figsize=(10, 6))
    # for its, adjusted_losses, label in results:
    #     plt.plot(its, adjusted_losses, label=label)
    for its, adjusted_losses, label in results:
        if label == 'Minibatch SGD':
            plt.plot(its * num_workers, adjusted_losses, label=label)  # Assuming 1 gradient per iteration for Minibatch SGD
        else:
            plt.plot(its , adjusted_losses, label=label)  # Adjust for total gradients computed

    plt.yscale('log')
    plt.xlabel('Number of gradients')
    plt.ylabel(r'$F(\mathbf{x}) - F^*$')
    plt.title('Comparison of Training Methods')
    plt.legend()
    plt.show()
    # Optionally, save the figure to a file
    # plt.savefig('training_methods_comparison.pdf', bbox_inches='tight')

    ray.shutdown()


if __name__ == "__main__":
    run_training(config)

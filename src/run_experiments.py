import ray
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.utils.data_generator import LinReg_DataGenerator, LogReg_DataGenerator
from configs.exp_config import config
from distributed_sys import DistributedSystem
import pandas as pd
import matplotlib.pyplot as plt


def collect_results(results, output_dir='results', plot=True, save_to_csv=True):
    """
    Collects, saves, and optionally plots the results from multiple experiment runs.

    Parameters:
    - results: List of tuples, where each tuple contains (its, losses, label) from one experiment.
    - output_dir: Directory to save output files.
    - plot: Boolean indicating whether to plot the results.
    - save_to_csv: Boolean indicating whether to save the results to a CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare DataFrame
    all_results = []
    for its, losses, label in results:
        for it, loss in zip(its, losses):
            all_results.append({'Iteration': it, 'Loss': loss, 'Label': label})
    df_results = pd.DataFrame(all_results)

    # Save to CSV
    if save_to_csv:
        csv_path = os.path.join(output_dir, 'experiment_results.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    # Plot results
    if plot:
        plt.figure(figsize=(10, 6))
        for label in df_results['Label'].unique():
            subset = df_results[df_results['Label'] == label]
            plt.plot(subset['Iteration'], subset['Loss'], label=label)

        plt.yscale('log')
        plt.xlabel('Number of gradients')
        plt.ylabel(r'$F(\mathbf{x}) - F^*$')
        plt.title('Comparison of Training Methods')
        plt.legend()
        # Optionally, save the figure to a file
        fig_path = os.path.join(output_dir, 'training_methods_comparison.pdf')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {fig_path}")

def main():
    ray.init(ignore_reinit_error=True)
    data_generator = LinReg_DataGenerator(n_data=config["n_data"], dim=config["dim"], noise_scale=config["noise_scale"])
    # data_generator = LogReg_DataGenerator(n_data=config["n_data"], dim=config["dim"], noise_scale=config["noise_scale"])
    x_opt, _, _, _ = np.linalg.lstsq(data_generator.A, data_generator.b, rcond=None)
    f_min = data_generator.evaluate(x_opt)



    distributed_system = DistributedSystem(config=config, data_generator=data_generator)

    experiments = [
        {"lr": 0.19, "asynchronous": False, "label": "Minibatch SGD"},
        {"lr": 0.19, "asynchronous": False, "cost_aware": True, "label": "Cost-Aware Minibatch SGD"},
        # {"lr": 0.43, "asynchronous": True, "label": "Asynchronous SGD"},
        # {"lr": 0.43, "asynchronous": True, "delay_adaptive": True, "label": "Delay-Adaptive AsySGD"},
    ]


    results = []
    # Run experiments and collect results
    for exp in experiments:
        if config["batch_size"] > config["n_data"]:
            exp["batch_size"] = config["n_data"]

        updated_config = {**config, **exp}
        its, ts, losses, delays, time_stamp = distributed_system.run_experiment(config=updated_config) # Ensure 'run' can accept and use updated_config correctly
        results.append((its, losses - f_min, time_stamp, exp["label"]))  # Store results for later plotting

    for its, adjusted_losses, time_stamp, label in results:
        if 'Minibatch SGD' in label:
            plt.plot(its * config['num_workers'], adjusted_losses, label=label)  # Assuming 1 gradient per iteration for Minibatch SGD
        else:
            plt.plot(its, adjusted_losses, label=label)  # Adjust for total gradients computed

    plt.yscale('log')
    plt.xlabel('Number of gradients')
    plt.ylabel(r'$F(\mathbf{x}) - F^*$')
    plt.title(r'$F(\mathbf{x}) - F^*$ against Number of Gradients')
    plt.legend()
    # Optionally, save the figure to a file
    plt.savefig('../data/num_grad_small_var.png')
    plt.show()


    for its, adjusted_losses,  time_stamp, label in results:
        if label == 'Minibatch SGD':
            plt.plot(time_stamp, adjusted_losses, label=label)  # Assuming 1 gradient per iteration for Minibatch SGD
        else:
            plt.plot(time_stamp, adjusted_losses, label=label)  # Adjust for total gradients computed

    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$F(\mathbf{x}) - F^*$')
    plt.title(r'$F(\mathbf{x}) - F^*$ against Time')
    plt.legend()
    plt.savefig('../data/time_small_var.png')
    plt.show()
    ray.shutdown()
    print("Done")


if __name__ == "__main__":
    main()

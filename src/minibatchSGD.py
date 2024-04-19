import numpy as np
# from utils.data_generator import LinReg_DataGenerator
from utils.data_generator import LogReg_DataGenerator
import matplotlib.pyplot as plt

# Run minibatch SGD

class PerformanceMetrics:
    def __init__(self):
        self.lrs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.gds = []
        self.num_grads = []
        self.batch_sizes = []

    def update(self, lr, train_loss, train_acc, val_loss, val_acc, gd, num_grad, batch_size = None):
        self.lrs.append(lr)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.gds.append(gd)
        self.num_grads.append(num_grad)
        if batch_size is not None:
            self.batch_sizes.append(batch_size)

    def get_results(self):
        return {
            "lrs": self.lrs,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "gds": self.gds,
            "num_grads": self.num_grads,
            "batch_sizes": self.batch_sizes
        }


def gradient_diversity(grad, grad_norm):
    gd = grad_norm / np.linalg.norm(grad)**2 + 1e-8
    if gd < 0:
        print("Gradient Diversity is negative")
    return gd


def horvath_grad(data_generator, train_indices,val_indices, num_iters, batch_size, eta, x):
    num_data = len(train_indices)
    metrics = PerformanceMetrics()
    gamma = 1
    for i in range(num_iters):
        grad, grad_norm, _ = data_generator.batch_grad_func(x, batch_size)
        gd = gradient_diversity(grad, grad_norm)
        gamma_max = (2 ** (batch_size/ num_data)) * gamma
        gamma = min(gamma_max, gd)
        x = x - eta * gamma * grad

        # Update the metrics object
        train_loss = data_generator.evaluate(x, train_indices)
        train_acc = data_generator.accuracy(x, train_indices)
        val_loss = data_generator.evaluate(x, val_indices)
        val_acc = data_generator.accuracy(x, val_indices)
        metrics.update(eta * gamma, train_loss, train_acc, val_loss, val_acc, gd / batch_size, (i + 1) * batch_size, batch_size)

        # if i % 100 == 0:
        #     print(f"Iteration {i}, Gamma_max: {gamma_max}, gd: {gd}, Gamma: {gamma}",'lr:', eta * gamma)
    return x, metrics.get_results()


def fixed_batch_size(data_generator, train_indices, val_indices, num_iters, batch_size, lr, x):
    metrics = PerformanceMetrics()  # Instantiate the metrics class
    for i in range(num_iters):
        grad, grad_norm, _ = data_generator.batch_grad_func(x, batch_size, train_indices)
        gd = gradient_diversity(grad, grad_norm) / batch_size  # Assume gradient_diversity exists
        x = x - lr * grad

        # Update the metrics object
        train_loss = data_generator.evaluate(x, train_indices)
        train_acc = data_generator.accuracy(x, train_indices)
        val_loss = data_generator.evaluate(x, val_indices)
        val_acc = data_generator.accuracy(x, val_indices)
        metrics.update(lr, train_loss, train_acc, val_loss, val_acc, gd, (i + 1) * batch_size)

    # Return the optimized parameters and all collected metrics
    return x, metrics.get_results()


def proposed(data_generator, train_indices, val_indices, num_iters, batch_size, lr, x, sweep_portion=0.5):
    num_data = len(train_indices)
    metrics = PerformanceMetrics()
    idx_seen = []
    grad_sum = 0
    grad_norm_sum = 0
    k = 0
    batch_size_init = batch_size
    lr_init = lr
    for i in range(num_iters):
        grad, grad_norm, idx = data_generator.batch_grad_func(x, batch_size, train_indices)
        idx_seen.extend(idx)
        grad_sum += grad
        grad_norm_sum += grad_norm
        x = x - lr * grad
        k += 1
        # if i % 100 == 0:
        #     print(
        #         f"Iteration {i}, Gradient Diversity: {gd}, Batch Size: {batch_size}, gamma_max: {gamma_max}, gamma: {gamma}, lr: {lr}")
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        if len(list(set(idx_seen))) >= sweep_portion * num_data:
            gd = gradient_diversity(grad_sum, grad_norm_sum)
            batch_size = int(min(max(0.1 * k * np.round(gd), batch_size), num_data))
            lr = lr_init * (batch_size * 10 / batch_size_init)
            idx_seen = []
            grad_sum = 0
            grad_norm_sum = 0
            k = 0

            # lr = lr_schedule(i, lr_init)
            # portion = min(portion * 1.5, 1)
        else:
            if len(metrics.gds) > 0:
                gd = metrics.gds[-1]
            else:
                gd = 0

        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        # Evaluate metrics
        train_loss = data_generator.evaluate(x, train_indices)
        train_acc = data_generator.accuracy(x, train_indices)
        val_loss = data_generator.evaluate(x, val_indices)
        val_acc = data_generator.accuracy(x, val_indices)
        metrics.update(lr, train_loss, train_acc, val_loss, val_acc, gd, sum(metrics.batch_sizes) + batch_size, batch_size)


    return x, metrics.get_results()


def full_grad(data_generator, train_indices, val_indices, num_iters, lr, x):
    metrics = PerformanceMetrics()  # Instantiate the metrics class

    for i in range(num_iters):
        # Compute full gradient based on training data
        grad, grad_norm = data_generator.grad_func(x, train_indices)
        gd = gradient_diversity(grad, grad_norm) / len(train_indices)  # Assume gradient_diversity exists

        # Update model parameters
        x = x - lr * grad

        # Calculate metrics for both training and validation sets
        train_loss = data_generator.evaluate(x, train_indices)
        train_acc = data_generator.accuracy(x, train_indices)
        val_loss = data_generator.evaluate(x, val_indices)
        val_acc = data_generator.accuracy(x, val_indices)

        # Update metrics object
        metrics.update(lr, train_loss, train_acc, val_loss, val_acc, gd, (i + 1) * len(train_indices), len(train_indices))

    # Return the optimized parameters and all collected metrics
    return x, metrics.get_results()




def adam(data_generator, train_indices, val_indices, num_iters, batch_size, lr, x, beta1=0.9, beta2=0.999, epsilon=1e-8):
    metrics = PerformanceMetrics()  # Instantiate the metrics class

    m = np.zeros_like(x)  # Initialize first moment vector
    v = np.zeros_like(x)  # Initialize second moment vector

    for i in range(num_iters):
        grad, grad_norm, _  = data_generator.batch_grad_func(x, batch_size, train_indices)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** (i + 1))
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2 ** (i + 1))

        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # Evaluate metrics
        train_loss = data_generator.evaluate(x, train_indices)
        train_acc = data_generator.accuracy(x, train_indices)
        val_loss = data_generator.evaluate(x, val_indices)
        val_acc = data_generator.accuracy(x, val_indices)
        metrics.update(lr * np.linalg.norm(1 / (np.sqrt(v_hat) + epsilon)), train_loss, train_acc, val_loss, val_acc, 0, (i + 1) * batch_size)

    # Return the optimized parameters and all collected metrics
    return x, metrics.get_results()


def lr_schedule(epoch, lr_init, alpha = 0.001):
    if epoch < 10:
        return lr_init
    else:
        return lr_init * (np.exp(- alpha * epoch))



def exp(num_trials=5, n_data=20000, dim=2**9, noise_scale=1e-1, num_iters=15, batch_size=1024, lr_init=1):
    # Define ratios for train, validation, and test sets
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    results = {
        'horvath': [],
        'full': [],
        'batch': [],
        'proposed': [],
        'adam': []
    }

    for _ in range(num_trials):
        logreg = LogReg_DataGenerator(n_data=n_data, dim=dim, noise_scale=noise_scale)

        # Split data into train, validation, and test sets
        indices = np.arange(n_data)
        logreg.rng.shuffle(indices)
        train_size = int(n_data * train_ratio)
        val_size = int(n_data * val_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Initialize parameters for all methods
        x_init = logreg.rng.normal(size=logreg.dim)

        # Run a single trial
        method_results = run_single_trial(logreg, train_indices, val_indices, test_indices, num_iters, batch_size, lr_init, x_init)
        for method, data in method_results.items():
            results[method].append(data)

    # Calculate statistics
    stats = {method: calculate_stats(data) for method, data in results.items()}
    return stats

def calculate_stats(data):
    # Compute mean, standard deviation of losses, accuracies, and learning rates
    stats = {
        'train_loss_mean': np.mean([trial[1]['train_losses'] for trial in data], axis=0),
        'train_loss_std': np.std([trial[1]['train_losses'] for trial in data], axis=0),
        'val_loss_mean': np.mean([trial[1]['val_losses'] for trial in data], axis=0),
        'val_loss_std': np.std([trial[1]['val_losses'] for trial in data], axis=0),
        'train_acc_mean': np.mean([trial[1]['train_accs'] for trial in data], axis=0),
        'train_acc_std': np.std([trial[1]['train_accs'] for trial in data], axis=0),
        'val_acc_mean': np.mean([trial[1]['val_accs'] for trial in data], axis=0),
        'val_acc_std': np.std([trial[1]['val_accs'] for trial in data], axis=0),
        'lr_mean': np.mean([trial[1]['lrs'] for trial in data], axis=0),
        'lr_std': np.std([trial[1]['lrs'] for trial in data], axis=0),
        'num_grads_mean': np.mean([trial[1]['num_grads'] for trial in data], axis=0)
    }
    return stats


def run_single_trial(logreg, train_indices, val_indices, test_indices, num_iters, batch_size, lr_init, x_init):
    # Dictionary mapping method names to function references and their specific arguments
    methods = {
        'adam': (adam, {'data_generator': logreg,'train_indices': train_indices, 'val_indices': val_indices, 'num_iters': num_iters, 'batch_size': batch_size, 'lr': lr_init, 'x': np.copy(x_init)}),
        'horvath': (horvath_grad, {'data_generator': logreg,'train_indices': train_indices, 'val_indices': val_indices, 'num_iters': num_iters, 'batch_size': batch_size, 'eta': lr_init, 'x': np.copy(x_init)}),
        'full': (full_grad, {'data_generator': logreg,'train_indices': train_indices, 'val_indices': val_indices, 'num_iters': num_iters, 'lr': lr_init, 'x': np.copy(x_init)}),
        'batch': (fixed_batch_size, {'data_generator': logreg,'train_indices': train_indices, 'val_indices': val_indices, 'num_iters': num_iters, 'batch_size': batch_size, 'lr': lr_init, 'x': np.copy(x_init)}),
        'proposed': (proposed, {'data_generator': logreg,'train_indices': train_indices, 'val_indices': val_indices, 'num_iters': num_iters, 'batch_size': batch_size, 'lr': lr_init, 'x': np.copy(x_init), 'sweep_portion': 0.5})
    }
    results = {}
    for method_name, (method_func, kwargs) in methods.items():
        results[method_name] = method_func(**kwargs)
    return results

def plot_single(method_dict, save = False, save_path = None):

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(losses, label=method)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(accs, label=method)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(gds, label=method)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Diversity')
    plt.legend()
    plt.show()

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(num_grads, losses, label=method)
    plt.yscale('log')
    plt.xlabel('Number of Gradients')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(num_grads, accs, label=method)
    plt.xlabel('Number of Gradients')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for method in method_dict:
        x, lrs, losses, accs, gds, num_grads = method_dict[method]
        plt.plot(lrs, label=method)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

    if save:
        plt.savefig(save_path)


def plot_exp(stats, save=False, save_path=None):
    # Plotting training and validation losses
    for plot_type in ['train', 'val']:
        for method, data in stats.items():
            iterations = range(len(data[f'{plot_type}_loss_mean']))
            plt.plot(iterations, data[f'{plot_type}_loss_mean'], label=f'{method} {plot_type.capitalize()} Loss')
            plt.fill_between(iterations,
                             data[f'{plot_type}_loss_mean'] - data[f'{plot_type}_loss_std'],
                             data[f'{plot_type}_loss_mean'] + data[f'{plot_type}_loss_std'],
                             alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{plot_type.capitalize()} Loss per Iteration')
        plt.legend()
        plt.show()

    # Plotting training and validation accuracies
    for plot_type in ['train', 'val']:
        for method, data in stats.items():
            iterations = range(len(data[f'{plot_type}_acc_mean']))
            plt.plot(iterations, data[f'{plot_type}_acc_mean'], label=f'{method} {plot_type.capitalize()} Accuracy')
            plt.fill_between(iterations,
                             data[f'{plot_type}_acc_mean'] - data[f'{plot_type}_acc_std'],
                             data[f'{plot_type}_acc_mean'] + data[f'{plot_type}_acc_std'],
                             alpha=0.2)  # Semi-transparent shaded area
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'{plot_type.capitalize()} Accuracy per Iteration')
        plt.legend()
        plt.show()

    # If saving the plots is required
    if save and save_path is not None:
        for fig_type in ['loss', 'acc']:
            for plot_type in ['train', 'val']:
                plt.figure()
                for method, data in stats.items():
                    iterations = range(len(data[f'{plot_type}_{fig_type}_mean']))
                    plt.plot(iterations, data[f'{plot_type}_{fig_type}_mean'], label=f'{method} {plot_type.capitalize()} {fig_type.capitalize()}')
                    plt.fill_between(iterations,
                                     data[f'{plot_type}_{fig_type}_mean'] - data[f'{plot_type}_{fig_type}_std'],
                                     data[f'{plot_type}_{fig_type}_mean'] + data[f'{plot_type}_{fig_type}_std'],
                                     alpha=0.5 if fig_type == 'loss' else 0.2)
                plt.xlabel('Iteration')
                plt.ylabel(fig_type.capitalize())
                plt.title(f'{plot_type.capitalize()} {fig_type.capitalize()} per Iteration')
                plt.legend()
                plt.savefig(f'{save_path}/{plot_type}_{fig_type}.png')
                plt.close()


def main():
    stats = exp(num_trials=3)
    plot_exp(stats)






def sweep_batch_size():
    # Initialize your logreg_DataGenerator with a specific dataset
    dim_list = [2**i for i in range(3, 10)]
    # Experiment parameters
    # batch_sizes = [2, 4, 6, 8, 10, 20, 30, 50, 100, 'full']
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 'full']
    num_iters = 1500  # Number of iterations
    lr = 2 # Learning rate
    for dim, batch_size, in zip(dim_list, batch_sizes):
        logreg = LogReg_DataGenerator(n_data=10000, dim=dim, noise_scale=1e-5)
        pass




if __name__ == '__main__':
    main()

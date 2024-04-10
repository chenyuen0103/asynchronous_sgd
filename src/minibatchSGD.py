import numpy as np
# from utils.data_generator import LinReg_DataGenerator
from utils.data_generator import LogReg_DataGenerator
import matplotlib.pyplot as plt

# Run minibatch SGD



def gradient_diveristy(grad, grad_norm):
    gd = grad_norm / np.linalg.norm(grad)**2
    if gd < 0:
        print("Gradient Diversity is negative")
    return gd


def exp():
    # Initialize Data
    # linreg = LinReg_DataGenerator(n_data=10000, dim=400, noise_scale=1e-4)
    linreg = LogReg_DataGenerator(n_data=1000, dim=2**9, noise_scale=1e-1)
    grad_func = linreg.grad_func
    sgrad_func = linreg.sgrad_func
    batch_grad_func = linreg.batch_grad_func
    evaluate = linreg.evaluate
    accuracy = linreg.accuracy
    num_iters = 1000

    x = linreg.rng.normal(size=linreg.dim)
    x_init = x.copy()

    lr = 0.5
    lr_init = lr
    batch_size = 32
    x_opt, _, _, _ = np.linalg.lstsq(linreg.A, linreg.b, rcond=None)
    f_min = evaluate(x_opt)
    batch_size_init = batch_size
    losses = []
    accs = []
    gds = []
    num_grads = []

    # Fixed batch size
    x = x_init
    for i in range(num_iters):
        grad, grad_norm, _ = batch_grad_func(x, batch_size)
        num_grads.append((i+1) * batch_size)
        gd = gradient_diveristy(grad, grad_norm)
        gds.append(gd/batch_size)
        x = x - lr * grad
        # batch_size = max(int(np.round(gd)),512)
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        losses.append(evaluate(x))
        accs.append(accuracy(x))


    # Adaptive batch size (ideal)
    x = x_init
    losses_adaptive_ideal = []
    gds_adaptive_ideal = []
    gds_full = []
    batch_size_list_ideal = []
    num_grads_adaptive_ideal = []
    accs_adaptive_ideal = []
    batch_size = batch_size_init
    for i in range(num_iters):
        # num_grads_adaptive.append(num_grads_adaptive[-1] + batch_size)
        grad, grad_norm, _ = batch_grad_func(x, batch_size)
        grad_full, grad_norm_full = grad_func(x)
        gd_full = gradient_diveristy(grad_full, grad_norm_full)
        x = x - lr * grad

        gds_full.append(gd_full/linreg.n_data)
        gd = gradient_diveristy(grad, grad_norm)
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}"  )
        gds_adaptive_ideal.append(gd/batch_size)
        # gds_adaptive_ideal.append(gd_full/linreg.n_data)
        batch_size = int(min(max(0.1 * np.round(gd_full), batch_size_init), linreg.n_data))
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        losses_adaptive_ideal.append(evaluate(x))
        accs_adaptive_ideal.append(accuracy(x))
        batch_size_list_ideal.append(batch_size)
        num_grads_adaptive_ideal.append(sum(batch_size_list_ideal))
        if i % 999 == 0 :
            print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd_full}")


    # Adaptive batch size
    x = x_init
    losses_adaptive = []
    gds_adaptive = []
    batch_size_list = []
    num_grads_adaptive = []
    accs_adaptive = []
    batch_size = batch_size_init
    for i in range(num_iters):
        # num_grads_adaptive.append(num_grads_adaptive[-1] + batch_size)
        grad, grad_norm, _ = batch_grad_func(x, batch_size)
        x = x - lr * grad
        gd = gradient_diveristy(grad, grad_norm)

        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        gds_adaptive.append(gd/batch_size)
        batch_size = int(min(max(0.1 * np.round(gd), batch_size_init), linreg.n_data))
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        losses_adaptive.append(evaluate(x))
        accs_adaptive.append(accuracy(x))
        batch_size_list.append(batch_size)
        num_grads_adaptive.append(sum(batch_size_list))


    # Proposed adaptive batch size
    x = x_init
    losses_adaptive_proposed = []
    gds_adaptive_proposed = []
    batch_size_list_proposed = []
    num_grads_adaptive_proposed = []
    accs_adaptive_proposed = []
    idx_seen = []
    grad_sum = 0
    grad_norm_sum = 0
    k = 0
    portion = 0.5
    batch_size = batch_size_init
    if len(gds_adaptive_proposed) < 1:
        gds_adaptive_proposed.append(0)
    for i in range(num_iters):
        grad, grad_norm, idx = batch_grad_func(x, batch_size)
        idx_seen.extend(idx)
        grad_sum += grad
        grad_norm_sum += grad_norm
        x = x - lr * grad
        k += 1
        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
        if len(list(set(idx_seen))) >= portion * linreg.n_data:
            # grad_sum, grad_norm_sum, _ = batch_grad_func(x, linreg.n_data)
            gd = gradient_diveristy(grad_sum, grad_norm_sum)
            gds_adaptive_proposed.append(gd/batch_size)
            print(f"Updating batch size after {k} iterations of batch size {batch_size}")
            # print(f"number of data points seen: {len(list(idx_seen))}")
            # print("Grad_norm_sum", grad_norm_sum)
            print("Number of data points seen", len(list(set(idx_seen))))
            print("Gradient Diversity", gd/batch_size)
            grad_full, grad_norm_full,_ = batch_grad_func(x, len(list(set(idx_seen))))
            # grad_full, grad_norm_full = grad_func(x)
            gd_full = gradient_diveristy(grad_full, grad_norm_full)
            print("Full Gradient Diversity", gd_full/linreg.n_data)


            batch_size = int(min(max(0.1 * k * np.round(gd), batch_size), linreg.n_data))
            idx_seen = []
            grad_sum = 0
            grad_norm_sum = 0
            k = 0
            # portion = min(portion * 1.5, 1)
        else:
            gds_adaptive_proposed.append(gds_adaptive_proposed[-1])

        # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")

        losses_adaptive_proposed.append(evaluate(x))
        accs_adaptive_proposed.append(accuracy(x))
        batch_size_list_proposed.append(batch_size)
        num_grads_adaptive_proposed.append(sum(batch_size_list_proposed))
        # if i % 999 == 0 :
        #     print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")

    # Full Gradient
    x = x_init
    losses_full = []
    accs_full = []
    num_grads_full = []
    # gds_full = []

    for i in range(num_iters):
        grad, grad_norm = grad_func(x)
        x = x - lr * grad
        gd = gradient_diveristy(grad, grad_norm)
        # gds_full.append(gd/linreg.n_data)
        losses_full.append(evaluate(x))
        accs_full.append(accuracy(x))
        num_grads_full.append((i+1) * linreg.n_data)


    # print("Number of gradients", num_grads[-1], num_grads_adaptive[-1])
    print("Optimal Loss (log scale)", np.log(f_min))
    plt.plot(losses, label=f'Fixed Batch Size = {batch_size_init}')
    plt.plot(losses_adaptive_ideal, label='Adaptive Batch Size (Ideal)')
    plt.plot(losses_adaptive, label='Adaptive Batch Size')
    plt.plot(losses_adaptive_proposed, label='Proposed Adaptive Batch Size')
    # plt.plot(losses_schedule, label='Batch Size Schedule (* 1/0.75 every 100 iterations)')
    # plt.plot(losses_adaptive_avg, label=f'Adaptive Batch Size (Average = {avg_batch_size})')
    plt.plot(losses_full, label='Full Gradient')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../data/losses_800_1000.png')
    plt.show()

    # plot against number of gradients
    plt.plot(num_grads, losses, label='Fixed Batch Size')
    plt.plot(num_grads_adaptive_ideal, losses_adaptive_ideal, label='Adaptive Batch Size (Ideal)')
    plt.plot(num_grads_adaptive, losses_adaptive, label='Adaptive Batch Size')
    plt.plot(num_grads_adaptive_proposed, losses_adaptive_proposed, label='Proposed Adaptive Batch Size')
    # plt.plot(num_grads_schedule, losses_schedule, label='Batch Size Schedule')
    # plt.plot(num_grads_adaptive_avg, losses_adaptive_avg, label=f'Adaptive Batch Size (Average = {avg_batch_size})')
    # plt.plot(num_grads_full, losses_full, label='Full Gradient')
    plt.yscale('log')
    plt.xlabel('Number of Gradients')
    plt.ylabel('Loss')
    plt.xlim(0, max(num_grads_adaptive))
    plt.legend()
    plt.show()


    # plt.plot(gds, label=f'Fixed Batch Size = {batch_size_init}')
    # plt.plot(gds_adaptive, label='Adaptive Batch Size')
    plt.title('Progression of Gradient Diversity')
    plt.plot(gds_full, label='Full Gradient Diversity')
    plt.plot(gds_adaptive_ideal, label='Batch Size Gradient Diversity (Ideal)')
    plt.plot(gds_adaptive_proposed, label='Proposed Adaptive Batch Size')
    # plt.plot(gds_schedule, label='Batch Size Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Diversity')
    plt.legend()
    # plt.savefig('../data/gradient_diversity.png')
    plt.show()

    # plt.plot(batch_size_list, label='Adaptive Batch Size')
    plt.plot(batch_size_list_ideal, label='Adaptive Batch Size (Ideal)')
    plt.plot(batch_size_list_proposed, label='Proposed Adaptive Batch Size')
    # plt.plot(batch_size_list_schedule, label='Batch Size Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Batch Size')
    plt.title(f'Adaptive Batch Size (Initial Batch Size = {batch_size_init})')
    # plt.savefig('../data/adaptive_batch_size.png')
    plt.legend()
    plt.show()




def sweep_batch_size():
    # Initialize your LinReg_DataGenerator with a specific dataset
    linreg = LogReg_DataGenerator(n_data=10000, dim=400, noise_scale=1e-5)
    x_init = linreg.rng.normal(size=linreg.dim)
    x_opt, _, _, _ = np.linalg.lstsq(linreg.A, linreg.b, rcond=None)
    f_min = linreg.evaluate(x_opt)
    evaluate = linreg.evaluate
    accuracy = linreg.accuracy
    # Experiment parameters
    # batch_sizes = [2, 4, 6, 8, 10, 20, 30, 50, 100, 'full']
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 'full']
    num_iters = 1000  # Number of iterations
    lr = 2 # Learning rate
    # Dictionary to hold loss arrays for each batch size
    losses_dict = {}

    for batch_size in batch_sizes:
        x = x_init.copy()
        losses = []
        accs = []
        gds = []
        num_grads = [0]
        for i in range(num_iters):
            if batch_size == 'full':
                grad, grad_norm = linreg.grad_func(x)  # Full batch gradient
            else:
                grad, grad_norm = linreg.batch_grad_func(x, batch_size)  # Minibatch gradient
            # num_grads.append(num_grads[-1] + batch_size)
            x = x - lr * grad
            losses.append(linreg.evaluate(x))
            accs.append(accuracy(x))

        # Store losses for this batch size
        losses_dict[batch_size] = losses

    # Plotting
    plt.figure(figsize=(10, 7))
    for batch_size, losses in losses_dict.items():
        plt.plot(losses, label=f'Batch Size = {batch_size}' if batch_size != 'full' else 'Full Batch')

    plt.title('Loss vs. Iterations for Different Batch Sizes')
    plt.xlabel('Iterations')
    plt.yscale('log')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.show()
def main():
    # Initialize Data

    exp()
    # sweep_batch_size()
    # linreg = LogReg_DataGenerator(n_data=10000, dim=400, noise_scale=1e-4)
    # grad_func = linreg.grad_func
    # evaluate = linreg.evaluate
    # accuracy = linreg.accuracy
    # x = linreg.rng.normal(size=linreg.dim)
    # x_init = x.copy()
    # lr = 0.19
    # losses = [evaluate(x)]
    # accs = [linreg.accuracy(x)]
    # gds = []
    # iter = 0
    # while np.linalg.norm(grad_func(x)[0]) > 1e-3 and iter < 10000:
    #     grad, grad_norm = grad_func(x)
    #     gd = gradient_diveristy(grad, grad_norm)
    #     gds.append(gd / linreg.n_data)
    #     x = x - lr * grad
    #     # batch_size = max(int(np.round(gd)),512)
    #     # print(f"Iteration {i}, Loss: {evaluate(x)}, Gradient Diversity: {gd}")
    #     losses.append(evaluate(x))
    #     accs.append(accuracy(x))
    #     iter += 1
    #     if iter % 1000 == 0:
    #         print(f"Iteration {iter}, Loss: {evaluate(x)}, Gradient norm: {np.linalg.norm(grad)}", f"Accuracy: {accuracy(x)}")
    #
    #
    # plt.plot(list(range(len(losses))), losses, label='Full Gradient')
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    #
    # plt.plot(gds, label=f'Full Gradient')
    # # plt.plot(gds_adaptive, label='Adaptive Batch Size')
    # plt.title('Progression of Gradient Diversity')
    # plt.xlabel('Iteration')
    # plt.ylabel('Gradient Diversity')
    # plt.legend()
    # # plt.savefig('../data/gradient_diversity.png')
    # plt.show()


if __name__ == '__main__':
    main()

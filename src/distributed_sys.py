from src.utils.cost_model import simulate_communication_costs, simulate_computation_costs, naive_partitions
import ray
# from server import ParameterServer
import numpy as np
import time
from tqdm import tqdm


class DistributedSystem:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        # Initialize Ray, ParameterServer, and DataWorkers in a method
        self.initialize_system()

    def initialize_system(self):
        if not ray.is_initialized():
            ray.init()
        self.ps = ParameterServer.remote(self.config["dim"], self.config["lr"], self.config["asynchronous"])
        rng = np.random.default_rng(self.config.get("seed", 42))
        seeds_workers = [rng.integers(self.config["max_seed"]) for _ in range(self.config["num_workers"])]

        # Simulate costs and initialize workers
        self.communication_costs = simulate_communication_costs(self.config['num_workers'],
                                                           **self.config['COST_DISTRIBUTION_PARAMS']['communication'])
        self.computation_costs = simulate_computation_costs(self.config['num_workers'], **self.config['COST_DISTRIBUTION_PARAMS']['computation'])

        self.workers = [DataWorker.remote(self.data_generator, self.config["lr"], self.config["batch_size"]//self.config["num_workers"],
                                          seeds_workers[i], self.computation_costs[i], self.communication_costs[i])
                        for i in range(self.config["num_workers"])]

    def updata_config(self, config):
        self.config = config
        self.initialize_system()
    def run_experiment(self, config=None):
        if config is not None:
            self.updata_config(config)

        # Extract configuration variables
        num_workers = self.config["num_workers"]
        lr = self.config["lr"]
        asynchronous = self.config["asynchronous"]
        iterations = self.config["iterations"]
        it_check = self.config["it_check"]
        batch_size = self.config["batch_size"]
        max_seed = self.config.get("max_seed", 424242)  # Providing a default value if not in configs
        delay_adaptive = self.config.get("delay_adaptive", False)  # Providing a default value if not in configs
        lr_decay = self.config.get("lr_decay", 0)
        worker_updates = [0 for _ in range(self.config["num_workers"])]
        self.cost_aware = self.config.get("cost_aware", False)
        if self.cost_aware:
            partition, _, _ = naive_partitions(self.computation_costs, self.communication_costs, batch_size)
            partition = [int(p) for p in partition]
            for i in range(num_workers):
                self.workers[i].update_batch_size.remote(partition[i])


        worker_batches = ray.get([worker.get_batch_size.remote() for worker in self.workers])
        x = self.ps.get_x.remote()

        if asynchronous:
            gradients = {}
            worker_last_it = [0 for _ in range(num_workers)]
            worker_id_to_num = {}
            # compuataion time

            for e, worker in enumerate(self.workers):
                gradients[worker.compute_gradients.remote(x)] = worker
                worker_id_to_num[worker] = e

        its = []
        ts = []
        delays = []
        t0 = time.perf_counter()
        delay = 0
        trace = []
        simulated_time = 0
        time_stamp = []
        grads_per_it = 1 if asynchronous else num_workers
        it = 0
        x = ray.get(x)
        grad_norm = np.linalg.norm(self.data_generator.grad_func(x)[0],2)
        # terminate condition is the norm of the gradient
        while grad_norm > 1e-6 and it < iterations* (num_workers if asynchronous else 1):
            # print(f"iteration {it} with norm {np.linalg.norm(self.data_generator.grad_func(x),2)}")
            it += num_workers if asynchronous else 1
            # for it in tqdm(range(iterations * (num_workers if asynchronous else 1))):
            n_grads = it * grads_per_it
            if asynchronous:
                ready_gradient_list, _ = ray.wait(list(gradients))
                ready_gradient_id = ready_gradient_list[-1]
                worker = gradients.pop(ready_gradient_id)

                # Compute and apply gradients.
                gradients[worker.compute_gradients.remote(x)] = worker
                worker_num = worker_id_to_num[worker]
                simulated_time += self.computation_costs[worker_num] * batch_size + 2 * self.communication_costs[worker_num]
                delay = it - worker_last_it[worker_num]
                if delay_adaptive:
                    lr_new = lr * num_workers / max(num_workers, delay)
                    self.ps.update_lr.remote(lr_new=lr_new)
                x = self.ps.apply_gradients.remote(grad=ready_gradient_id)
                worker_last_it[worker_num] = it
                worker_updates[worker_num] += 1
            else:
                # gradients = [
                #     worker.compute_gradients.remote(x) for worker in self.workers
                # ]

                # Initialize lists to hold the separate returned values from each worker
                sum_gradients_list = []
                sum_grad_norm_list = []

                # Use a loop or list comprehension to call the remote function and collect results
                for worker in self.workers:
                    sum_gradients, sum_grad_norm = ray.get(worker.compute_gradients.remote(x))
                    sum_gradients_list.append(sum_gradients)
                    sum_grad_norm_list.append(sum_grad_norm)

                # Calculate update after all gradients are available.
                x = self.ps.apply_gradients.remote(None, *sum_gradients_list)
                time_per_worker = [self.computation_costs[i] * worker_batches[i] + 2 * self.communication_costs[i] for i in range(num_workers)]
                max_time = max(time_per_worker)
                simulated_time += max_time
                if it % 100 == 0:
                    print(f"{'Cost-aware Sync SGD' if self.cost_aware else 'Sync SGD'}")
                    print("Batch per worker: ", worker_batches)
                    print("Computation time for each worker: ", time_per_worker)
                    print("Wait time: ", max_time)


            if isinstance(x, ray.ObjectRef):
                x = ray.get(self.ps.get_x.remote())

            grad_norm = np.linalg.norm(self.data_generator.grad_func(x)[0],2)

            if it % it_check == 0 or (not asynchronous and it % (max(it_check // num_workers, 1)) == 0):
                # Evaluate the current model.
                # if not asynchronous:
                #     print("Save at: ", it)
                # x = ray.get(self.ps.get_x.remote())
                trace.append(x.copy())
                its.append(it)
                ts.append(time.perf_counter() - t0)
                time_stamp.append(simulated_time)
                # print(f"Iteration {it}, Gradient Norm: {grad_norm}")

            lr_new = lr / (1 + lr_decay * n_grads)
            self.ps.update_lr.remote(lr_new=lr_new)
            if asynchronous:
                delays.append(delay)


        ray.shutdown()
        return np.asarray(its), np.asarray(ts), np.asarray([self.data_generator.evaluate(x) for x in trace]), np.asarray(
            delays), np.asarray(time_stamp)




@ray.remote
class DataWorker(object):
    """
    The class for an individual Ray worker.
    Arguments:
        lr (float): the stepsize to be used at initialization
        label (int, optional): batch size for sampling gradients (default: 1)
        seed (int, optional): random seed to generate random variables for reproducibility (default: 0)
        bad_worker (bool, optional): if True, the worker will be forced to be slower than others (default: False)
    """

    def __init__(self, data_generator, lr, batch_size=1, seed=0, comp_cost= 1e-9, comm_cost= 1e-6):
        self.lr = lr
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.comp_cost = comp_cost
        self.comm_cost = comm_cost
        self.rng = np.random.default_rng(seed)

    def compute_gradients(self, x):
        '''Compute the aggregated gradient of self.batch_size samples at parameter x,
        as well as the aggregated norm of the gradients.'''
        t0 = time.perf_counter()
        if self.batch_size is None:
            grad, norm_grad = self.data_generator.grad_func(x)
        elif self.batch_size == 1:
            grad, norm_grad = self.data_generator.sgrad_func(x)
        else:
            grad, norm_grad = self.data_generator.batch_grad_func(x, self.batch_size)
        # time taken
        # dt = time.perf_counter() - t0
        return grad, norm_grad

    def update_lr(self, lr_coef_mul=1, lr_new=None):
        if lr_new is not None:
            self.lr = lr_new
        else:
            self.lr *= lr_coef_mul

    def get_hyperparams(self):
        return self.lr, self.batch_size

    def get_batch_size(self):
        return self.batch_size


    def get_lr(self):
        return self.lr

    def get_comp_cost(self):
        return self.comp_cost

    def get_comm_cost(self):
        return self.comm_cost

    def update_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size




@ray.remote
class ParameterServer(object):
    def __init__(self, dim, lr, asynchronous):
        self.x = np.zeros(dim)
        self.lr = lr
        self.asynchronous = asynchronous

    def apply_gradients(self, grad, *gradients):
        if self.asynchronous:
            self.x -= self.lr * grad
        else:
            summed_gradients = np.sum(gradients, axis=0)
            self.x -= self.lr * summed_gradients
        return self.x

    def get_x(self):
        return self.x

    def update_lr(self, lr_coef_mul=1, lr_new=None):
        if lr_new is not None:
            self.lr = lr_new
        else:
            self.lr *= lr_coef_mul

    def get_hyperparams(self):
        return self.lr, self.asynchronous

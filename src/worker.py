import numpy as np
import psutil
import ray
import time

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
        t0 = time.perf_counter()
        if self.batch_size is None:
            grad = self.data_generator.grad_func(x)
        elif self.batch_size == 1:
            grad = self.data_generator.sgrad_func(x)
        else:
            grad = self.data_generator.batch_grad_func(x, self.batch_size)

        dt = time.perf_counter() - t0
        time.sleep(max(0, self.comp_cost * self.batch_size - dt))
        return grad

    def update_lr(self, lr_coef_mul=1, lr_new=None):
        if lr_new is not None:
            self.lr = lr_new
        else:
            self.lr *= lr_coef_mul

    def get_hyperparams(self):
        return self.lr, self.batch_size

    def get_lr(self):
        return self.lr



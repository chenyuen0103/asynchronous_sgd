import numpy as np
import psutil
import ray


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

import numpy as np
# import torch
# import torch.nn.functional as F

class LinReg_DataGenerator:
    def __init__(self, n_data=10000, dim=400, noise_scale=1e-5):
        self.n_data = n_data
        self.dim = dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng()  # Initialize a random number generator
        self.A = self.rng.uniform(size=(n_data, dim)) / np.sqrt(dim)
        self.x_rand = self.rng.normal(size=dim)
        self.b = self.A @ self.x_rand + noise_scale * self.rng.normal(size=n_data)

    def grad_func(self, x):
        """Compute the full gradient and sum of squares of gradients."""
        assert len(x) == self.dim
        # compute gradient with respect to each data point
        grad = (self.A @ x - self.b) @ self.A / self.n_data
        # sum over all data points
        # grad_norm0 = np.sum([((self.A[i] @ x - self.b[i]) * self.A[i]) ** 2 for i in range(self.n_data)])/self.n_data
        grad_norm = np.sum((self.A * ((self.A @ x - self.b)[:, None]))**2) / self.n_data
        # assert np.allclose(grad_norm0, grad_norm)

        return grad, grad_norm

    def sgrad_func(self, x):
        """Compute a stochastic gradient."""
        # sample a random index
        i = self.rng.integers(self.n_data)
        grad = (self.A[i] @ x - self.b[i]) * self.A[i]
        grad_norm = np.sum((self.A[i] @ x - self.b[i]) * self.A[i] ** 2)
        return grad, grad_norm

    def batch_grad_func(self, x, batch_size):
        """Compute a mini-batch gradient."""
        idx = self.rng.choice(self.n_data, size=batch_size, replace=False)
        A_batch = self.A[idx]
        b_batch = self.b[idx]

        # grad = (self.A[idx@ x - self.b[idx]) @ self.A[idx] / batch_size
        grad = (A_batch @ x - b_batch) @ A_batch / batch_size

        # grad_norm = np.sum([((self.A[i] @ x - self.b[i]) * self.A[i]) ** 2 for i in idx])/batch_size
        grad_norm = np.sum((A_batch * ((A_batch @ x - b_batch)[:, None]))**2) / batch_size
        return grad, grad_norm, idx

    def evaluate(self, x):
        """Evaluate the model."""
        assert len(x) == self.dim
        return 0.5 * np.mean((self.A @ x - self.b)**2)


class LogReg_DataGenerator:
    def __init__(self, n_data=10000, dim=400, noise_scale=1e-5):
        # self.n_data = n_data
        # self.dim = dim
        # self.noise_scale = noise_scale
        # self.rng = np.random.default_rng()
        # self.X = torch.FloatTensor(self.rng.uniform(-1, 1, size=(n_data, dim)))
        # self.weights = torch.FloatTensor(self.rng.normal(0, 1, size=(dim, 1)))
        # logits = self.model(self.X)
        # self.y = torch.sigmoid(logits) > 0.5  # Generating binary labels
        self.n_data = n_data
        self.dim = dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng()  # Initialize a random number generator
        self.A = self.rng.uniform(-1, 1, size=(n_data, dim))
        self.x_rand = self.rng.normal(0, 1, size=dim)
        # Generate linear combination
        linear_combination = self.A @ self.x_rand
        # Add noise
        noise = noise_scale * self.rng.normal(0, noise_scale, size=n_data)
        # Generate binary labels based on threshold
        logits = self.sigmoid(linear_combination + noise)
        self.b = (logits > 0.5).astype(int)
    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def get_subset(self, indices):
        """Return subsets of A and b based on provided indices."""
        return self.A[indices], self.b[indices]

    def grad_func(self, x, indices):
        """Compute the full gradient using only the specified indices."""
        A_subset, b_subset = self.get_subset(indices)
        y_pred = self.sigmoid(A_subset @ x)
        grad = (y_pred - b_subset) @ A_subset / len(indices)
        grad_norm = np.sum((A_subset * ((self.sigmoid(A_subset @ x) - b_subset)[:, None]))**2) / len(indices)
        return grad, grad_norm

    def evaluate(self, x, indices):
        """Evaluate the logistic loss using binary cross-entropy on a subset."""
        A_subset, b_subset = self.get_subset(indices)
        y_pred = self.sigmoid(A_subset @ x)
        cost = self.cost_func(y_pred, b_subset)
        return cost

    def accuracy(self, x, indices):
        """Evaluate the accuracy on a subset."""
        A_subset, b_subset = self.get_subset(indices)
        y_pred = self.sigmoid(A_subset @ x)
        return np.mean((y_pred > 0.5) == b_subset)

    def sgrad_func(self, x, indices=None):
        """Compute a stochastic gradient using a random example from provided indices."""
        if indices is None:
            indices = range(self.n_data)
        i = self.rng.choice(indices)  # Choose one index from the provided subset
        A_i = self.A[i]
        b_i = self.b[i]
        y_pred_i = self.sigmoid(A_i @ x)
        grad = np.dot(A_i, y_pred_i - b_i)
        grad_norm = np.sum((A_i * (y_pred_i - b_i)) ** 2)
        return grad, grad_norm

    def batch_grad_func(self, x, batch_size, indices=None):
        """Compute a mini-batch gradient from the provided indices subset or the whole dataset."""
        if indices is None:
            indices = range(self.n_data)
        batch_idx = self.rng.choice(indices, size=batch_size, replace=False)  # Sample batch indices
        A_batch = self.A[batch_idx]
        b_batch = self.b[batch_idx]
        y_pred = self.sigmoid(A_batch @ x)
        grad = np.dot(A_batch.T, y_pred - b_batch) / batch_size
        grad_norm = np.sum((A_batch * ((y_pred - b_batch)[:, None])) ** 2) / batch_size
        return grad, grad_norm, batch_idx


    def cost_func(self, y_pred, y):
        """Compute the logistic loss using binary cross-entropy."""
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))/self.n_data
        return cost



class Sine_DataGenerator:
    def __init__(self, n_data=10000, dim=400, noise_scale=1e-5):
        self.n_data = n_data
        self.dim = dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng()  # Initialize a random number generator
        self.A = self.rng.uniform(-1, 1, size=(n_data, dim))
        self.x_rand = self.rng.normal(0, 1, size=dim)
        self.b = np.sin(self.A @ self.x_rand) + noise_scale * self.rng.normal(size=n_data)

    def grad_func(self, x):
        """Compute the full gradient and sum of squares of gradients."""
        assert len(x) == self.dim
        # compute gradient with respect to each data point
        grad = (np.sin(self.A @ x) - self.b) @ self.A / self.n_data
        # sum over all data points
        grad_norm = np.sum((self.A * ((np.sin(self.A @ x) - self.b)[:, None]))**2) / self.n_data
        return grad, grad_norm

    def sgrad_func(self, x):
        """Compute a stochastic gradient."""
        # sample a random index
        i = self.rng.integers(self.n_data)
        grad = (np.sin(self.A[i] @ x) - self.b[i]) * self.A[i]
        grad_norm = np.sum((self.A[i] @ x - self.b[i]) * self.A[i] ** 2)
        return grad, grad_norm

    def batch_grad_func(self, x, batch_size):
        """Compute a mini-batch gradient."""
        idx = self.rng.choice(self.n_data, size=batch_size, replace=False)
        A_batch = self.A[idx]
        b_batch = self.b[idx]
        grad = (np.sin(A_batch @ x) - b_batch) @ A_batch / batch_size
        grad_norm = np.sum((A_batch * ((np.sin(A_batch @ x) - b_batch)[:, None]))**2) / batch_size
        return grad, grad_norm

    def evaluate(self, x):
        """Evaluate the model."""
        assert len(x) == self.dim
        return 0.5 * np.mean((np.sin(self.A @ x) - self.b)**2)
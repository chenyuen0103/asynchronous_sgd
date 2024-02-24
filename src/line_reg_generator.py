import numpy as np

class DataGenerator:
    def __init__(self, n_data=10000, dim=400, noise_scale=1e-5):
        self.n_data = n_data
        self.dim = dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng()  # Initialize a random number generator
        self.A = self.rng.uniform(size=(n_data, dim)) / np.sqrt(dim)
        self.x_rand = self.rng.normal(size=dim)
        self.b = self.A @ self.x_rand + noise_scale * self.rng.normal(size=n_data)

    def grad_func(self, x):
        """Compute the full gradient."""
        assert len(x) == self.dim
        return (self.A @ x - self.b) @ self.A / self.n_data

    def sgrad_func(self, x):
        """Compute a stochastic gradient."""
        # sample a random index
        i = self.rng.integers(self.n_data)
        return (self.A[i] @ x - self.b[i]) * self.A[i]

    def batch_grad_func(self, x, batch_size):
        """Compute a mini-batch gradient."""
        idx = self.rng.choice(self.n_data, size=batch_size, replace=False)
        return (self.A[idx] @ x - self.b[idx]) @ self.A[idx] / batch_size

    def evaluate(self, x):
        """Evaluate the model."""
        assert len(x) == self.dim
        return 0.5 * np.mean((self.A @ x - self.b)**2)
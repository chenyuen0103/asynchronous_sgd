import numpy as np
import torch
import torch.nn.functional as F

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


class LogReg_DataGenerator:
    def __init__(self, n_data=10000, dim=400, noise_scale=1e-5):
        self.model = torch.nn.Linear(dim, 1, bias=False)
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
        noise = noise_scale * self.rng.normal(0, 1, size=n_data)
        # Generate binary labels based on threshold
        logits = self.sigmoid(linear_combination + noise)
        self.b = (logits > 0.5).astype(int)
    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def grad_func(self, x):
        """Compute the full gradient."""
        self.model.linear.weight.data = x.t()  # Update model weights
        logits = self.model(self.X)
        loss = F.binary_cross_entropy(logits, self.y.float(), reduction='sum')
        loss.backward()
        return self.model.linear.weight.grad.data.t() / self.n_data

    def sgrad_func(self, x):
        """Compute a stochastic gradient."""
        i = self.rng.integers(self.n_data)
        xi = self.X[i].unsqueeze(0)
        yi = self.y[i].unsqueeze(0).float()
        self.model.linear.weight.data = x.t()  # Update model weights
        logits = self.model(xi)
        loss = F.binary_cross_entropy(logits, yi, reduction='sum')
        loss.backward()
        return self.model.linear.weight.grad.data.t()

    def batch_grad_func(self, x, batch_size):
        """Compute a mini-batch gradient."""
        idx = self.rng.choice(self.n_data, size=batch_size, replace=False)
        X_batch = self.X[idx]
        y_batch = self.y[idx].float()
        self.model.linear.weight.data = x.t()  # Update model weights
        logits = self.model(X_batch)
        loss = F.binary_cross_entropy(logits, y_batch, reduction='sum')
        loss.backward()
        return self.model.linear.weight.grad.data.t() / batch_size

    def evaluate(self, x):
        """Evaluate the logistic loss using binary cross-entropy."""
        # Ensure x is a PyTorch tensor with the correct shape
        x = x.reshape(-1, self.dim)  # Reshape x if necessary
        self.model.weight.data = x  # Update model weights

        # Compute logits using the current model parameters
        logits = self.model(self.X)

        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, self.y.unsqueeze(1), reduction='mean')

        return loss.item()
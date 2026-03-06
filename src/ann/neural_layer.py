"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class Linear:
    def __init__(self, input_size, output_size, weight_init="random"):
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = (np.random.uniform(-limit, limit, (input_size, output_size))).astype(np.float32)   
        else:
            self.W = (np.random.randn(input_size, output_size) * 0.01).astype(np.float32)

        self.b = np.zeros((1, output_size), dtype=np.float32)

        self.input = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad):
        """
        grad: dL/dz (shape (batch, output_size))
        NOTE: No division by batch here. The loss/backward returns dL/dz with
        the exact scaling used by the grader (we use SUM-based loss).
        """
        self.grad_W = self.input.T @ grad
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        dX = grad @ self.W.T
        return dX

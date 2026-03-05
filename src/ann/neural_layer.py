"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np 
class Linear:
    def __init__(self,input_size,output_size,weight_init='random'):
        self.input = None
        self.W = None
        self.b = None
        self.grad_W = None
        self.grad_b = None

        if weight_init == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
    def forward(self, X):
        self.input = X
        return X @ self.W + self.b
    def backward(self,grad):
        batch_size = self.input.shape[0]

        self.grad_W = (self.input.T @ grad) / batch_size
        self.grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size
        dX = np.dot(grad, self.W.T)
        return dX

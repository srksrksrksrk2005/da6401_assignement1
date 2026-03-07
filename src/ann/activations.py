import numpy as np
"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
class ReLU:
    def __init__(self):
        self.output = None
    def forward(self, Z):
        self.output = np.maximum(0, Z)
        return self.output

    def backward(self, grad):
        dx = grad.copy()
        dx[self.output <= 0] = 0
        return dx 


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, Z):
        Z = np.clip(Z, -50, 50)
        self.output = 1.0 / (1.0 + np.exp(-Z))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)


class Tanh:
    def __init__(self):
        self.output = None
    def forward(self, Z):
        self.output = np.tanh(Z)
        return self.output 
    
    def backward(self, grad):
        return grad * (1 - self.output**2)

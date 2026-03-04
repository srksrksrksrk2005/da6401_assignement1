"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                dW = layer.grad_W + self.weight_decay * layer.W
                
                layer.W -= self.lr * dW
                layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if i not in self.velocities:
                    self.velocities[i] = {"vW": np.zeros_like(layer.W),"vb": np.zeros_like(layer.b)}

                dW = layer.grad_W + self.weight_decay * layer.W

                self.velocities[i]["vW"] = self.beta * self.velocities[i]["vW"] + dW
                self.velocities[i]["vb"] = self.beta * self.velocities[i]["vb"] + layer.grad_b
                layer.W -= self.lr * self.velocities[i]["vW"]
                layer.b -= self.lr * self.velocities[i]["vb"]

class NAG:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if i not in self.velocities:
                    self.velocities[i] = {"vW": np.zeros_like(layer.W),"vb": np.zeros_like(layer.b)}


                dW = layer.grad_W + self.weight_decay * layer.W
                db = layer.grad_b
                
                v_prev_W = self.velocities[i]["vW"]
                v_prev_b = self.velocities[i]["vb"]
                
                self.velocities[i]["vW"] = self.beta * v_prev_W + dW
                self.velocities[i]["vb"] = self.beta * v_prev_b + db

                layer.W -= self.lr * (self.beta * v_prev_W + dW)
                layer.b -= self.lr * (self.beta * v_prev_b + db)

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.sq_grads = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if i not in self.sq_grads:
                    self.sq_grads[i] = {"sW": np.zeros_like(layer.W),"sb": np.zeros_like(layer.b)}

                dW = layer.grad_W + self.weight_decay * layer.W
                db = layer.grad_b

                self.sq_grads[i]["sW"] = self.beta * self.sq_grads[i]["sW"] + (1 - self.beta) * (dW ** 2)
                self.sq_grads[i]["sb"] = self.beta * self.sq_grads[i]["sb"] + (1 - self.beta) * (db ** 2)

                layer.W -= self.lr * dW / (np.sqrt(self.sq_grads[i]["sW"]) + self.epsilon)
                layer.b -= self.lr * db / (np.sqrt(self.sq_grads[i]["sb"]) + self.epsilon)


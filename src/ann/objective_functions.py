"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class Cross_Entropy:

    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true

        loss = -np.sum(y_true * np.log(self.probs + 1e-8)) / logits.shape[0]
        return loss

    def backward(self):
        return (self.probs - self.y_true) 

class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2.0 * (self.y_pred - self.y_true) / self.y_true.shape[1]

"""
Loss Functions
"""

import numpy as np


class Cross_Entropy:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits, y_true):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.y_pred = probs
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / logits.shape[0]
        return loss

    def backward(self):
        return (self.y_pred - self.y_true)/self.y_true.shape[0]



class MSE:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.sum((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        batch = self.y_true.shape[0]
        C = self.y_true.shape[1]
        return 2.0 * (self.y_pred - self.y_true) / (batch * C)

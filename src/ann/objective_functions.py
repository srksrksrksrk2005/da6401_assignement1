"""
Loss Functions
"""

import numpy as np


class Cross_Entropy:

    def __init__(self):
        self.y_pred= None
        self.y_true = None

    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - logsumexp
        self.y_pred = np.exp(log_probs)
        self.y_true = y_true
        loss = -np.sum(y_true * log_probs) / logits.shape[0]
        return loss

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]



class MSE:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.y_pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.y_true = y_true
        loss = np.mean((self.y_pred - y_true) ** 2)
        return loss

    def backward(self):
        batch = self.y_true.shape[0]
        num_classes = self.y_true.shape[1]
        grad_prob = 2.0 * (self.y_pred - self.y_true) / (batch * num_classes)
        dot = np.sum(grad_prob * self.y_pred, axis=1, keepdims=True)
        return self.y_pred * (grad_prob - dot)

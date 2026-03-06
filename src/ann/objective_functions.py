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
        self.y_pred = exp / np.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true

        loss = -np.sum(y_true * np.log(self.y_pred + 1e-8))

        return loss


    def backward(self):

        return (self.y_pred - self.y_true)/self.y_pred.shape[0]



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

        return 2 * (self.y_pred - self.y_true))/self.y_pred.shape[0]*)self.y_pred.shape[1]

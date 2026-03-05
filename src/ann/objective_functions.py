"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class Cross_Entropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        
    def forward(self, y_pred, y_true):
        exp = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))    
        self.y_pred = exp / np.sum(exp, axis=1, keepdims=True)
        self.y_true = y_true
        
        loss = -np.sum(y_true * np.log(self.y_pred + 1e-8)) / y_true.shape[0]
        return loss

    def backward(self,y_pred, y_true):
        return (self.y_pred - self.y_true) 



class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self,y_pred, y_true):
        return 2.0 * (self.y_pred - self.y_true) / self.y_true.shape[1]

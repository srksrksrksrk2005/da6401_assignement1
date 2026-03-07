"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np

from .neural_layer import Linear
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import Cross_Entropy, MSE
from sklearn.metrics import f1_score

class NeuralNetwork:

    def __init__(self, args):

        self.layers = []

        input_size = 784
        output_size = 10

        if hasattr(args, "hidden_size"):
            hidden_sizes = args.hidden_size
        elif hasattr(args, "hidden_sizes"):
            hidden_sizes = args.hidden_sizes
        elif hasattr(args, "sz"):
            hidden_sizes = args.sz
        else:
            hidden_sizes = []

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        activation_name = getattr(args, "activation", "relu")
        weight_init = getattr(args, "weight_init", "random")

        activation_map = {"relu": ReLU,"sigmoid": Sigmoid,"tanh": Tanh}

        activation_class = activation_map[activation_name]

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            in_sz = int(layer_sizes[i]); out_sz = int(layer_sizes[i+1])
            self.layers.append(Linear(in_sz, out_sz, weight_init))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_class())
                
        loss_name = getattr(args, "loss", "cross_entropy")

        if loss_name == "cross_entropy":
            self.loss = Cross_Entropy()
        else:
            self.loss = MSE()

        self.optimizer = getattr(args, "optimizer", None)
        self.grad_W = None
        self.grad_b = None

    def update_weights(self):
        self.optimizer.step(self.layers)
        
    def forward(self, X):
        """
            Forward propagation through all layers.
            Returns logits (no softmax applied)
            X is shape (b, D_in) and output is shape (b, D_out).
            b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out


    def backward(self, y_true, logits):
        """
            Backward propagation to compute gradients.
            Returns two numpy arrays: grad_Ws, grad_bs.
            - `grad_Ws[0]` is gradient for the last (output) layer weights,
            `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        self.loss.forward(logits, y_true)
        grad = self.loss.backward()

        grad_W = []
        grad_b = []

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if hasattr(layer, "grad_W"):
                grad_W.append(layer.grad_W)
                grad_b.append(layer.grad_b)
                
        self.grad_W = grad_W
        self.grad_b = grad_b
        return grad_W, grad_b


    def get_weights(self):

        weights = {}
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                weights[f"W{idx}"] = layer.W.copy()
                weights[f"b{idx}"] = layer.b.copy()
                idx += 1

        return weights


    def set_weights(self, weights):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = weights[f"W{idx}"].copy()
                layer.b = weights[f"b{idx}"].copy()
                idx += 1
                
    def evaluate(self, X, y):
        
        logits  = self.forward(X)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)

        return accuracy 
    
    def train(self, X_train, y_train, epochs, batch_size):
        n_samples = X_train.shape[0]
        iteration = 0
        for epoch in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                logits  = self.forward(X_batch)
                
                loss = self.loss.forward(logits , y_batch)
                epoch_loss += loss
                
                self.backward(y_batch, logits)
                self.update_weights()
                iteration += 1  
            avg_loss = epoch_loss /  int(np.ceil(n_samples / batch_size))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            train_acc = self.evaluate(X_train[:5000], y_train[:5000])[0]
        return train_acc 

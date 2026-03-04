"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from sklearn.metrics import f1_score
from .neural_layer import Linear
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import Cross_Entropy, MSE
import numpy as np
import wandb

class NeuralNetwork:
    """
        Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.layers = []
        input_size = 784
        output_size = 10
        
        hidden_sizes = cli_args.hidden_size  
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init
        
        activations = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }   
        activation_class = activations[activation_name]

        prev_size = input_size
        
        for h in hidden_sizes:
            self.layers.append(Linear(prev_size, h,weight_init))
            self.layers.append(activation_class())
            prev_size = h

        self.layers.append(Linear(prev_size, output_size, weight_init))

        self.loss = Cross_Entropy() if cli_args.loss.lower() == "cross_entropy" else MSE()
        
        self.optimizer = cli_args.optimizer
        
    def get_weights(self):
        d = {}
        lin_idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                d[f"W{lin_idx}"] = layer.W.copy()
                d[f"b{lin_idx}"] = layer.b.copy()
                lin_idx += 1
        return d

    def set_weights(self, weights_list):
        lin_idx = 0
        for layer in self.layers:   
            if hasattr(layer, "W"):
                layer.W = weights_list[f"W{lin_idx}"].copy()
                layer.b = weights_list[f"b{lin_idx}"].copy()
                lin_idx += 1
                
    def forward(self, X):
        """
            Forward propagation through all layers.
            Returns logits (no softmax applied)
            X is shape (b, D_in) and output is shape (b, D_out).
            b is batch size, D_in is input dimension, D_out is output dimension.
        """
        logits = X
        for layer in self.layers:
            logits = layer.forward(logits)
        return logits
    
    def backward(self, y_true, y_pred):
        """
            Backward propagation to compute gradients.
            Returns two numpy arrays: grad_Ws, grad_bs.
            - `grad_Ws[0]` is gradient for the last (output) layer weights,
            `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad = self.loss.backward(y_pred, y_true)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers): # Backprop through layers in reverse; collect grads so that index 0 = last layer
            grad = layer.backward(grad)
            if hasattr(layer, "W"):
                grad_W_list.append(layer.grad_W)
                grad_b_list.append(layer.grad_b)
                
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

            
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    
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
                self.optimizer.step(self.layers)
                
                
            avg_loss = epoch_loss /  int(np.ceil(n_samples / batch_size))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            train_acc = self.evaluate(X_train[:5000], y_train[:5000])[0]
        return train_acc ,f1_score(y_train[:5000].argmax(axis=1), self.forward(X_train[:5000]).argmax(axis=1), average='macro')
    
    
    def evaluate(self, X, y):
        
        logits  = self.forward(X)

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return accuracy ,f1_score(true_labels, predictions, average='macro')

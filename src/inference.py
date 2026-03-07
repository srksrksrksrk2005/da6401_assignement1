"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD
import json


    
"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument("-d", "--dataset",type=str,choices=["mnist", "fashion_mnist"],default="mnist",help="Choose dataset: mnist or fashion_mnist")

    # Epochs
    parser.add_argument("-e", "--epochs",type=int,default=10,help="Number of training epochs")

    # Batch size
    parser.add_argument("-b", "--batch_size",type=int,default=32,help="Mini-batch size")

    # Loss function
    parser.add_argument( "-l", "--loss",type=str,choices=["mean_squared_error", "cross_entropy"],default="cross_entropy",help="Loss function")

    # Optimizer
    parser.add_argument("-o", "--optimizer",type=str,choices=["sgd", "momentum", "nag", "rmsprop"],default="rmsprop",help="Optimizer")

    # Learning rate
    parser.add_argument("-lr", "--learning_rate",type=float,default=0.001,help="Initial learning rate")

    # Weight decay
    parser.add_argument("-wd", "--weight_decay",type=float,default=0.0,help="Weight decay for L2 regularization")

    # Number of hidden layers
    parser.add_argument("-nhl", "--num_layers",type=int,default=2,help="Number of hidden layers")

    # Hidden layer sizes
    parser.add_argument("-sz", "--hidden_size",type=int,nargs="+",default=[128, 64],help="Number of neurons in each hidden layer")

    # Activation
    parser.add_argument("-a", "--activation",type=str,choices=["sigmoid", "tanh", "relu"],default="relu",help="Activation function for hidden layers")

    # Weight initialization
    parser.add_argument("-w_i", "--weight_init",type=str,choices=["random", "xavier"],default="xavier",help="Weight initialization method")

    # W&B project
    parser.add_argument("-w_p", "--wandb_project",type=str,default="DA6401_assignement1",help="Weights & Biases project ID")

    parser.add_argument("-mp", "--model_path",type=str,default="best_model.npy",help="Relative path to save trained model")
    return parser.parse_args()



def load_model(model_path):
    """
    Load trained model from disk.
    """
    weights = np.load(model_path, allow_pickle=True).item()  
    return weights




def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    loss = model.loss.forward(logits, y_test)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1}
    
def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    dataset = load_data(args.dataset)
    X_test, y_test = dataset[2::]  
    model = NeuralNetwork(args)  
    model.set_weights(load_model(args.model_path)) 
    results = evaluate_model(model, X_test, y_test)
    print("Evaluation complete!")
    print("accuracy :",results["accuracy"])
    print("precision :",results["precision"])
    print("f1 :",results["f1"])
    print("logits :",results["logits"])
    return results

if __name__ == '__main__':
    main()

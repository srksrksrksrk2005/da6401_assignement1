"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
import wandb
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Relative path to saved model weights (.npy)"
    )

    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        default="mnist",
        choices=['mnist', 'fashion_mnist']
    )

    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=64
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="best_config.json",
        required=True,
        help="Path to best_config.json"
    )

    return parser.parse_args()

def load_model(model_path):
    """
    Load trained model from disk.
    """
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    loss = model.loss.forward(logits, y_test)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def main():
    """
    Main inference function.
    
    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    _, _, X_test, y_test = load_data(args.dataset)

    weights = load_model(args.model_path)
    with open( args.config_path, "r") as f:
        config = json.load(f)

    class DummyArgs:
        pass

    args = DummyArgs()
    args.hidden_size = config["hidden_size"]
    args.activation = config["activation"]
    args.loss = config["loss"]
    args.weight_init = config["weight_init"]
    args.num_layers = config["num_layers"]
    args.batch_size = config["batch_size"]
    
    args.optimizer = SGD(lr=config["learning_rate"])
    model = NeuralNetwork(args)
    model.set_weights(weights)
    
    results = evaluate_model(model, X_test, y_test)
    
    
    print("Evaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()

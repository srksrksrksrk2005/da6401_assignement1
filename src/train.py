"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from sklearn.metrics import f1_score
import wandb
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, RMSProp, Momentum, NAG
import json
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Train a neural network')
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        default="mnist",
        choices=['mnist', 'fashion_mnist']
    )
    
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=20,
        required=True
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        required=True,
        default=64
    )
    parser.add_argument(
        '-l', '--loss', 
        type=str, 
        required=True,
        default="cross_entropy",
        choices=['cross_entropy', 'mse']
    )
    parser.add_argument(
        '-o', '--optimizer',
        type=str,
        required=True,
        default="sgd",  
        choices=['sgd', 'momentum', 'nag', 'rmsprop']
    )
    
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=0.001,
        required=True
    )
    
    parser.add_argument(
        '-wd', '--weight_decay',
        type=float,
        default=0.0
    )
    
    parser.add_argument(
        '-nhl', '--num_layers',
        type=int,
        default=2,
        required=True
    )

    parser.add_argument(
        '-sz', '--hidden_size',
        type=int,
        nargs='+',
        default=[128, 64],
        required=True
    )
    
    parser.add_argument(
        '-a', '--activation',
        type=str,
        required=True,
        default="relu",
        choices=['relu', 'sigmoid', 'tanh']
    )
    
    parser.add_argument(
        '-w_i', '--weight_init',
        type=str,
        required=True,
        default="xavier",
        choices=['random', 'xavier'],
        help="Weight initialization method"
    )

    
    
    parser.add_argument(
        '--model_path',
        type=str,
        default="best_model.npy",
        help="Relative path to save trained model"
    )
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='best_config.json', 
        help='relative path for saving config (json)'
    )
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    best_f1 = -1
    args = parse_arguments()
    
    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match length of hidden_size list")
    
    X_train, y_train, X_test, y_test = load_data(args.dataset)
            
    if args.optimizer == "sgd":
        optimizer = SGD(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NAG(lr=args.learning_rate,weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    args.optimizer = optimizer
    args.hidden_size = args.hidden_size
    
    model = NeuralNetwork(args)

    model.train(X_train, y_train, args.epochs, args.batch_size)
    
    train_acc, train_f1 = model.evaluate(X_train, y_train)
    test_acc, test_f1 = model.evaluate(X_test, y_test)
    if test_f1 > best_f1:
        best_f1 = test_f1

        model_data = {
            "weights": model.get_weights(),
            "config": {
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "loss": args.loss,
                "weight_init": args.weight_init,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
            }
        

        np.save("best_model.npy", model_data["weights"])

        with open("best_config.json", "w") as f:
            json.dump(model_data["config"], f, indent=2)

      
    print("Training complete!")


if __name__ == '__main__':
    main()

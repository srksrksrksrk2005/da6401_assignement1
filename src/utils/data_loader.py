"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_data(dataset_name):
    
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be either 'mnist' or 'fashion_mnist'")

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test

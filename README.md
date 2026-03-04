# Deep Learning Assignment 1  
Neural Network from Scratch using NumPy  

## 📌 Overview

This project implements a fully connected neural network from scratch using NumPy.  
The model supports multiple hidden layers, activation functions, loss functions, and optimizers.

The implementation satisfies:

- Forward pass verification
- Analytical gradient computation
- Multiple optimizers
- Weight initialization strategies
- Mini-batch training
- Model saving and inference

---

## 📂 Project Structure

# Deep Learning Assignment 1  
Neural Network from Scratch using NumPy  

## 📌 Overview

This project implements a fully connected neural network from scratch using NumPy.  
The model supports multiple hidden layers, activation functions, loss functions, and optimizers.

The implementation satisfies:

- Forward pass verification
- Analytical gradient computation
- Multiple optimizers
- Weight initialization strategies
- Mini-batch training
- Model saving and inference

---

## 📂 Project Structure
DL_assignment1/
├─ ann/
│  ├─ __init__.py
│  ├─ activations.py
│  ├─ neural_layer.py
│  ├─ neural_network.py
│  ├─ objective_functions.py
│  └─ optimizers.py
├─ utils/
|   ├─ __init__.py
|   ├─data_loader.py
├─ train.py
├─ inference.py
└─ README.md

---

## 🚀 How to Train

Example:

python train.py -d mnist -e 5 -b 64 -l cross_entropy -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -w_i xavier 

## 🚀 How to Run Inference

Example:

python inference.py --model_path best_model.npy -d mnist --config_path best_config.json


---

## 🚀 Weights & Biases report link

https://wandb.ai/sadamrk2005-indian-institute-of-technology-madras/DA6401_assignement1/reportlist

## GIT HUB REPO
https://github.com/srksrksrksrk2005/da6401_assignement1/tree/main




# Deep Learning Basics with PyTorch

## What I Learned
- What tensors are
- Tensor shapes and operations
- Building neural networks using torch.nn
- Forward pass through a neural network

## Training a Neural Network

- Implemented training loop
- Used MSE loss and SGD optimizer
- Observed loss decreasing over epochs
- Verified model learning simple function

## Convolutional Neural Network on MNIST

- Built CNN using Conv2D, ReLU, and MaxPooling
- Trained on MNIST handwritten digits
- Achieved high classification accuracy

## Preventing Overfitting

- Split training data into train and validation sets
- Evaluated validation accuracy each epoch
- Added dropout layer to improve generalization

## Files
- tensors.py: tensor creation and operations
- first_nn.py: simple neural network forward pass
- train_nn.py: forward and backward pass with loss function and training loop
- cnn_mnist.py: cnn for mnist digit classification

## Tools
- PyTorch
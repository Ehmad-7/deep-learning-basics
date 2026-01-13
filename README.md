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

I ran controlled experiments by changing one factor at a time:

### Experiment 1: Learning Rate
- LR = 0.0001
- Result: slower learning, validation accuracy dropped (~94%)
- Conclusion: too small LR caused underfitting in limited epochs

### Experiment 2: Batch Size
- Batch size = 128
- Result: similar accuracy (~98.3%), more stable training
- Conclusion: batch size affects stability, not major accuracy

### Experiment 3: Deeper Network
- Added extra convolution layer
- Result: slightly lower accuracy (~97.9%)
- Conclusion: extra capacity not needed for simple MNIST

Final model uses batch size 128 and original CNN architecture.

## Transfer Learning with ResNet

- Used pretrained ResNet18 from ImageNet
- Converted MNIST to 3-channel 224x224 images
- Froze backbone and trained final classifier layer
- Achieved high accuracy in very few epochs

## Files
- tensors.py: tensor creation and operations
- first_nn.py: simple neural network forward pass
- train_nn.py: forward and backward pass with loss function and training loop
- cnn_mnist.py: cnn for mnist digit classification
-transfer_learning_mnist.py: transfer learning with pretrained ResNet18 on MNIST

## Tools
- PyTorch
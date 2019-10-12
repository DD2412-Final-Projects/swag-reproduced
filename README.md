# Reproducability Project - "A Simple Baseline for Bayesian Uncertainty in Deep Learning"

Paper: https://arxiv.org/pdf/1902.02476.pdf

## Resources used in the implementation
* TensorFlow implementation of VGG-16: https://www.cs.toronto.edu/~frossard/post/vgg16/.   
Pre-trained weights: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz

## Suggested file structure

```
+-- train.py (loads a network architecture, runs SWAG to train it and saves the learned weights)
+-- test.py (loads learned weights, runs the SWAG test procedure and reports resulting metrics and plots)
+-- networks/ (directory with all architectures)
|   +-- vgg16/ (directory for the vgg16 implementation)
    |   +-- vgg16.py (the vgg16 implementation)
    |   +-- vgg16_usage.py (shows example usage of vgg16.py)
+-- weights/ (directory for model weights, *not* tracked by git)
+-- data/ (directory for data, *not* tracked by git)
```

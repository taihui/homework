# Define some auxiliary functions
import torchvision
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import os


################# Process Dataset #################
def get_dataset():
    # Mnist digits dataset
    training_dataset = torchvision.datasets.MNIST(
        root='dataset/mnist/training/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,  # download it if you don't have it
    )

    print(training_dataset.data.size())
    print(training_dataset.targets.size())

    test_dataset = torchvision.datasets.MNIST(
        root='dataset/mnist/training/',
        train=False,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,  # download it if you don't have it
    )

    print(test_dataset.data.size())
    print(test_dataset.targets.size())

    return training_dataset, test_dataset



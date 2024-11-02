import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import medmnist
from medmnist import INFO, Evaluator

from medmnist.dataset import BreastMNIST
import yaml
with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)

def load_iris_data(test_size=0.8):
    iris = load_iris()
    data = iris['data']
    labels = iris['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if(args['method'] == 'beta' or args['method'] == 'uniform_norm' or args['method'] == 'classical_CNN' or args['method'] == 'var_encoders' or args['method'] == 'time_nonlocal'):
        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test
def load_mnist_data(test_size=0.995):
    # Define the transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Extract data (images) and labels
    data = mnist_dataset.data.numpy()  # Convert images to NumPy array
    labels = mnist_dataset.targets.numpy()  # Convert labels to NumPy array

    # Reshape the data to flatten 28x28 images into vectors of 784 pixels
    data = data.reshape(data.shape[0], -1)  # Reshape to (n_samples, 784)

    # Train-test split using sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    # Scale the data to standardize it (mean 0, variance 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if(args['method'] == 'beta' or args['method'] == 'uniform_norm' or args['method'] == 'classical_CNN' or args['method'] == 'var_encoders' or args['method'] == 'time_nonlocal'):
        # Convert the data and labels to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test
def load_medmnist_data(test_size=0.995):
    # Define the dataset info
    dataset_name = 'breastmnist'  # Can be changed to other MedMNIST datasets like 'dermamnist', etc.
    info = INFO[dataset_name]

    # Define the transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MedMNIST dataset
    train_dataset = BreastMNIST(split='train', transform=transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=transform, download=True)

    # Extract the data (images) and labels
    train_data = train_dataset.imgs  # Images from training set
    train_labels = train_dataset.labels  # Corresponding labels for the images

    test_data = test_dataset.imgs  # Images from testing set
    test_labels = test_dataset.labels  # Corresponding labels for the images

    # Reshape the data to flatten (for compatibility with MLPs)
    train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten images (n_samples, width * height)
    test_data = test_data.reshape(test_data.shape[0], -1)  # Flatten images (n_samples, width * height)

    # Concatenate the train and test data
    data = np.concatenate((train_data, test_data), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    # Train-test split using sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    # Scale the data to standardize it (mean 0, variance 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if(args['method'] == 'beta' or args['method'] == 'uniform_norm' or args['method'] == 'classical_CNN' or args['method'] == 'var_encoders' or args['method'] == 'time_nonlocal'):
        # Convert the data and labels to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
        y_test = torch.tensor(y_test.squeeze(), dtype=torch.long)

    return X_train, X_test, y_train, y_test
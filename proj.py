#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from skorch import NeuralNetClassifier
import urllib3
import torch.nn.functional as F
import argparse

# Data Preprocessing
def get_data_loaders(train_transforms, test_transforms, batch_size=64, num_workers=8):
    train_dataset = datasets.ImageFolder(root='archive/Training', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root='archive/Testing', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# Converting data and labels to arrays
def preprocess_data(train_loader):
    num_images = len(train_loader.dataset)
    image_shape = next(iter(train_loader))[0].shape[1:]

    data_all = np.zeros((num_images,) + image_shape, dtype=np.float32)
    labels_all = np.zeros(num_images, dtype=np.int64)

    start_idx = 0
    for images, labels in train_loader:
        end_idx = start_idx + images.size(0)
        data_all[start_idx:end_idx] = images.numpy()
        labels_all[start_idx:end_idx] = labels.numpy()
        start_idx = end_idx

    return data_all, labels_all

# Model Architecture
class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Calculate the size of the flattened output after conv and pooling layers
        #self.flattened_size = (64 // (2**3)) * (64 // (2**3)) * 128
        self.flattened_size = 8192
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 4)
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x

# skorch Wrapper (GridSearchCV)
def grid_search(train_loader):
    net = NeuralNetClassifier(
        BrainTumorClassifier,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        batch_size=64,
        max_epochs=25,
        device='cpu'
    )

    param_grid = {
        'lr': [0.0001, 0.001, 0.01, 0.005, 0.0005],
        'max_epochs': list(range(20, 55, 5))
    }


    X_train, y_train = preprocess_data(train_loader)

    # Assuming X_train and y_train are your training data and labels
    grid = GridSearchCV(net, param_grid, refit=True, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    # Print the results
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

# Subset Testing
def subset_testing(train_dataset, test_dataset):
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    device = torch.device("cpu")
    num_epochs = 25
    subset_size = 80

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold+1}/{k_folds}')
        # Randomly sample indices for training and testing
        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), subset_size, replace=False)

        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=0)

        # Instantiate a new model for this fold
        model = BrainTumorClassifier()
        model.to(device)

        # Define the loss function and optimizer for this fold
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Start the training loop for this fold
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            train_loss = running_loss / len(train_loader)

            # Validation loop
            model.eval()
            running_loss_val = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss_val += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            val_loss = running_loss_val / len(test_loader)

            # Print statistics for this epoch
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        print(f'Finished Training for Fold {fold+1}')

        # Save the trained model state
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, 'subset_trained_model.pth')

# Full Training / Validation
def full_training_validation(train_dataset, test_dataset):
    k_folds = 5
    num_epochs = 25

    for fold in range(k_folds):
        print(f'Fold {fold+1}/{k_folds}')

        # Create data loaders for the entire datasets
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Instantiate a new model for this fold
        model = BrainTumorClassifier()
        device = 'cpu'
        model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            train_loss = running_loss / len(train_loader)

            # Validation loop
            model.eval()
            running_loss_val = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss_val += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            val_loss = running_loss_val / len(test_loader)

            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        print(f'Finished Training for Fold {fold+1}')



def predict_sample(image_path, model_state):
    """
    Parameters: path to an image, and the saved model state
    Returns: Prediction of cancer class from CNN classifier, and probabilities of other classes
    """
    # Load and preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = ImageOps.fit(image, (64, 64), Image.ANTIALIAS)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension


    # Instantiate the model
    model = BrainTumorClassifier()

    # Load the model state
    checkpoint = torch.load(model_state, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Use the model to make predictions
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)[0]

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    # Print prediction
    if predicted_class.item() == 0:
        print(predicted_class.item(),": Predicted glioma")
    elif predicted_class.item() == 1:
        print(predicted_class.item(),": Predicted meningionma")
    elif predicted_class.item() == 2:
        print(predicted_class.item(),": Predicted no tumor")
    elif predicted_class.item() == 3:
        print(predicted_class.item(),": Predicted pituitary")
    
    print("Probabilities of prediction:", probabilities.numpy())

    # Plot image (for checking)
    image_array = np.asarray(image)
    plt.imshow(image_array)
    plt.show()

    return predicted_class.item(), probabilities.numpy()



if __name__ == "__main__":
    # Train image normalization
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Test image normalization
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, test_loader = get_data_loaders(train_transforms, test_transforms)

    #grid_search(train_loader)

    train_dataset = datasets.ImageFolder(root='archive/Training', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root='archive/Testing', transform=test_transforms)
    # Train the model
    #subset_testing(train_dataset, test_dataset)

    #full_training_validation(train_dataset, test_dataset)

    # Command line functionality
    

    # Use the saved trained model state to classify the provided sample
    predict_sample("sample_nt.jpg", "subset_trained_model.pth")
    
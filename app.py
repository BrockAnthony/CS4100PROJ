import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from skorch import NeuralNetClassifier
import urllib3

# Defining BrainTumorClassifier
class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened output after conv and pooling layers
        # self.flattened_size = (64 // (2**3)) * (64 // (2**3)) * 128
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

def predict_image(image_path, model_state_path):
    """
    Parameters: path to an image, and the saved model state path
    Returns: Model tumor prediction on the image, and probabilities of other predictions
    """
    # Load and preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Instantiate and load the model
    model = BrainTumorClassifier()
    model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    model.eval()

    # Use the model to make predictions
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        _, predicted_class = torch.max(output, 1)

    # Print prediction
    tumor_types = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    print(f"Predicted Class: {tumor_types[predicted_class.item()]}")
    print(f"Probabilities: {probabilities.numpy()}")

    # Plot image (for checking)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Class: {tumor_types[predicted_class.item()]}")
    plt.show()

    return predicted_class.item(), probabilities.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain tumor categories for an image.')
    parser.add_argument('image_path', type=str, help='Path to the image for classification.')
    args = parser.parse_args()

    # Replace the following path with the actual model path if different
    model_state_path = '/Users/brockada/CS4100PROJ/final_model.pth'
    predict_image(args.image_path, model_state_path)
# CNNs for Brain Tumor Classification

## Overview

This repository hosts the Brain Tumor Classifier project, a deep learning application designed to classify MRI images into one of four categories: glioma, meningioma, no tumor, and pituitary tumor. Utilizing a Convolutional Neural Network (CNN) and PyTorch, this project aims to provide accurate classifications to aid medical diagnoses.

## Dataset

The dataset comprises MRI images in four distinct classes: glioma, meningioma, no tumor, and pituitary tumor. Images are preprocessed for training, including resizing, normalization, and data augmentation techniques like random flips, rotations, and color adjustments.

## Model Architecture

The core of the project is the `BrainTumorClassifier`, a CNN model with several convolutional and pooling layers, followed by fully connected layers. The model uses ReLU activation and a softmax output layer for classifying the images into the respective categories.

## Training Process

- **Model Training**: The model is trained over 25 epochs using cross-entropy loss and the Adam optimizer.
- **Validation Strategy**: Performance is evaluated using training and validation loss and accuracy metrics.
- **Grid Search**: Hyperparameters are optimized using GridSearchCV, integrated with the model using a Skorch wrapper.
- **Data Handling**: Custom functions are implemented for handling and loading the data, including utilities for denormalization and visualization.

## Evaluation

- **Model Accuracy**: The model's accuracy is tested on a separate test dataset, and metrics like precision, recall, and F1-score are computed.
- **Confusion Matrix**: A confusion matrix is generated to visualize the performance across different classes.
- **Loss and Accuracy Trends**: The trends in training and validation loss and accuracy are plotted for each fold of the cross-validation.

## Usage

The final trained model is saved and can be used to make predictions on new MRI images. The repository includes scripts for evaluating the model's performance and visualizing its predictions.

## Future Work

Further improvements are planned, including:
- Expanding the dataset for better generalization.
- Enhancing the usability and accessibility of the model for practical medical applications.

## Dependencies

- PyTorch
- Sklearn
- Matplotlib
- PIL
- Numpy
- Pandas

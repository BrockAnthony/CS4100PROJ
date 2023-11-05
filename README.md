# Brain Tumor Classifier

This repository contains the implementation of a Convolutional Neural Network (CNN) model for classifying brain tumor images into four categories: glioma, meningioma, no tumor, and pituitary tumor.

## Project Overview

The project uses PyTorch to create and train a deep learning model on brain MRI images. The aim is to accurately classify images into the respective tumor categories to assist in medical diagnoses.

## Dataset

The dataset used for training the model consists of MRI images categorized into four classes: glioma, meningioma, no tumor, and pituitary tumor. The images are preprocessed and normalized before being fed into the model.

## Model Architecture

The model is a CNN with multiple convolutional and pooling layers, followed by fully connected layers. ReLU activation functions are used, along with a softmax output layer to classify the images into the four categories.

## Training

The model is trained using a cross-entropy loss function and the Adam optimizer. Training is performed over 25 epochs, and the performance is evaluated based on training and validation loss and accuracy.
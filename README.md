# Teeth_Disease_Classifier

This project implements a Convolutional Neural Network (CNN) model based on a ResNet architecture to classify various teeth diseases from images. The model is trained using TensorFlow and Keras, with data augmentation and regularization techniques to enhance its performance.

# Project Overview
This project aims to develop a model capable of classifying images of teeth into different disease categories. The dataset consists of labeled images organized into separate folders for training, validation, and testing. The model is built using a residual network (ResNet) structure, which helps in handling deeper models more effectively.

# Prerequisites
    -Python 3.x
    -TensorFlow 2.x
    -Keras
    -numpy
    -matplotlib
# Data Augmentation and Preprocessing
The training data was augmented to increase the diversity of the dataset, helping the model generalize better. Augmentation techniques included rotation, zoom, and horizontal flipping.

The images were also normalized by scaling pixel values to the range [0, 1].

# Training
The model was trained over 200 epochs with the following configuration:

Batch Size: 32
Optimizer: Adam
Loss Function: sparse_categorical_crossentropy
Metrics: Accuracy

# Model Architecture
The model uses a architecture, which includes residual blocks that help in training deeper networks by allowing gradients to bypass certain layers, thereby mitigating the vanishing gradient problem.

# Residual Block Structure
Each residual block consists of two convolutional layers followed by a skip connection that adds the input of the block to the output.

# Model Performance
The model achieved an accuracy of 87%, indicating a high level of performance in classifying the various teeth diseases.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

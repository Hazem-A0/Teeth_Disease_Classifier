# Teeth_Disease_Classifier

This project implements a Convolutional Neural Network (CNN) model based on a ResNet architecture to classify various teeth diseases from images. The model is trained using TensorFlow and Keras, with data augmentation and regularization techniques to enhance its performance.

# Project Overview
This project aims to develop a model capable of classifying images of teeth into different disease categories. The dataset consists of labeled images organized into separate folders for training, validation, and testing. The model is built using a residual network (ResNet) structure, which helps in handling deeper models more effectively.

# Prerequisites
Python 3.x
TensorFlow 2.x
Keras

# Augmentation
Data augmentation is performed during training to generate new images from the existing training set, helping to prevent overfitting.

# Model Architecture
The model uses a ResNet-based architecture, which includes residual blocks that help in training deeper networks by allowing gradients to bypass certain layers, thereby mitigating the vanishing gradient problem.

# Residual Block Structure
Each residual block consists of two convolutional layers followed by a skip connection that adds the input of the block to the output.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

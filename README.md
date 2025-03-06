# Malaria-Classifier
This project implements a Malaria Classifier using a Convolutional Neural Network (CNN). The model is designed to classify cell images into two categories: Parasitized and Uninfected, based on microscopic images of red blood cells.

## Dataset

The dataset used for training the CNN model is the **[MALARIA](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)** dataset, The dataset contains 2 folders Infected, Uninfected and a total of 27,558 images.

## Model Architecture

The model is based on a Convolutional Neural Network (CNN) that consists of the following layers:

1. **Convolutional layers** with ReLU activation functions.
2. **MaxPooling layers** for dimensionality reduction.
3. **Batch Normalization** for improving the model’s performance.
4. **Fully connected layers** with Dropout for regularization.
5. **Sigmoid output layer** to classify the input image into one of the binary results.

6. ## Evaluate

![Loss Result]()

![Accuracy Result]()

## Results

The True Label and Predicted Label above the Image.

P = **Parasitized**
U = **Uninfected**

![]()


---

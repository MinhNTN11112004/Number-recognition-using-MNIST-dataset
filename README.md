# Number Recognition Using MNIST Dataset

## Overview

This project implements a number recognition system using the MNIST dataset. The MNIST dataset is a large collection of handwritten digits that is commonly used for training various image processing systems.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The goal of this project is to build a machine learning model that can accurately classify handwritten digits (0-9) from images. The MNIST dataset consists of 70,000 images of handwritten digits, split into 60,000 training images and 10,000 test images.
Also, I have a small game to utilize the model, wanna watch, click [here](https://drive.google.com/file/d/1FEoxcFAh-Y7i9WGd6vM7j2iTJWj2JFfZ/view?usp=sharing​)

## Dataset

The MNIST dataset can be downloaded from [here](http://yann.lecun.com/exdb/mnist/). The dataset contains:

- 60,000 training images
- 10,000 testing images

Each image is a 28x28 pixel grayscale image.

## Methodology

1. **Data Preprocessing**:
   - Load the MNIST dataset.
   - Augment the data to increase the recognition ability of the models.
   - Divide training dataset into training dataset and validation dataset.
   - Normalize the images for better model performance.
   - Combine multiple single-digit images into a single image for YOLO training.
     
2. **Model Selection**:
   - Testing with basic methods such as KNN and SVM
   - Build a fully connected network model using libraries such as TensorFlow or PyTorch.
   - Then testing by using convolutional neural network with layers such as convolutional layers, pooling layers, and fully connected layers.
   - Build a YOLO model to detect multiple digits in one image.

3. **Training**:
   - Train the model on the training dataset and validation dataset.
   - Use different techniques such as KNN, SVM, FCN, CNN for single digit recognition and YOLO for multiple digit recognition.

4. **Evaluation**:
   - Evaluate the model on the test dataset.
   - Analyze the model's accuracy and other performance metrics.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```
pip install numpy matplotlib torch torchvision
```
## Usage

To run the number recognition model, you can execute the following command:
```
python main.py
```
This will train the model and evaluate its performance on the test dataset. You can visualize the results using the provided scripts.

## Results

The model achieved an accuracy of approximately **99.5%** on the test dataset in best model(CNN). You can view the training and validation accuracy over epochs in the generated plots.

The YOLO model's performance was evaluated using mean Average Precision (mAP) as the primary metric. The results are as follows:

- **mAP**: The model achieved a mean Average Precision of **81%**, indicating its effectiveness in detecting and classifying multiple digits within a single image.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yann LeCun for the MNIST dataset.
- PyTorch for the deep learning framework.
- Matplotlib for visualization.

# PyTorch Experiment 3 – CNN on CIFAR-10

## Overview

This experiment implements and evaluates a **Convolutional Neural Network (CNN)** using **PyTorch** for image classification on the **CIFAR-10 dataset**.

The experiment focuses on understanding how CNNs extract image features using convolution and pooling layers, and how different architectural choices affect classification performance.

## Objectives

* Understand the basic architecture of a CNN.
* Perform image classification using PyTorch.
* Visualize feature maps produced by convolution layers.
* Compare **Max Pooling** and **Average Pooling**.
* Study the effect of different activation functions.
* Monitor training and validation accuracy and loss.
* Evaluate the model using a confusion matrix.

## Dataset

The **CIFAR-10** dataset consists of 60,000 colour images belonging to 10 different classes.

* Image size: **32 × 32 pixels**
* Number of classes: **10**
* Training images: **50,000**
* Testing images: **10,000**
* Images per class in the training set: **5,000**

The classes include:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The dataset is balanced, with an equal number of images in each class.

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn

## CNN Architecture

The CNN uses convolutional layers to extract spatial features from the input images. Pooling layers are then used to reduce the spatial dimensions while retaining important information.

The general flow of the network is:

```text
Input Image
     ↓
Convolution
     ↓
Activation Function
     ↓
Pooling
     ↓
Convolution
     ↓
Activation Function
     ↓
Pooling
     ↓
Fully Connected Layer
     ↓
Output Classes
```

## Experiments Performed

### 1. CIFAR-10 Image Visualization

Sample images from the dataset were displayed to understand the different classes and image characteristics.

### 2. Feature Map Visualization

Feature maps generated after convolution were visualized to understand how CNN layers detect patterns such as edges, textures, and shapes.

### 3. Pooling Comparison

Two pooling techniques were compared:

* Max Pooling
* Average Pooling

Both reduce the spatial dimensions of the feature maps, but Max Pooling preserves the strongest activations while Average Pooling produces a smoother representation.

### 4. Activation Function Comparison

Different activation functions were studied, including:

* ReLU
* Sigmoid

ReLU is commonly used in hidden layers because it provides efficient gradient propagation, while Sigmoid produces values between 0 and 1.

### 5. Training and Validation Analysis

Training and validation accuracy and loss were plotted across epochs to understand the learning behaviour of the model and identify possible overfitting.

### 6. Confusion Matrix

A confusion matrix was generated to analyse class-wise classification performance and identify classes that were frequently confused with one another.

## Results

The pooling comparison showed that **Max Pooling performed better than Average Pooling** for this experiment.

| Pooling Method  | Test Accuracy |
| --------------- | ------------: |
| Max Pooling     |    **53.15%** |
| Average Pooling |    **50.28%** |

The results indicate that retaining the strongest activations using Max Pooling helped the CNN learn more useful features for CIFAR-10 classification.

The confusion matrix also showed that most predictions were concentrated along the diagonal, indicating correct classifications. Some visually similar classes, such as **cat and dog** or **automobile and truck**, were more difficult for the model to distinguish.

## Visualizations

The notebook contains visualizations for:

* Sample CIFAR-10 images
* Class distribution
* CNN feature maps
* Max Pooling and Average Pooling outputs
* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Confusion matrix
* ReLU and Sigmoid activation functions

## Conclusion

This experiment demonstrates the basic working of a CNN for image classification using PyTorch. The visualizations help show how convolutional layers extract features and how pooling and activation functions influence the learning process.

The comparison between pooling methods showed that **Max Pooling achieved better performance than Average Pooling**, while the training and validation plots demonstrated the learning behaviour of the CNN over multiple epochs.

Overall, the experiment provides a practical understanding of CNN components and their role in image classification.

## How to Run

1. Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

2. Open the notebook:

```text
DeepLearning_Experiment3_Pytorch (1).ipynb
```

3. Run the cells sequentially.

4. The CIFAR-10 dataset will be downloaded automatically through `torchvision` if it is not already available.

## Project Structure

```text
.
├── DeepLearning_Experiment3_Pytorch (1).ipynb
└── README.md
```

## Author

**Pavithra Ramesh Babu**

AI & Data Science

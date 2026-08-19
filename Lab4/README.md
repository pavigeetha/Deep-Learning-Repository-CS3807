# CNN Architecture Comparison and VGG16 Transfer Learning on CIFAR-10

## Overview

This project explores image classification on the **CIFAR-10 dataset** using different Convolutional Neural Network (CNN) architectures and transfer learning techniques.

The experiment focuses on:

- Comparing different CNN architectures
- Implementing transfer learning using pretrained VGG16 and ResNet50
- Fine-tuning VGG16
- Analysing misclassified images
- Studying the effect of different hyperparameter configurations
- Comparing models based on accuracy, parameter count, and training time

## Dataset

The experiment uses the **CIFAR-10 dataset**, which contains 60,000 colour images belonging to 10 classes.

The classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into:

- 50,000 training images
- 10,000 testing images
- Image size: `32 × 32 × 3`

The pixel values are normalized to the range `[0, 1]`, and the class labels are converted to one-hot encoded vectors.

## Models Implemented

The experiment implements and evaluates the following architectures:

### LeNet-5

A classic CNN architecture consisting of:

- Convolutional layers
- Average pooling layers
- Fully connected layers
- `tanh` activations
- Softmax output layer

### AlexNet

A deeper CNN architecture containing:

- Multiple convolutional layers
- Max pooling
- Fully connected layers
- ReLU activations
- Dropout for regularization
- Softmax output layer

### VGG16

VGG16 is used through transfer learning with ImageNet pretrained weights.

The CIFAR-10 images are resized from `32 × 32` to `224 × 224` before being passed to the pretrained network.

The original VGG16 classification head is removed and replaced with:

- Global Average Pooling
- Dense layer with 128 units
- 10-class softmax output

Initially, the pretrained VGG16 layers are frozen and only the newly added classification layers are trained.

### GoogleNet

A simplified GoogleNet-style architecture is implemented using custom Inception blocks.

Each Inception block contains parallel branches using:

- `1 × 1` convolution
- `3 × 3` convolution
- `5 × 5` convolution
- Max pooling followed by `1 × 1` convolution

The outputs of the branches are concatenated before being passed to subsequent layers.

### ResNet50

ResNet50 is used with ImageNet pretrained weights.

The model uses:

- ResNet50 convolutional base
- Global Average Pooling
- Dense layer with 128 units
- 10-class softmax output

The pretrained base is initially frozen during training.

## VGG16 Transfer Learning

The VGG16 experiment follows a transfer learning workflow.

### Initial Training

The pretrained VGG16 convolutional base is frozen, and a new classification head is added for CIFAR-10 classification.

The model is trained using:

- Adam optimizer
- Learning rate: `0.001`
- Batch size: `32`
- Epochs: `10`
- Categorical cross-entropy loss

### Fine-Tuning

After the initial training phase, fine-tuning is performed by:

1. Unfreezing the VGG16 base
2. Freezing all layers initially
3. Making the layers belonging to `block5` trainable
4. Recompiling the model
5. Reducing the learning rate to `0.0001`
6. Training for an additional 5 epochs

This allows the higher-level pretrained features to adapt to the CIFAR-10 dataset while keeping most of the pretrained representation fixed.

## Model Evaluation

The classification models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification report
- Confusion matrix

For VGG16, evaluation is performed both **before and after fine-tuning**.

The experiment also identifies misclassified test images and visualizes them along with their true and predicted class labels.

## Training Visualizations

The VGG16 experiment generates training and validation plots for:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

The plots also indicate the point at which fine-tuning begins.

Sample CIFAR-10 images and VGG16 misclassified images are also visualized during the experiment.

## CNN Architecture Comparison

The implemented architectures are compared using:

- Number of trainable and non-trainable parameters
- Test accuracy
- Training time

The comparison includes:

| Architecture | Approach |
|---|---|
| LeNet-5 | CNN trained from scratch |
| AlexNet | CNN trained from scratch |
| VGG16 | ImageNet transfer learning |
| GoogleNet | Custom Inception-style CNN |
| ResNet50 | ImageNet transfer learning |

## Hyperparameter Study

A hyperparameter study is performed using VGG16 transfer learning.

A subset of **10,000 CIFAR-10 training images** is used for the study, along with a separate validation subset of **2,000 images**.

Each configuration is trained for 10 epochs.

The hyperparameters investigated include:

- Learning rate
- Batch size
- Optimizer
- Dense layer size
- Frozen or partially trainable VGG16 layers

### Configurations

Five configurations are evaluated:

| Learning Rate | Batch Size | Optimizer | Dense Units | VGG16 Layers |
|---:|---:|---|---:|---|
| 0.001 | 32 | Adam | 128 | All frozen |
| 0.0001 | 32 | Adam | 128 | All frozen |
| 0.001 | 64 | Adam | 256 | All frozen |
| 0.001 | 32 | SGD | 128 | All frozen |
| 0.0001 | 32 | Adam | 256 | Partial fine-tuning |

For each configuration, the experiment records:

- Best validation accuracy
- Final validation accuracy
- Training time
- Number of parameters

The configurations are then sorted according to their best validation accuracy to identify the most effective hyperparameter combination.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
.
├── Experiment_4.ipynb
├── sample_data/
│   ├── 01_sample_cifar10_images.png
│   ├── vgg16_misclassified_images.png
│   ├── vgg16_training_accuracy.png
│   ├── vgg16_validation_accuracy.png
│   ├── vgg16_training_loss.png
│   ├── vgg16_validation_loss.png
│   ├── vgg16_hyperparameter_study.csv
│   └── cnn_architecture_comparison.csv
└── README.md
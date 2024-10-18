# Brain Tumor Detection using Deep Learning

This repository contains a deep learning model for detecting brain tumors in MRI images. The project leverages a classification model for initial defect detection and a segmentation model for localizing the tumor.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Brain tumors are a critical health concern, and early detection plays a vital role in treatment. This project aims to develop a model that accurately detects brain tumors from MRI scans using Convolutional Neural Networks (CNNs).

## Dataset

The dataset consists of MRI images with corresponding masks indicating the presence of tumors. It is divided into training, validation, and test sets.

## Model Architecture

The project includes two main models:

1. **Classification Model**: This model predicts whether a brain MRI contains a tumor.
2. **Segmentation Model**: If the classification model indicates a possible tumor, this model localizes the tumor in the MRI image.

### Classification Model

The classification model is built using a CNN architecture with several convolutional layers, batch normalization, and activation functions.

### Segmentation Model

The segmentation model is based on a modified U-Net architecture with residual blocks to enhance feature learning.

## Training

The models were trained using TensorFlow and Keras. Key training parameters include:

- **Optimizer**: Adam
- **Loss Function**: Focal Tversky Loss
- **Metrics**: Tversky Score

Training callbacks include early stopping and model checkpointing to save the best model.

## Evaluation

The models were evaluated on the test set, and the following metrics were reported:

- **Classification Accuracy**: Achieved high accuracy in detecting the presence of tumors.
- **Segmentation Performance**: Evaluated using the Tversky score, indicating the model's effectiveness in localizing tumors.

## Usage

To use the trained models:

1. Load the model architecture from the JSON file.
2. Load the model weights.
3. Pass MRI images through the classification model for initial prediction.
4. If a tumor is detected, use the segmentation model to localize it.

## Conclusion

This project demonstrates the capability of deep learning models in detecting and localizing brain tumors from MRI scans. Future work could focus on improving the models' performance and exploring different architectures.

## Requirements

To run this project, you need the following libraries:

- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- seaborn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

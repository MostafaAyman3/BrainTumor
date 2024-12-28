# Brain Tumor Classification

## Overview
This project aims to classify brain tumors using MRI images. By employing various image processing techniques and machine learning models, specifically Support Vector Machines (SVM), the system predicts tumor types from input images.

## Dataset
The dataset used for this project is sourced from Kaggle and includes MRI images categorized into four types:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

# Brain Tumor Classification using Image Processing and Machine Learning

## Overview
This project aims to classify brain tumors using MRI images. By employing various image processing techniques and machine learning models, specifically Support Vector Machines (SVM), the system predicts tumor types from input images with high accuracy. The project implements a comprehensive pipeline that includes image preprocessing, feature extraction, and machine learning classification.

## Dataset
The dataset used for this project is sourced from Kaggle and includes MRI images categorized into four types:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

## Technical Implementation

### Image Processing Pipeline
1. **Image Preprocessing**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
   - Image resizing to standardized dimensions (256x256)
   - Grayscale conversion for consistent processing

2. **Feature Extraction**:
   - Local Binary Patterns (LBP) for texture analysis
   - Contour detection for tumor boundary identification
   - Morphological features extraction including:
     - Area and perimeter measurements
     - Centroid coordinates
     - Circularity metrics
     - Eccentricity
     - Major/minor axis ratios

3. **Feature Engineering**:
   - Standardization of features
   - Dimensionality reduction using PCA (70 components)
   - Feature vector padding for consistent dimensions

### Machine Learning Model
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Hyperparameter Optimization**: Grid Search with Cross-Validation
- **Model Parameters**:
  - Balanced class weights for handling class imbalance
  - Optimized C and gamma parameters
  - 5-fold cross-validation

### Data Split
- Training and testing sets split with 80:20 ratio
- Stratified splitting to maintain class distribution

## Model Training and Evaluation
The model undergoes rigorous training with hyperparameter tuning using GridSearchCV, ensuring optimal performance. Evaluation metrics include precision, recall, and F1-score for each tumor category.

### Data Acquisition
Data was downloaded from Kaggle using the following commands:

```bash
!pip install kaggle
files.upload()  # Upload your kaggle.json file
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
!unzip brain-tumor-classification-mri.zip -d ./brain_tumor_dataset
Project Structure
Brain_Tumor_Classification.ipynb: The Jupyter notebook containing the code for data processing, model training, and evaluation.
brain_tumor_dataset/: Directory containing the training images.
Methodology
Data Preprocessing:

Load images and their labels.
Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for image enhancement.
Resize images to a uniform size of 128x128 pixels.
Feature Extraction:

Local Binary Patterns (LBP) were calculated for texture representation.
Morphological features were computed using contour detection algorithms.
Model Training:

The dataset was split into training and testing sets.
Labels were encoded using LabelEncoder.
A Support Vector Machine model was trained using a pipeline that included data scaling and Principal Component Analysis (PCA) for dimensionality reduction.
Hyperparameter Tuning:

Grid search was employed to find the optimal parameters for the SVM model.
Evaluation:

The model's performance was evaluated using a classification report, which included precision, recall, and F1-score for each class.
Installation
To run the project, ensure you have the following libraries installed:

OpenCV
NumPy
scikit-learn
Matplotlib
You can install these libraries using pip:

bash
Copy code
pip install opencv-python numpy scikit-learn matplotlib
Results
The best parameters found for the SVM model were:

C: (best value)
Gamma: (best value)
The classification report indicated the model's accuracy across the different tumor classes.

Usage
To use this project:

Clone the repository to your local machine.
Run the Jupyter notebook in a compatible environment (e.g., Google Colab, Jupyter Notebook).
Follow the steps outlined in the notebook to preprocess the data and train the model.

To use this project:

Clone the repository to your local machine.
Run the Jupyter notebook in a compatible environment (e.g., Google Colab, Jupyter Notebook).
Follow the steps outlined in the notebook to preprocess the data and train the model.

# 🖥️ Deep Learning Assignment 1

Welcome to the Deep Learning Assignment 1 repository for CSL7590. This assignment involves implementing a Convolutional Neural Network (CNN) from scratch using PyTorch to solve multi-class classification problems. Follow the guidelines and instructions provided below.

## 📄 Table of Contents
- [General Instructions](#general-instructions)
- [Submission Guidelines](#submission-guidelines)
- [Objective](#objective)
- [Network Architecture](#network-architecture)
- [Training Configuration](#training-configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Solution Summary](#solution-summary)
- [Contact Information](#contact-information)

---

## General Instructions 📋
1. Clearly mention any assumptions you have made.
2. Report any resources you used while attempting the assignment.
3. Submissions in any other format or after the deadline will not be evaluated.
4. Add references to the resources used.
5. Plagiarism will result in zero marks.
6. Select your dataset correctly.

## Submission Guidelines 📑
1. Prepare a Python code file named `YourRollNo.py`. There should be only one .py file.
2. Submit a single report named `YourRollNo.pdf`, containing methods, results, and observations.
3. Provide your Colab file link in the report.
4. Upload both the code and report directly on Google Classroom.
5. Do not upload .ipynb files or screenshots in the report.

## Objective 🎯
Implement a Convolutional Neural Network from scratch in Python using PyTorch. The network should be able to train on a simple dataset for multi-class classification.

## Network Architecture 🏗️
1. **Convolution Layer 1**: Kernel Size=7x7, Maxpool, Stride=1, Output Channels=16
2. **Convolution Layer 2**: Kernel Size=5x5, Maxpool, Stride=1, Output Channels=8
3. **Convolution Layer 3**: Kernel Size=3x3, Average Pooling, Stride=2, Output Channels=4
4. **Output Layer**: Softmax activation, Output size=number of classes

Use zero padding to preserve the input image dimension.

## Training Configuration ⚙️
1. **Batch Size**: Determined by roll number (e.g., Batch Size = 20 for roll number ending in 23).
2. **Activation Functions**: ReLU for convolution layers, Softmax for the output layer.
3. **Optimizer**: Adam
4. **Loss Function**: Cross-Entropy
5. **Training Epochs**: 10 (or more if accuracy is low).

## Experiments 🧪
### Experiment 1: 10-Class Classification
- Train a CNN on the MNIST dataset (handwritten digits).
- Dataset: 60k images for training, 10k images for testing.
- Report accuracy and loss per epoch.
- Prepare a confusion matrix for the test set.
- Calculate total trainable and non-trainable parameters.

### Experiment 2: 4-Class Classification
- Combine digit images into the following classes:
  - Class 1: {0, 6}
  - Class 2: {1, 7}
  - Class 3: {2, 3, 8, 5}
  - Class 4: {4, 9}
- Use the CNN model from Experiment 1 to solve the 4-class classification problem.

## Results 📊
1. **Accuracy and Loss per Epoch**: Graphs showing the training progress.
2. **Confusion Matrix**: Visual representation of model performance on the test set.
3. **Model Parameters**: Calculation of trainable and non-trainable parameters.

## Solution Summary 📝

### Experiment 1: 10-Class Classification
- **Dataset**: MNIST dataset with 60k training and 10k testing images.
- **Network Architecture**: Three convolution layers followed by a fully connected output layer.
- **Training Configuration**: Batch Size = 20, ReLU activation, Adam optimizer, Cross-Entropy loss, 10 epochs.
- **Results**:
  - Accuracy and loss plotted per epoch.
  - Confusion matrix created for test set.
  - Total trainable parameters: Calculated based on the model architecture.

### Experiment 2: 4-Class Classification
- **Dataset Modification**: Combined digit images into 4 classes.
- **Network Architecture**: Same as Experiment 1 with output layer adjusted for 4 classes.
- **Training Configuration**: Same as Experiment 1.
- **Results**:
  - Accuracy and loss plotted per epoch.
  - Confusion matrix created for test set.
  - Total trainable parameters: Calculated based on the model architecture.

### Performance Improvement (Bonus)
- Implemented techniques to improve performance and avoid overfitting:
  - Dropout
  - Data Augmentation
  - Learning Rate Scheduling
  - Batch Normalization
  - Early Stopping


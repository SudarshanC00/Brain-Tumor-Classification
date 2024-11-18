# Brain Tumor Classification Using PyTorch
## Overview
This project implements a deep learning model using PyTorch to classify brain MRI images into four different tumor types. The goal is to assist in early diagnosis, which can be valuable for healthcare professionals in planning treatment. Two versions of a ResNet-based model were developed, evaluated, and optimized to maximize classification accuracy.

## Dataset
![1999](https://github.com/user-attachments/assets/4e5f11f8-00c0-4cd2-8f89-d8777ba655ff)

- **Total Images**: 7,023
- **Classes**: 4 (glioma, healthy, meningioma, pituitary)
- **Train/Test Split**: 5,618 images for training, 1,405 for testing
- **Preprocessing**: Images resized to 224x224 and normalized to ImageNet standards.

## Model Versions and Adjustments

### Model Version 1
- **Base Model**: ResNet10 with modified pooling and classification layers.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Scheduler**: StepLR with a gamma of 0.1, stepping every 7 epochs.
- **Results**:
  - Training Accuracy: 98.1%
  - Testing Accuracy: 94.2%
  - Training Loss: ≈ 0.05
  - Testing Loss: ≈ 0.06

*Note*: The training accuracy increased consistently, though testing accuracy fluctuated, suggesting some overfitting.

### Model Version 2
- **Modifications**: Switched to SGD with momentum (0.9), increased dropout (0.4), and used a MultiStepLR scheduler.
- **Results**:
  - Training Accuracy: 98.7%
  - Testing Accuracy: 95.4%
  - Training Loss: ≈ 0.04
  - Testing Loss: ≈ 0.05

This version showed improved generalization with reduced loss and higher testing accuracy.

## Model Export and Inference Optimization
- The model was exported in ONNX format for deployment.
- TorchScript was applied for inference optimization, achieving slight improvements in inference speed.

## Conclusion
Both models achieved high accuracy, with Version 2 performing slightly better on testing data. This classifier demonstrates promise for use in medical imaging analysis and could potentially support healthcare applications in diagnosing brain tumor types.

## Links
- [Dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)

#Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->

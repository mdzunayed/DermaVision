# DermaVision: A Deep Learning Framework for Skin Cancer Detection

## 1. Abstract

DermaVision is an interpretable and explainable deep learning framework designed for the binary classification of skin lesions, distinguishing between benign and malignant cases. This system specifically utilizes the **Xception** convolutional neural network (CNN) architecture, which has been fine-tuned on **Dataset 1 (D1)**, a clinically validated dermoscopic image dataset. The model's ability to identify key features in lesions is enhanced through the use of Grad-CAM visualization, increasing clinician confidence. Our results demonstrate that the Xception model, when applied to D1, achieves a high level of accuracy and interpretability, making it a powerful tool for assisting dermatologists in early skin cancer detection.

## 2. Introduction

Early and accurate diagnosis is critical for improving patient prognosis and reducing healthcare costs associated with skin cancer, particularly melanoma. While traditional visual inspection can be subjective, deep learning models like DermaVision offer a consistent and rapid diagnostic aid. This system is designed to overcome common challenges such as high data variability by leveraging the efficient architecture of the Xception model and focusing on the well-balanced D1 dataset.

## 3. Methodology: The Xception Model & Dataset 1

This project exclusively uses the **Xception** CNN, which is based on depthwise separable convolutions. This architectural choice allows for efficient performance while maintaining a high level of accuracy. The model was trained and evaluated solely on **Dataset 1 (D1)**, a collection of 10,605 dermoscopic images that are nearly evenly split between benign and malignant lesions.

The training process involved careful data preprocessing, including resizing images to $299 \times 299$ pixels to align with the Xception architecture's design. This ensures the model can learn and distinguish the subtle visual differences between lesion types.

## 4. User Interface

The web application provides a simple and intuitive user interface to interact with the deep learning model.

### 4.1 Upload Page

The main page of the application allows users to upload a dermoscopic image. The clean, drag-and-drop interface makes the process straightforward.

![DermaVision Upload Page](https://github.com/mdzunayed/DermaVision/blob/main/image/web_1.png)

### 4.2 Prediction Results

After analysis, the application displays a detailed results page. This includes the model's prediction, a confidence score, and a Grad-CAM visualization to show which parts of the image were most relevant to the prediction.

![DermaVision Prediction Results](https://github.com/mdzunayed/DermaVision/blob/main/image/web_2.png)

## 5. Performance on Dataset 1

The Xception model's performance on Dataset 1 was evaluated using a suite of clinically relevant metrics. Our results demonstrate the model's effectiveness in binary classification.

The key performance metrics are as follows:

* **Specificity**: 0.94
* **Matthews Correlation Coefficient (MCC)**: 0.86
* **Precision-Recall AUC (PR-AUC)**: 0.93
* **F1-Score**: 0.92

These results highlight the Xception model's robust ability to accurately identify malignant cases while minimizing false positives, a critical factor for clinical decision support.

## 6. Future Directions

This work provides a strong foundation for an AI-powered diagnostic tool. Future efforts will focus on external validation and deployment in teledermatology applications, with the goal of making expert-level skin cancer screening accessible to a wider patient base.

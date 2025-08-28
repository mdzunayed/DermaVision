# DermaVision: A Deep Learning Framework for Skin Cancer Detection

## 1. Abstract

DermaVision is an interpretable and explainable deep learning framework designed for the binary classification of skin lesions, distinguishing between benign and malignant cases. This system specifically utilizes the **Xception** convolutional neural network (CNN) architecture, which has been fine-tuned on **Dataset 1 (D1)**, a clinically validated dermoscopic image dataset. The model's ability to identify key features in lesions is enhanced through the use of Grad-CAM visualization, increasing clinician confidence. Our results demonstrate that the Xception model, when applied to D1, achieves a high level of accuracy and interpretability, making it a powerful tool for assisting dermatologists in early skin cancer detection.

## 2. Introduction

Early and accurate diagnosis is critical for improving patient prognosis and reducing healthcare costs associated with skin cancer, particularly melanoma. While traditional visual inspection can be subjective, deep learning models like DermaVision offer a consistent and rapid diagnostic aid. This system is designed to overcome common challenges such as high data variability by leveraging the efficient architecture of the Xception model and focusing on the well-balanced D1 dataset.

## 3. Methodology: The Xception Model & Dataset 1

This project exclusively uses the **Xception** CNN, which is based on depthwise separable convolutions. This architectural choice allows for efficient performance while maintaining a high level of accuracy. The model was trained and evaluated solely on **Dataset 1 (D1)**, a collection of 10,605 dermoscopic images that are nearly evenly split between benign and malignant lesions.

The training process involved careful data preprocessing, including resizing images to $299 \\times 299$ pixels to align with the Xception architecture's design. This ensures the model can learn and distinguish the subtle visual differences between lesion types.

### Dataset

The model was trained on the **Melanoma Skin Cancer Dataset** from Kaggle, which can be found here: <https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images>. The dataset was split for training and evaluation as follows:

* **Training**: 80% (8,000 images)
* **Testing**: 15% (1,500 images)
* **Validation**: 5% (500 images)

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

## 6. Installation and Usage

### Clone with Git

Follow these steps to set up and run the project locally. This process assumes you have Python and Git installed.

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/mdzunayed/DermaVision.git](https://github.com/mdzunayed/DermaVision.git)
    ```

2.  **Navigate to the Project Directory**

    ```bash
    cd DermaVision
    ```

3.  **Create a Virtual Environment**

    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv

    # For Windows:
    .\venv\Scripts\activate

    # For macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies**

    Install all the required Python packages from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application**

    Start the Flask server.

    ```bash
    python app.py
    ```

    The application will now be running on `http://127.0.0.1:5000`.

### Download as a ZIP File

Alternatively, you can download the repository directly without using Git.

1.  Go to the [DermaVision GitHub page]([https://www.google.com/search?q=https://github.com/mdzunayed/DermaVision](https://github.com/mdzunayed/DermaVision/)).
2.  Click the green **`< > Code`** button.
3.  Click **`Download ZIP`**.
4.  Extract the contents of the ZIP file to your preferred location.
5.  Follow steps 2-5 from the **Clone with Git** section to run the application.

### Usage

1.  Navigate to the application URL in your web browser.
2.  Upload an image of a dermoscopic skin lesion.
3.  The application will process the image using the deployed model and provide a classification result along with a confidence score.

## 7. Future Directions

This work provides a strong foundation for an AI-powered diagnostic tool. Future efforts will focus on external validation and deployment in teledermatology applications, with the goal of making expert-level skin cancer screening accessible to a wider patient base.

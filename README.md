Cats vs Dogs Image Classification using CNN

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The model is trained on the Cats vs Dogs dataset from TensorFlow Datasets (TFDS) and achieves 82% test accuracy.

📌 Project Overview

This project demonstrates an end-to-end deep learning pipeline for binary image classification.

It includes:

Data loading and preprocessing

CNN model design and implementation

Model training and validation

Performance evaluation using multiple metrics

Confusion matrix visualization

Real-time prediction on custom images

The goal is to build a reliable image classifier that distinguishes between cats and dogs.

🧠 Model Architecture

The Convolutional Neural Network consists of:

Conv2D (32 filters, 3×3) + ReLU

MaxPooling2D

Conv2D (64 filters, 3×3) + ReLU

MaxPooling2D

Conv2D (128 filters, 3×3) + ReLU

MaxPooling2D

Flatten

Dense (128 units, ReLU)

Dropout (0.5)

Output Layer (1 unit, Sigmoid activation)

Loss Function: Binary Crossentropy
Optimizer: Adam

⚙️ Data Preprocessing

Dataset loaded using TensorFlow Datasets (TFDS)

80% training / 20% testing split

Images resized to 150×150

Pixel values normalized to range [0,1]

Batching and prefetching for performance optimization

📊 Model Performance

Test Accuracy: 82%

Loss: ~0.43

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

The confusion matrix is visualized using Seaborn heatmaps.

🚀 Features

✔ End-to-end deep learning workflow
✔ CNN-based binary image classification
✔ Performance evaluation with multiple metrics
✔ Confusion matrix visualization
✔ Real-time prediction on uploaded images
✔ Implemented and tested using Google Colab

🛠️ Technologies Used

Python

TensorFlow

Keras

NumPy

Scikit-learn

Matplotlib

Seaborn

Google Colab

📂 Project Structure
├── model_training.ipynb / .py
├── evaluation_metrics.py
├── README.md
▶️ How to Run

Clone the repository:

git clone https:[//github.com/your-username/cats-vs-dogs-classification.git](https://github.com/OmarAhmed770/Cats-vs-Dogs-Image-Classification-CNN-.git)

Install dependencies:

pip install tensorflow tensorflow-datasets numpy scikit-learn matplotlib seaborn

Run the notebook or Python script to train the model.

Upload a custom image to test real-time prediction.

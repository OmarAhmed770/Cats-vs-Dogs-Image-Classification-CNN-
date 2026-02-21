# Cats-vs-Dogs-Image-Classification-CNN-
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The model is trained on the Cats vs Dogs dataset from TensorFlow Datasets (TFDS) and achieves 82% test accuracy.

📌 Project Overview

The goal of this project is to build a deep learning model capable of distinguishing between cat and dog images using supervised learning. The project covers the full machine learning pipeline including data preprocessing, model building, training, evaluation, and prediction.

🧠 Model Architecture

The CNN architecture consists of:
3 Convolutional (Conv2D) layers with ReLU activation
3 MaxPooling layers
Flatten layer
Fully connected Dense layer (128 units)
Dropout layer (0.5) for regularization
Output layer with Sigmoid activation (Binary Classification)

⚙️ Data Preprocessing

Dataset loaded using TensorFlow Datasets (TFDS)
80/20 Train-Test split
Images resized to 150×150
Pixel normalization (scaled to range 0–1)
Batched and optimized using prefetching

📊 Model Performance

Test Accuracy: 82%
Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix Visualization

🚀 Features

End-to-end deep learning pipeline
Model evaluation with multiple performance metrics
Confusion matrix heatmap visualization
Real-time prediction on custom uploaded images (Google Colab supported)

🛠️ Technologies Used

Python
TensorFlow
Keras
NumPy
Scikit-learn
Matplotlib
Seaborn

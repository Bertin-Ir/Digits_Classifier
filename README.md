# Digits_Classifier
## The Problem
The goal of this project is to build a machine learning model capable of automatically identifying handwritten digits (0–9). While humans can easily read numbers written in different styles, computers perceive images only as grids of pixel intensities. The challenge lies in creating an algorithm robust enough to handle the high variance in human handwriting—such as different slants, thicknesses, and shapes—while maintaining high classification accuracy. Solving this problem is a foundational step in computer vision, with real-world applications ranging from bank check processing to automated postal mail sorting

## Methodology

The project was conducted in two distinct experimental phases to demonstrate the impact of architectural choice on computer vision tasks:

### 1. Baseline Approach: Multi-Layer Perceptron (MLP)
The initial approach utilized a standard feed-forward architecture. 
* **Preprocessing:** Images were flattened into 1D vectors ($28 \times 28 = 784$ pixels).
* **Architecture:** Multiple fully connected (dense) layers with ReLU activations.
* **Limitation:** The MLP failed to capture spatial hierarchies, treating neighboring pixels the same as distant ones, which limited its ultimate accuracy.

### 2. Advanced Approach: Convolutional Neural Network (CNN)
To improve performance, we implemented a CNN designed to recognize local patterns such as edges, curves, and shapes.
* **Feature Extraction:** A series of convolutional layers using $3 \times 3$ kernels and **Batch Normalization** to accelerate and stabilize learning.
* **Spatial Reduction:** Strategic use of **Max-Pooling** layers to reduce dimensionality while preserving dominant features.
* **Classification:** A dense head utilizing **Dropout (0.5)** to ensure robust generalization and prevent overfitting.

---

## Performance Comparison

The transition to a convolutional architecture resulted in a dramatic reduction in error rate and significantly faster convergence.

| Model | Test Accuracy |
| :--- | :--- |
| **MLP (Baseline)** | 97.56% | 
| **CNN (Final)** | **99.33%** |

---

## Detailed CNN Metrics

The final CNN model achieved near-perfect scores across all 10 digit classes. The high F1-scores indicate that the model is exceptionally balanced, with virtually no bias toward specific digits.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.99      1.00      1.00      1135
           2       0.99      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.99      1.00      1.00       982
           5       0.99      0.99      0.99       892
           6       1.00      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       1.00      0.99      1.00       974
           9       0.99      0.99      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000


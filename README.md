# Digits_Classifier
A handwritten digit recognizer built in PyTorch, comparing a multi-layer perceptron with a convolutional neural network on MNIST, and deployed as a live app where you can draw a digit or upload a photo of one.
 
**[Try the live demo](https://digitsclassifierb.streamlit.app/)**
![uploading a digit in the live demo](live-demo.jpg)
 
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
````
## From notebook to live demo
 
A model scoring 99% on MNIST does not automatically read a digit drawn in a browser or photographed on paper. MNIST images follow a precise format, and so the demo demands converting inputs to match it.
 
**Drawn digits.** The canvas is white ink on black, like MNIST. Each drawing is cropped to its ink, scaled so its longer side fits a 20×20 box, and placed in a 28×28 frame with its center of mass at the center, which is how MNIST digits were prepared. It is then normalized exactly as in training.
 
**Uploaded photos.** Photos needed much more work. The pipeline:
 
1. Applies the phone's stored rotation.
2. Estimates the paper brightness across the photo and subtracts it, removing uneven lighting and shadows.
3. Removes ruled notebook lines and reconnects any strokes they crossed.
4. Keeps only the digit, discarding specks and marks touching the photo edges.
5. Redraws the strokes at MNIST thickness (about 2.8 pixels in the 28×28 frame, measured on the test set).
 
## Limitations and next steps
 
- **Rotated digits.** The model was trained without augmentation, so it has never seen a tilted digit and struggles with them. Next step: retrain with random rotations, shifts and scaling.
- **One digit at a time.** Multi-digit numbers are not supported yet. Splitting a drawing into separate digits would allow reading numbers such as "42" without retraining.
## Run it locally
 
Requires Python 3.12.
 
```bash
git clone https://github.com/Bertin-Ir/Digits_Classifier.git
cd Digits_Classifier
pip install -r requirements.txt
streamlit run app.py
```
 
```text
├── app.py                    Streamlit app
├── model.py                  MLP and CNN definitions
├── preprocessing.py          converts drawings and photos to MNIST format
├── check_weights.py          checks the trained weights load correctly
├── weights/                  trained model weights
├── digits_classifier.ipynb   training notebook
└── requirements.txt
```
 
## Tech stack
 
PyTorch, Streamlit, NumPy, SciPy, Pillow, scikit-learn, Matplotlib, Seaborn.
 
## Author
 
Bertin Iradukunda


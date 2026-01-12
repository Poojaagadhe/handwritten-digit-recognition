# AI Assignment – Handwritten Digit Recognition (MNIST)

## Project Overview
This project implements and evaluates multiple classical machine learning models for handwritten digit recognition using the MNIST dataset. The objective is to build a complete machine learning pipeline covering data exploration, preprocessing, model training, hyperparameter tuning, evaluation, and comparative analysis.

The assignment focuses on understanding model behavior, performance trade-offs, and error patterns rather than relying on deep learning approaches.

---

## Problem Statement
Given grayscale images of handwritten digits (0–9), the task is to correctly classify each image into its corresponding digit class using supervised machine learning algorithms.

---

## Dataset Description
- **Dataset:** MNIST (CSV format)
- **Training Samples:** 60,000
- **Testing Samples:** 10,000
- **Image Resolution:** 28 × 28 pixels
- **Features:** 784 pixel intensity values
- **Classes:** 10 (digits 0–9)

---

## Workflow and Methodology
The project follows a structured machine learning workflow:

1. **Data Loading and Exploration**
   - Dataset size verification
   - Class distribution analysis
   - Sample image visualization
   - Missing value detection

2. **Data Preprocessing**
   - Pixel normalization (0–1 scaling)
   - Train–test split with stratification
   - Dimensionality reduction using PCA (50 components)

3. **Model Implementation**
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM – RBF Kernel)
   - Decision Tree Classifier

4. **Hyperparameter Tuning**
   - KNN: Number of neighbors (k)
   - SVM: Regularization parameter (C) and gamma
   - Decision Tree: Max depth and minimum samples split

5. **Model Evaluation**
   - Accuracy computation
   - Confusion matrix generation
   - Visualization of misclassified samples

6. **Model Comparison**
   - Performance comparison across all models

7. **Voting Ensemble (Optional)**
   - Hard voting ensemble combining KNN, SVM, and Decision Tree

---

## Models and Performance

| Model | Best Parameters | Accuracy |
|------|----------------|----------|
| KNN | k = 3 | 97.1% |
| SVM (RBF Kernel) | C = 10, gamma = 0.01 | **98.4%** |
| Decision Tree | max_depth = 20, min_samples_split = 10 | 87.8% |
| Voting Ensemble | Hard Voting | 92.6% |

---

## Results and Observations
- The **SVM model achieved the highest accuracy**, demonstrating strong performance on high-dimensional data after PCA.
- KNN produced competitive results but is computationally expensive during prediction due to distance calculations.
- Decision Tree showed comparatively lower accuracy due to overfitting on pixel-level features.
- Common misclassifications were observed between visually similar digits such as:
  - 3 and 5
  - 4 and 9
  - 7 and 1

---

## Error Analysis
Misclassified samples were analyzed visually to understand model limitations. Variations in handwriting style, stroke thickness, and incomplete digit formation were the primary reasons for incorrect predictions.

---

## Tools and Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

## How to Run the Project

### Step 1: Clone the Repository
    git clone <repository-url>
    cd handwritten-digit-recognition

### 2.Install Dependencies
    pip install numpy pandas matplotlib seaborn scikit-learn

### 3. Run the Notebook
 - Open the Jupyter Notebook
 - Execute all cells sequentially

## Conclusion:
 - This assignment demonstrates the effectiveness of classical machine learning models for image classification tasks.
 - Among all evaluated models, Support Vector Machine proved to be the most robust and reliable for handwritten digit recognition.
 - Performance can be further improved using advanced feature extraction techniques or ensemble learning methods.


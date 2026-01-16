
# Neural Network vs Logistic Regression — Iris Classification

## 📌 Project Overview
This project implements a **feedforward Neural Network** from scratch (using PyTorch) to perform **multiclass classification** on the classic **Iris dataset**, and compares its performance against a strong linear baseline: **Logistic Regression**.

The goal is not to “beat” traditional models blindly, but to **understand when and why Neural Networks help**, and when simpler models are equally effective.

---

## 🎯 Problem Statement
Given four flower measurements:
- Sepal length
- Sepal width
- Petal length
- Petal width  

Predict the flower species:
- Setosa  
- Versicolor  
- Virginica  

This is a **3-class classification problem**.

---

## 🧠 Models Implemented

### 1️⃣ Neural Network (PyTorch)
A fully connected feedforward neural network trained using a **manual training loop**.

**Architecture**
```
Input (4 features)
↓
Dense (16) + ReLU
↓
Dense (16) + ReLU
↓
Dense (3 logits)
```

- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Feature Scaling: StandardScaler  
- Labels: Encoded using LabelEncoder  

> Note: Softmax is **not applied explicitly**, as `CrossEntropyLoss` expects raw logits.

---

### 2️⃣ Logistic Regression (Baseline)
A classic linear classifier implemented using **scikit-learn**, serving as a strong baseline for comparison.

- Multinomial Logistic Regression  
- Max iterations: 200  

---

## 📊 Results

| Model               | Accuracy |
|--------------------|----------|
| Neural Network     | **0.9333** |
| Logistic Regression| **0.9333** |

### Interpretation
- The Iris dataset is **nearly linearly separable**
- Logistic Regression performs extremely well
- The Neural Network matches the baseline, demonstrating **correct learning and generalization**
- This highlights an important ML principle:

> **More complex models do not always outperform simpler ones on well-structured data.**

---

## 📈 Visualizations Included

- **Neural Network training loss vs epochs**
- **Confusion Matrix** for Neural Network
- **Confusion Matrix** for Logistic Regression
- **PCA projection** of the Iris dataset (2D visualization of class separability)

These plots help validate both **learning behavior** and **model performance**.

---

## 🗂️ Project Structure

```
nn-iris-classification/
│
├── data/
│   └── iris.csv
│
├── src/
│   ├── model.py          # Neural Network architecture
│   ├── train_nn.py       # NN training loop
│   ├── train_logr.py     # Logistic Regression baseline
│   ├── evaluate.py       # Evaluation utilities
│
├── visualizations/
│   └── visualize.py      # Loss curves, confusion matrices, PCA
│
├── main.py               # Pipeline orchestration
├── requirements.txt
└── README.md
```

---

## 🧪 Key ML Concepts Demonstrated

- Multiclass classification
- Neural Networks vs linear models
- Feature scaling importance
- Label encoding for deep learning
- Manual PyTorch training loop
- Fair baseline comparison
- Model evaluation beyond accuracy
- Visualization-driven validation

---

## 🧠 Key Takeaway
This project demonstrates that **Neural Networks are powerful function approximators**, but also reinforces a core machine learning lesson:

> *Model selection should be driven by data characteristics, not model complexity.*

---

## 🚀 Possible Extensions
- Decision boundary visualization (2-feature subspace)
- Deeper neural networks
- Hyperparameter tuning
- Classification report (precision, recall, F1)
- Comparison with SVM or Random Forest

---

## 🛠️ Tech Stack
- Python
- PyTorch
- scikit-learn
- NumPy
- pandas
- matplotlib

---

## ✅ Status
✔ Complete  
✔ Reproducible  
✔ Portfolio-ready  

---

*Built to understand, not just to score.*

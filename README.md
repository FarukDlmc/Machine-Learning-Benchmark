# 📊 Classification Algorithms Benchmark & ML Portfolio

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

A comprehensive Data Mining and Machine Learning repository demonstrating data preprocessing, feature engineering, model training, and performance benchmarking across multiple classification algorithms: **Support Vector Machines (SVM)**, **Random Forest**, and **Logistic Regression**.

---

## 📌 Benchmark Architecture & Portfolio Overview

| Project / Dataset Domain | Evaluated Algorithms | Dataset Details | Key Technical Concepts Covered |
| :--- | :--- | :--- | :--- |
| **1. Diabetes Prediction (Pima Indians)** | SVM (Linear vs RBF), Random Forest, Logistic Regression | Medical Dataset (Target column at end) | Feature Scaling, Kernel Tuning, Algorithm Benchmarking, Heatmap Confusion Matrices |
| **2. Diabetes Prediction (BRFSS 2015)** | SVM (Linear vs RBF), Random Forest, Logistic Regression | Large-scale Health Survey (Target column at start) | Large Dataset Handling, Random Sampling (25k), Classification Reports, Overfitting vs Underfitting Analysis |
| **3. Social Media Ads Targeting** | SVM (RBF Kernel), Random Forest, Logistic Regression | Demographic Data (Age & Salary) | Non-linear Classification, Feature Standardization, 2D Decision Boundary Plots (`meshgrid` & `contourf`) |

---

## 🔬 Detailed Project & Model Breakdown

### Project 1: Diabetes Prediction (Pima Indians)
* **Goal:** Predict whether a patient has diabetes based on clinical measurements (Pregnancies, Glucose, Blood Pressure, BMI, etc.).
* **Methodology & Benchmarking:** 
  * Applied `StandardScaler` to prevent high-variance feature dominance.
  * Evaluated and benchmarked **Linear SVM**, **RBF SVM**, **Logistic Regression**, and **Random Forest Classifier**.
  * **Output:** Generates comparative Seaborn heatmaps and performance classification reports to identify the optimal model.

### Project 2: Diabetes Prediction (BRFSS 2015)
* **Goal:** High-volume medical data classification using health survey indicators.
* **Methodology & Benchmarking:**
  * Processed a massive dataset by extracting a randomized, reproducible subset ($N = 25,000$).
  * Handled target feature isolation from the first column (`iloc[:, 0]`).
  * Benchmarked algorithms based on Precision, Recall, F1-Score, and overall Accuracy.
  * **Output:** Generates dynamic evaluation matrices highlighting how Random Forest handles complex tabular features vs. traditional Linear/RBF models.

### Project 3: Social Media Ads Targeting
* **Goal:** Classify user purchasing decisions based on demographic features (Age & Salary).
* **Methodology & Benchmarking:**
  * Extracted specific feature dimensions (`iloc[:, [2, 3]]`) for geometric decision boundary evaluation.
  * Trained SVM, Logistic Regression, and Ensemble Random Forest models to compare linear vs non-linear decision thresholds.
  * **Output:** Uses NumPy `meshgrid` and Matplotlib `contourf` to map and render complex 2D decision boundaries separating 'Purchased' (Green) and 'Not Purchased' (Red) zones.

---

## ⚙️ Model Comparison & Insights

* **Logistic Regression:** Serves as a fast, baseline model with high interpretability for linearly separable datasets.
* **Support Vector Machines (SVM):** Highly effective in high-dimensional space; RBF kernel captures non-linear boundary relationships better than linear models.
* **Random Forest Classifier:** Demonstrates superior accuracy on complex tabular data by leveraging ensemble decision trees, minimizing individual tree overfitting.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Data Mining:** `scikit-learn` (`SVC`, `RandomForestClassifier`, `LogisticRegression`, `StandardScaler`, `train_test_split`, `metrics`)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

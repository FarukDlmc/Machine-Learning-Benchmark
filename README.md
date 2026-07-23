# 🤖 Support Vector Machines (SVM) - Machine Learning Portfolio

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

A comprehensive collection of Machine Learning projects demonstrating the implementation of **Support Vector Machines (SVM)** for classification problems. This repository highlights practical experience in data preprocessing, feature scaling, model training, hyperparameter comparison, and advanced decision boundary visualization.

---

## 📌 Project Architecture & Portfolio

| Project Name | Domain | Dataset Details | Key Technical Concepts Covered |
| :--- | :--- | :--- | :--- |
| **1. Diabetes Prediction (Pima Indians)** | Healthcare | Small-scale dataset (target column at end) | Data Splitting, Feature Scaling, Linear vs RBF Kernel Comparison, Heatmap Confusion Matrices |
| **2. Diabetes Prediction (BRFSS 2015)** | Healthcare | Large-scale dataset (target column at start) | Large Dataset Handling, Random Sampling (25k subset), Classification Reports, SVM Optimization |
| **3. Social Media Ads Targeting** | E-Commerce | Customer Demographics (Age & Salary) | RBF Kernel Classification, 2D Decision Boundary Visualization (`meshgrid` & `contourf`) |

---

## 🔬 Detailed Project Breakdown

### Project 1: Diabetes Prediction (Model 1)
* **Goal:** Predict whether a patient has diabetes based on diagnostic measurements (Pregnancies, Glucose, Blood Pressure, BMI, etc.).
* **Methodology:** 
  * Implemented `StandardScaler` to prevent feature dominance.
  * Trained two separate SVM classifiers to compare **Linear Kernel** vs **RBF (Radial Basis Function) Kernel**.
  * **Output:** Generates side-by-side Seaborn heatmaps comparing the Confusion Matrices of both kernels.

### Project 2: Diabetes Prediction (BRFSS 2015)
* **Goal:** A robust evaluation of medical indicators using a massive health survey dataset.
* **Methodology:**
  * Handled a large dataset by extracting a randomized, reproducible sample of 25,000 records.
  * Separated the target variable located at the first column (`iloc[:, 0]`).
  * Compared precision, recall, and f1-scores dynamically using Scikit-Learn's `classification_report`.
  * **Output:** Generates a timestamped accuracy and confusion matrix comparison plot.

### Project 3: Social Media Ads Targeting
* **Goal:** Predict whether a user will purchase a product based on their Age and Estimated Salary.
* **Methodology:**
  * Sliced specific feature columns (`iloc[:, [2, 3]]`) to isolate relevant demographic data.
  * Applied an **RBF Kernel** to capture the non-linear relationship between age, income, and purchasing behavior.
  * **Output:** Utilized Matplotlib and NumPy `meshgrid` to plot the complex non-linear decision boundary, clearly separating the 'Purchased' (Green) and 'Not Purchased' (Red) zones.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (SVC, StandardScaler, train_test_split, metrics)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

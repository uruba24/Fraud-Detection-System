
# 🕵️‍♀️ Credit Card Fraud Detection System

This project builds a fraud detection system using machine learning techniques on the Credit Card Fraud Detection dataset. The dataset is sourced from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📌 Project Overview

The goal is to detect fraudulent credit card transactions using machine learning. Fraudulent transactions are rare and make the dataset highly imbalanced, which poses challenges during model training.

---

## 🚀 Steps Taken

### 1. **Dataset Loading**
- Used the original `creditcard.csv` dataset from Kaggle.
- Verified and explored the columns.

### 2. **Data Preprocessing**
- Separated features (`X`) and label (`y`).
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.

### 3. **Model Training**
- Used **Random Forest Classifier** for detecting fraud.
- Trained the model on 70% of the balanced dataset.

### 4. **Evaluation**
- Evaluated the model on the test set using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

### 5. **Transaction Prediction Interface**
- Tested the model with a single transaction to predict if it is fraudulent or legitimate.

---

## 🛠️ How to Run This Project in Google Colab

### 🔹 Step 1: Set up Kaggle API
1. Download your `kaggle.json` from your Kaggle account settings.
2. Upload it in Colab:

```python
from google.colab import files
files.upload()  # Upload kaggle.json
```

3. Configure:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

### 🔹 Step 2: Download the Dataset

```python
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip creditcardfraud.zip
```

---

### 🔹 Step 3: Install Dependencies

```python
!pip install imbalanced-learn
```

---

### 🔹 Step 4: Run the Full Script

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("creditcard.csv")

# Preprocess
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

### 🔹 Step 5: Test a Single Transaction

```python
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)[0]
print("Prediction:", "Fraudulent" if prediction == 1 else "Legitimate")
```

---

## 📊 Observations & Results

- **Imbalanced Dataset**: Original data had only ~0.17% fraud cases. Using SMOTE significantly improved model sensitivity to fraud.
- **Random Forest** achieved high precision and recall for fraud cases post-balancing.
- The model is able to correctly flag fraudulent transactions with minimal false positives.

---

## ✅ Future Improvements

- Try more advanced models like **XGBoost**, **LightGBM**.
- Deploy a **Streamlit web app** or **Flask API**.
- Add explainability with **SHAP** or **LIME**.

---

## 📁 Files in This Repository

- `fraud_detection_colab.ipynb` - Jupyter Notebook for Colab
- `README.md` - Project overview and instructions
- `creditcard.csv` - Dataset (must be downloaded via Kaggle API)

---

Made with 💡 and Python!

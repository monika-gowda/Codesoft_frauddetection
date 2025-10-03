## 📂 `frauddetection/README.md`
```markdown
# Credit Card Fraud Detection

This project detects **fraudulent credit card transactions** using machine learning.

## Dataset
- Source: [Kaggle – Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Place the files `fraudTrain.csv` and `fraudTest.csv` inside this folder.

## Model
- Features used: `unix_time`, `amt`
- Preprocessing: **StandardScaler**
- Balancing: **SMOTE** (Synthetic Minority Over-sampling Technique)
- Classifier: **Random Forest**

## How to Run
```bash
cd frauddetection
python fraud_detection.py

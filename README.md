# 📊 Sales Prediction using Advertising Data

[![Live Demo](https://img.shields.io/badge/LIVE-DEMO-red?style=for-the-badge&logo=streamlit)](https://sales-prediction-using-python.onrender.com)
[![Status](https://img.shields.io/badge/STATUS-ONLINE-green?style=for-the-badge&logo=python)](https://sales-prediction-using-python.onrender.com)

---

## 🚀 Project Overview
This project predicts product sales using advertising spend across TV, Radio, and Newspaper channels.  
It includes a complete machine learning workflow: data loading, cleaning, exploratory data analysis, model comparison, best-model persistence, and an interactive Streamlit dashboard.  

The project is designed to be **internship submission ready**, **GitHub portfolio ready**, and suitable for a **resume showcase**.

<img width="1534" height="1018" alt="Dashboard Screenshot" src="https://github.com/user-attachments/assets/76a30dc6-420f-4e47-aca6-d6dcc8501091" />

---

## 📂 Dataset Description
Dataset: [Advertising CSV on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)

**Columns:**
- `TV`: Advertising budget spent on TV  
- `Radio`: Advertising budget spent on Radio  
- `Newspaper`: Advertising budget spent on Newspaper  
- `Sales`: Product sales target variable  

---

## 📁 Project Structure
```text
Sales_Prediction_Project/
|
|-- data/
|   |-- Advertising.csv
|
|-- notebooks/
|   |-- EDA.ipynb
|
|-- models/
|   |-- best_model.pkl
|
|-- screenshots/
|   |-- sales_distribution.png
|   |-- tv_vs_sales.png
|   |-- radio_vs_sales.png
|   |-- newspaper_vs_sales.png
|   |-- correlation_heatmap.png
|   |-- pairplot.png
|   |-- boxplot_outliers.png
|   |-- feature_importance.png
|
|-- app.py
|-- train.py
|-- requirements.txt
|-- README.md

✨ Features

 Loads and inspects Advertising.csv

 Displays dataset shape, columns, missing values, duplicates, and summary statistics

Handles missing values with median imputation

Removes duplicate rows

Detects outliers using boxplots and IQR method

Creates professional EDA visualizations

Trains and compares multiple regression models

Evaluates models using R² Score, MAE, MSE, RMSE

Saves the best model with pickle

Provides an interactive Streamlit dashboard

🤖 Machine Learning Mode



⚙️ Installation Guide

bash
# 1. Clone project
git clone <repo-link>

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
python train.py

# 5. Run Streamlit dashboard
streamlit run app.py


📸 Screenshots
Recommended dashboard screenshots:

Dataset Overview

EDA Visualizations

Model Performance

Feature Importance

Sales Prediction Too

📊 Results
Best model: models/best_model.pkl

Model comparison: models/model_performance.csv

Feature importance: models/feature_importance.csv

Model summary: models/model_summary.json

🔮 Future Enhancements

Cross‑validation & hyperparameter tuning

Deploy to Streamlit Cloud

Automated tests for pipeline

REST API with FastAPI

Model explainability with SHAP

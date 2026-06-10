# Sales Prediction using Advertising Data

## Project Overview

This project predicts product sales using advertising spend across TV, Radio, and Newspaper channels. It includes a complete machine learning workflow: data loading, cleaning, exploratory data analysis, model comparison, best-model persistence, and an interactive Streamlit dashboard.

The project is designed to be internship submission ready, GitHub ready, and suitable for a resume portfolio.

## Dataset Description

Dataset: [Advertising CSV on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)

Columns:

- `TV`: Advertising budget spent on TV
- `Radio`: Advertising budget spent on Radio
- `Newspaper`: Advertising budget spent on Newspaper
- `Sales`: Product sales target variable

## Project Structure

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
```

## Features

- Loads and inspects `Advertising.csv`
- Displays dataset shape, columns, missing values, duplicates, and summary statistics
- Handles missing values with median imputation
- Removes duplicate rows
- Checks numeric consistency and removes invalid negative values
- Detects outliers using boxplots and the IQR method
- Creates professional EDA visualizations
- Trains and compares multiple regression models
- Evaluates models using R2 Score, MAE, MSE, and RMSE
- Automatically saves the best model with pickle
- Provides an interactive Streamlit sales prediction dashboard

## Machine Learning Models

The following models are trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected automatically based on the highest R2 Score and lowest RMSE.

## Installation Guide

1. Clone or download this project.

2. Open a terminal inside the project folder.

3. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Train the models:

```bash
python train.py
```

6. Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

## Screenshots

Screenshots and generated plots are saved in the `screenshots/` folder after running:

```bash
python train.py
```

Recommended dashboard screenshots to add before GitHub submission:

- Dataset Overview
- EDA Visualizations
- Model Performance
- Feature Importance
- Sales Prediction Tool

## Results

The training script saves:

- Best model: `models/best_model.pkl`
- Model comparison table: `models/model_performance.csv`
- Feature importance table: `models/feature_importance.csv`
- Model summary: `models/model_summary.json`

Typical result: ensemble models such as Random Forest Regressor or Gradient Boosting Regressor usually perform strongly on this dataset because they capture non-linear relationships between advertising channels and sales.

## How Prediction Works

The Streamlit dashboard accepts:

- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

It returns:

- Predicted Sales
- Model Used
- Confidence Information

## Future Enhancements

- Add cross-validation and hyperparameter tuning
- Deploy the Streamlit dashboard to Streamlit Community Cloud
- Add automated tests for the training pipeline
- Add experiment tracking with MLflow
- Build a REST API using FastAPI
- Add model explainability with SHAP

## Author

Created as a professional machine learning internship project.

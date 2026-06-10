"""
Training pipeline for Sales Prediction using Advertising Data.

This script performs dataset inspection, cleaning, EDA visualization export,
model training, evaluation, feature importance analysis, and model persistence.
Run from the project root:

    python train.py
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
RANDOM_STATE = 42


def ensure_directories() -> None:
    """Create output directories used by the project."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Advertising.csv and remove Kaggle's unnamed index column if present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place it in data/Advertising.csv."
        )

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed|^$", regex=True)]
    df.columns = [col.strip() for col in df.columns]
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print high-signal dataset information for the project report."""
    print("\n========== DATASET OVERVIEW ==========")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nSummary Statistics:")
    print(df.describe().round(3))

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, remove duplicates, and enforce numeric consistency."""
    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = df.copy()

    for col in required_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    numeric_medians = cleaned[required_columns].median(numeric_only=True)
    cleaned[required_columns] = cleaned[required_columns].fillna(numeric_medians)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Advertising spend and sales cannot be negative in this dataset context.
    for col in required_columns:
        cleaned = cleaned[cleaned[col] >= 0]

    return cleaned.reset_index(drop=True)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers with the IQR rule and return a summary table."""
    rows = []
    for col in FEATURES + [TARGET]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        rows.append(
            {
                "Feature": col,
                "Q1": round(q1, 3),
                "Q3": round(q3, 3),
                "IQR": round(iqr, 3),
                "Lower Bound": round(lower_bound, 3),
                "Upper Bound": round(upper_bound, 3),
                "Outlier Count": count,
            }
        )
    return pd.DataFrame(rows)


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure in a consistent format."""
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def create_eda_visualizations(df: pd.DataFrame) -> None:
    """Create professional EDA charts and save them to screenshots/."""
    sns.set_theme(style="whitegrid", palette="deep")

    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET], kde=True, color="#2563eb", bins=18)
    plt.title("Sales Distribution", fontsize=15, weight="bold")
    plt.xlabel("Sales")
    save_plot("sales_distribution.png")

    for feature, color in zip(FEATURES, ["#2563eb", "#059669", "#dc2626"]):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=df, x=feature, y=TARGET, scatter_kws={"alpha": 0.75}, line_kws={"color": "black"}, color=color)
        plt.title(f"{feature} Advertising Budget vs Sales", fontsize=15, weight="bold")
        save_plot(f"{feature.lower()}_vs_sales.png")

    plt.figure(figsize=(8, 5))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="RdBu_r", center=0, linewidths=0.5, fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=15, weight="bold")
    save_plot("correlation_heatmap.png")

    pairplot = sns.pairplot(df[FEATURES + [TARGET]], diag_kind="kde", corner=False)
    pairplot.fig.suptitle("Pairplot of Advertising Features and Sales", y=1.02, fontsize=15, weight="bold")
    pairplot.savefig(SCREENSHOTS_DIR / "pairplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[FEATURES + [TARGET]], orient="h", palette="Set2")
    plt.title("Boxplot Outlier Check", fontsize=15, weight="bold")
    save_plot("boxplot_outliers.png")


def build_models() -> Dict[str, object]:
    """Return the regression models used for comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def train_and_evaluate(df: pd.DataFrame) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """Train models, evaluate them, save the best model, and return results."""
    x = df[FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    results = []
    trained_models = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        results.append(
            {
                "Model": name,
                "R2 Score": r2_score(y_test, predictions),
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by=["R2 Score", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]

    feature_importance = calculate_feature_importance(best_model, best_model_name, x_test, y_test)
    save_model(best_model, best_model_name, results_df, feature_importance)
    save_results(results_df, feature_importance)
    create_feature_importance_plot(feature_importance, best_model_name)

    return best_model, results_df, feature_importance


def calculate_feature_importance(model: object, model_name: str, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Calculate feature importance using native attributes or permutation importance."""
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        method = "Model Feature Importance"
    else:
        permutation = permutation_importance(model, x_test, y_test, n_repeats=25, random_state=RANDOM_STATE)
        importance_values = permutation.importances_mean
        method = "Permutation Importance"

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": importance_values,
            "Method": method,
            "Model": model_name,
        }
    )
    return importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)


def create_feature_importance_plot(feature_importance: pd.DataFrame, model_name: str) -> None:
    """Save a feature importance bar chart."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title(f"Feature Importance - {model_name}", fontsize=15, weight="bold")
    save_plot("feature_importance.png")


def save_model(model: object, model_name: str, results_df: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    """Persist the best model and useful metadata with pickle."""
    model_package = {
        "model": model,
        "model_name": model_name,
        "features": FEATURES,
        "target": TARGET,
        "metrics": results_df.to_dict(orient="records"),
        "feature_importance": feature_importance.to_dict(orient="records"),
    }

    with open(MODELS_DIR / "best_model.pkl", "wb") as file:
        pickle.dump(model_package, file)


def save_results(results_df: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    """Save evaluation and feature-importance outputs for the dashboard."""
    results_df.to_csv(MODELS_DIR / "model_performance.csv", index=False)
    feature_importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    with open(MODELS_DIR / "model_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": results_df.loc[0, "Model"],
                "best_r2_score": float(results_df.loc[0, "R2 Score"]),
                "best_rmse": float(results_df.loc[0, "RMSE"]),
            },
            file,
            indent=4,
        )


def load_saved_model(path: Path = MODELS_DIR / "best_model.pkl") -> Dict[str, object]:
    """Load the persisted model package from disk."""
    with open(path, "rb") as file:
        return pickle.load(file)


def predict_sales(tv: float, radio: float, newspaper: float) -> float:
    """Load the saved model and return a single sales prediction."""
    package = load_saved_model()
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=package["features"])
    prediction = package["model"].predict(input_df)[0]
    return float(prediction)


def main() -> None:
    ensure_directories()
    df = load_dataset()
    inspect_dataset(df)

    cleaned_df = clean_dataset(df)
    print("\n========== CLEANED DATASET ==========")
    print(f"Shape after cleaning: {cleaned_df.shape}")
    print(f"Duplicate rows after cleaning: {cleaned_df.duplicated().sum()}")
    print(f"Missing values after cleaning:\n{cleaned_df.isnull().sum()}")

    outlier_summary = detect_outliers_iqr(cleaned_df)
    print("\n========== IQR OUTLIER SUMMARY ==========")
    print(outlier_summary)
    outlier_summary.to_csv(PROJECT_ROOT / "data" / "outlier_summary.csv", index=False)

    create_eda_visualizations(cleaned_df)
    best_model, results_df, feature_importance = train_and_evaluate(cleaned_df)

    print("\n========== MODEL PERFORMANCE ==========")
    print(results_df.round(4))
    print("\nBest Model:", results_df.loc[0, "Model"])
    print("\n========== FEATURE IMPORTANCE ==========")
    print(feature_importance.round(4))

    sample_prediction = predict_sales(tv=150, radio=25, newspaper=30)
    print(f"\nSample Prediction for TV=150, Radio=25, Newspaper=30: {sample_prediction:.2f}")


if __name__ == "__main__":
    main()

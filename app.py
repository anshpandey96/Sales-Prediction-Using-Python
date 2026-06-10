"""
Streamlit dashboard for Sales Prediction using Advertising Data.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PERFORMANCE_PATH = PROJECT_ROOT / "models" / "model_performance.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importance.csv"

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed|^$", regex=True)]
    df.columns = [col.strip() for col in df.columns]
    return df


@st.cache_resource
def load_model_package() -> dict:
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_model_performance() -> pd.DataFrame:
    if PERFORMANCE_PATH.exists():
        return pd.read_csv(PERFORMANCE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in FEATURES + [TARGET]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[FEATURES + [TARGET]] = cleaned[FEATURES + [TARGET]].fillna(cleaned[FEATURES + [TARGET]].median())
    cleaned = cleaned.drop_duplicates()
    for col in FEATURES + [TARGET]:
        cleaned = cleaned[cleaned[col] >= 0]
    return cleaned.reset_index(drop=True)


def prediction_confidence(prediction: float, sales_std: float, best_r2: float) -> tuple[str, float]:
    """Create friendly confidence information from model fit and data variability."""
    confidence = max(0.0, min(100.0, best_r2 * 100))
    interval = 1.96 * sales_std * (1 - best_r2)
    return f"{confidence:.1f}% model fit confidence", max(interval, 0.1)


def render_header() -> None:
    st.title("Sales Prediction using Advertising Data")
    st.caption("Machine learning dashboard for estimating sales from TV, Radio, and Newspaper advertising budgets.")


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    col4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(df.head(20), use_container_width=True)
    with right:
        st.dataframe(df.describe().round(2), use_container_width=True)


def render_eda(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x=TARGET, nbins=18, marginal="box", title="Sales Distribution")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    tab_tv, tab_radio, tab_newspaper, tab_pair = st.tabs(["TV", "Radio", "Newspaper", "Pairplot View"])
    for tab, feature in [(tab_tv, "TV"), (tab_radio, "Radio"), (tab_newspaper, "Newspaper")]:
        with tab:
            fig = px.scatter(
                df,
                x=feature,
                y=TARGET,
                trendline="ols",
                title=f"{feature} Advertising Budget vs Sales",
                labels={feature: f"{feature} Budget", TARGET: "Sales"},
            )
            fig.update_traces(marker=dict(size=9, opacity=0.78))
            fig.update_layout(height=470)
            st.plotly_chart(fig, use_container_width=True)

    with tab_pair:
        fig = px.scatter_matrix(df, dimensions=FEATURES + [TARGET], title="Pairplot of Advertising Data")
        fig.update_traces(diagonal_visible=False, marker=dict(size=6, opacity=0.65))
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Outlier Detection")
    box_fig = go.Figure()
    for col in FEATURES + [TARGET]:
        box_fig.add_trace(go.Box(y=df[col], name=col, boxmean=True))
    box_fig.update_layout(height=430, title="Boxplots for IQR Outlier Review")
    st.plotly_chart(box_fig, use_container_width=True)


def render_model_performance(performance_df: pd.DataFrame, model_package: dict) -> None:
    st.subheader("Model Performance")
    if performance_df.empty:
        st.warning("Model performance file not found. Run `python train.py` first.")
        return

    best_model_name = model_package.get("model_name", performance_df.iloc[0]["Model"])
    st.success(f"Best model selected automatically: {best_model_name}")

    st.dataframe(performance_df.style.format({"R2 Score": "{:.4f}", "MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}), use_container_width=True)

    metric_fig = px.bar(
        performance_df,
        x="Model",
        y=["R2 Score", "MAE", "RMSE"],
        barmode="group",
        title="Model Comparison",
    )
    metric_fig.update_layout(height=460)
    st.plotly_chart(metric_fig, use_container_width=True)


def render_feature_importance(feature_importance_df: pd.DataFrame) -> None:
    st.subheader("Feature Importance")
    if feature_importance_df.empty:
        st.warning("Feature importance file not found. Run `python train.py` first.")
        return

    fig = px.bar(
        feature_importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="viridis",
        title="Feature Importance Analysis",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(feature_importance_df, use_container_width=True)


def render_prediction_tool(df: pd.DataFrame, model_package: dict, performance_df: pd.DataFrame) -> None:
    st.subheader("Sales Prediction Tool")
    model = model_package["model"]
    model_name = model_package["model_name"]

    input_col, output_col = st.columns([1, 1])
    with input_col:
        tv = st.number_input("TV Advertising Budget", min_value=0.0, max_value=400.0, value=150.0, step=1.0)
        radio = st.number_input("Radio Advertising Budget", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
        predict_button = st.button("Predict Sales", type="primary", use_container_width=True)

    with output_col:
        if predict_button:
            input_df = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)
            prediction = float(model.predict(input_df)[0])

            best_r2 = float(performance_df.iloc[0]["R2 Score"]) if not performance_df.empty else 0.0
            confidence_text, interval = prediction_confidence(prediction, df[TARGET].std(), best_r2)

            st.metric("Predicted Sales", f"{prediction:.2f}")
            st.write(f"Model Used: **{model_name}**")
            st.info(f"{confidence_text}. Approximate prediction band: {prediction - interval:.2f} to {prediction + interval:.2f}.")
        else:
            st.info("Enter advertising budgets and click Predict Sales.")


def main() -> None:
    render_header()

    if not DATA_PATH.exists():
        st.error("Dataset not found. Place Advertising.csv inside the data folder.")
        st.stop()

    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `python train.py` before launching the dashboard.")
        st.stop()

    df = clean_data(load_data())
    model_package = load_model_package()
    performance_df = load_model_performance()
    feature_importance_df = load_feature_importance()

    section = st.sidebar.radio(
        "Navigation",
        ["Dataset Overview", "EDA Visualizations", "Model Performance", "Feature Importance", "Sales Prediction Tool"],
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Best Model", model_package.get("model_name", "Not trained"))
    if not performance_df.empty:
        st.sidebar.metric("Best R2 Score", f"{performance_df.iloc[0]['R2 Score']:.4f}")

    if section == "Dataset Overview":
        render_dataset_overview(df)
    elif section == "EDA Visualizations":
        render_eda(df)
    elif section == "Model Performance":
        render_model_performance(performance_df, model_package)
    elif section == "Feature Importance":
        render_feature_importance(feature_importance_df)
    else:
        render_prediction_tool(df, model_package, performance_df)


if __name__ == "__main__":
    main()

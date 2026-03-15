"""Model utilities for training and evaluating a linear regression model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    feature_col: str = "Hours",
    target_col: str = "Score",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split dataset into training and testing sets."""

    X = df[[feature_col]].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a simple linear regression model."""

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Make predictions and compute evaluation metrics."""

    y_pred = model.predict(X_test)

    metrics = {
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred),
    }

    return metrics, y_pred


def plot_regression(
    df: pd.DataFrame,
    model: LinearRegression,
    feature_col: str = "Hours",
    target_col: str = "Score",
    figsize=(8, 6),
):
    """Plot data points and fitted regression line."""

    plt.figure(figsize=figsize)
    plt.scatter(df[feature_col], df[target_col], color="blue", label="Data points")

    # Create a line for predictions
    hours_range = np.linspace(df[feature_col].min(), df[feature_col].max(), 100).reshape(-1, 1)
    score_pred = model.predict(hours_range)

    plt.plot(hours_range, score_pred, color="red", linewidth=2, label="Regression line")
    plt.title("Study Hours vs Exam Score")
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model(model: LinearRegression, output_path: Path) -> None:
    """Persist the trained model to disk."""
    from joblib import dump

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)


def load_model(model_path: Path) -> LinearRegression:
    """Load a persisted model from disk."""
    from joblib import load

    return load(model_path)

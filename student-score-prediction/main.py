"""Student Score Prediction using Linear Regression.

This script walks through data loading, exploration, visualization, training a linear regression model,
and evaluating its performance. It also allows users to provide study hours and receive a score prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import explore_data, load_data
from src.model import (
    evaluate_model,
    plot_regression,
    save_model,
    split_data,
    train_linear_regression,
)


def main() -> None:
    project_root = Path(__file__).parent

    data_path = project_root / "data" / "student_scores.csv"
    model_output_path = project_root / "models" / "linear_regression_model.joblib"

    # --- Data Handling ---
    df = load_data(data_path)
    print("\nLoaded dataset:")
    print(df.head())

    explore_data(df)

    # --- Visualization ---
    print("\nPlotting the relationship between study hours and exam score...")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Hours"], df["Score"], color="darkblue")
    plt.title("Study Hours vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Data Preprocessing ---
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Model Training ---
    model = train_linear_regression(X_train, y_train)
    save_model(model, model_output_path)

    # --- Model Evaluation ---
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    print("\nModel Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # --- Visualization: Regression Line ---
    print("\nPlotting regression line with data points...")
    plot_regression(df, model)

    # --- Prediction Feature ---
    while True:
        user_input = input(
            "\nEnter study hours to predict exam score (or type 'q' to quit): "
        ).strip()

        if user_input.lower() in {"q", "quit", "exit"}:
            print("Exiting prediction loop. Goodbye!")
            break

        try:
            hours = float(user_input)
            if hours < 0:
                raise ValueError("Study hours cannot be negative")

            predicted_score = model.predict([[hours]])[0]
            print(
                f"If a student studies {hours:.2f} hours, the predicted exam score is {predicted_score:.2f}."
            )
        except ValueError as exc:
            print(f"Invalid input: {exc}. Please enter a positive number.")


if __name__ == "__main__":
    main()

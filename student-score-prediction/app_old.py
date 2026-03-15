"""
Student Score Prediction Web App using Streamlit.

This Streamlit application provides an interactive web interface for predicting
student exam scores based on study hours using a Linear Regression model.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from src.data_loader import load_data
from src.model import load_model, plot_regression, split_data, train_linear_regression


def main():
    """Main function to run the Streamlit web application."""

    # Configure page
    st.set_page_config(
        page_title="Student Score Prediction",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title Section
    st.title("🎓 Student Score Prediction using Machine Learning")
    st.markdown("---")

    # Description Section
    st.subheader("📖 About This Application")
    st.write("""
    This machine learning application predicts student exam scores based on study hours
    using a **Linear Regression** model. The model has been trained on historical data
    showing the relationship between study time and exam performance.

    **How it works:**
    - Enter the number of study hours using the slider below
    - Click the "Predict Score" button to get an instant prediction
    - View the data visualization to understand the prediction trend
    """)

    st.markdown("---")

    # Load data and model
    @st.cache_data
    def load_data_and_train_model():
        """Load data and train the model (cached for performance)."""
        project_root = Path(__file__).parent
        data_path = project_root / "data" / "student_scores.csv"

        # Load and prepare data
        df = load_data(data_path)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # Train model
        model = train_linear_regression(X_train, y_train)

        return df, model

    # Load data and model
    try:
        df, model = load_data_and_train_model()

        # User Input Section
        st.subheader("⏰ Study Hours Input")
        st.write("Select the number of study hours (0-15 hours):")

        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])

        with col1:
            hours = st.slider(
                "Study Hours",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Drag the slider to select study hours"
            )
            st.write(f"**Selected: {hours} hours**")

        with col2:
            st.markdown("### Quick Stats")
            st.metric("Min Hours", f"{df['Hours'].min():.1f}")
            st.metric("Max Hours", f"{df['Hours'].max():.1f}")
            st.metric("Avg Hours", f"{df['Hours'].mean():.1f}")

        # Prediction Button and Output
        st.markdown("---")
        st.subheader("🔮 Score Prediction")

        if st.button("Predict Score", type="primary", use_container_width=True):
            # Make prediction
            predicted_score = model.predict([[hours]])[0]

            # Display prediction with formatting
            st.success("✅ Prediction Complete!")

            # Create columns for prediction display
            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                st.metric(
                    label="Predicted Exam Score",
                    value=f"{predicted_score:.1f}",
                    delta=f"Based on {hours} hours of study"
                )

            with pred_col2:
                # Add some context
                if predicted_score >= 90:
                    st.info("🌟 Excellent! Keep up the great work!")
                elif predicted_score >= 80:
                    st.info("👍 Good job! You're on the right track!")
                elif predicted_score >= 70:
                    st.info("📚 Solid performance! A bit more study could help!")
                else:
                    st.warning("💪 Consider increasing study time for better results!")

        # Data Visualization Section
        st.markdown("---")
        st.subheader("📊 Data Visualization")

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot of actual data
        ax.scatter(df['Hours'], df['Score'],
                  color='darkblue', alpha=0.7, s=50,
                  label='Actual Data Points')

        # Regression line
        import numpy as np
        hours_range = np.linspace(df['Hours'].min(), df['Hours'].max(), 100).reshape(-1, 1)
        score_pred = model.predict(hours_range)

        ax.plot(hours_range, score_pred,
               color='red', linewidth=3, alpha=0.8,
               label='Regression Line (Prediction)')

        # Add prediction point if prediction was made
        if 'predicted_score' in locals():
            ax.scatter([hours], [predicted_score],
                      color='green', s=100, marker='*',
                      label=f'Your Prediction: {predicted_score:.1f}')

        # Customize plot
        ax.set_title('Study Hours vs Exam Score', fontsize=16, fontweight='bold')
        ax.set_xlabel('Study Hours', fontsize=12)
        ax.set_ylabel('Exam Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 110)

        # Display the plot
        st.pyplot(fig)

        # Additional Information
        st.markdown("---")
        st.subheader("📈 Model Insights")

        # Display some statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Data Points", len(df))

        with col2:
            correlation = df['Hours'].corr(df['Score'])
            st.metric("Correlation", f"{correlation:.3f}")

        with col3:
            st.metric("Model Type", "Linear Regression")

        # Show sample data
        with st.expander("📋 View Sample Data"):
            st.dataframe(df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please ensure all required files are present and dependencies are installed.")


if __name__ == "__main__":
    main()
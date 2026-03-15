"""
Student Score Prediction - Modern ML Dashboard

A professional machine learning web application for predicting student exam scores
based on study hours using Linear Regression. Features a modern dashboard-style UI
inspired by Figma design systems.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_loader import load_data
from src.model import load_model, split_data, train_linear_regression, evaluate_model


def main():
    """Main function to run the modern Streamlit dashboard application."""

    # Configure page settings
    st.set_page_config(
        page_title="Student Score Prediction",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for modern styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4CAF50 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #6C63FF;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(108, 99, 255, 0.3);
    }
    .section-header {
        color: #6C63FF;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #6C63FF;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 3px solid #4CAF50;
    }
    .footer {
        background: #f8f9fa;
        padding: 1rem;
        text-align: center;
        border-radius: 8px;
        margin-top: 2rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load data and model
    @st.cache_data
    def load_data_and_model():
        """Load data and trained model (cached for performance)."""
        try:
            # Try to load pre-trained model first
            project_root = Path(__file__).parent
            model_path = project_root / "models" / "linear_regression_model.joblib"
            data_path = project_root / "data" / "student_scores.csv"

            df = load_data(data_path)
            model = load_model(model_path)

            # Get model metrics
            X_train, X_test, y_train, y_test = split_data(df)
            metrics, _ = evaluate_model(model, X_test, y_test)

            return df, model, metrics
        except Exception as e:
            st.error(f"Error loading model: {e}")
            # Fallback to training a new model
            df = load_data(data_path)
            X_train, X_test, y_train, y_test = split_data(df)
            model = train_linear_regression(X_train, y_train)
            metrics, _ = evaluate_model(model, X_test, y_test)
            return df, model, metrics

    # Load data and model
    df, model, metrics = load_data_and_model()

    # ===== TOP NAVIGATION HEADER =====
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">Student Score Prediction</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Machine Learning Regression Model</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== TWO COLUMN LAYOUT =====
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT COLUMN: PREDICTION PANEL
    with col1:
        st.markdown('<div class="section-header">📊 Prediction Panel</div>', unsafe_allow_html=True)

        # Prediction input card
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### Enter Study Hours")

        # Study hours slider
        hours = st.slider(
            "",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Select the number of study hours"
        )

        st.markdown(f"**Selected: {hours} hours**")

        # Calculate prediction for current slider value
        predicted_score = model.predict([[hours]])[0]

        # Predict button
        if st.button("🔮 Predict Score", type="primary", use_container_width=True):
            # Display prediction result
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Predicted Score")
            st.markdown(f"**{predicted_score:.1f} / 100**")
            st.markdown('</div>', unsafe_allow_html=True)

            # Contextual message
            if predicted_score >= 90:
                st.success("🌟 Excellent performance expected!")
            elif predicted_score >= 80:
                st.info("👍 Good score predicted!")
            elif predicted_score >= 70:
                st.warning("📚 Consider additional study time")
            else:
                st.error("💪 More study hours recommended")

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT COLUMN: VISUALIZATION PANEL
    with col2:
        st.markdown('<div class="section-header">📈 Data Visualization</div>', unsafe_allow_html=True)

        # Create interactive Plotly chart
        fig = go.Figure()

        # Scatter plot of actual data
        fig.add_trace(go.Scatter(
            x=df['Hours'],
            y=df['Score'],
            mode='markers',
            name='Actual Data',
            marker=dict(
                color='#6C63FF',
                size=8,
                opacity=0.7
            )
        ))

        # Regression line
        hours_range = np.linspace(df['Hours'].min(), df['Hours'].max(), 100)
        score_pred = model.predict(hours_range.reshape(-1, 1))

        fig.add_trace(go.Scatter(
            x=hours_range,
            y=score_pred,
            mode='lines',
            name='Regression Line',
            line=dict(
                color='#4CAF50',
                width=3
            )
        ))

        # User prediction point
        fig.add_trace(go.Scatter(
            x=[hours],
            y=[predicted_score],
            mode='markers',
            name='User Prediction',
            marker=dict(
                color='red',
                size=12,
                symbol='circle',
                line=dict(width=2, color='darkred')
            )
        ))

        # Update layout
        fig.update_layout(
            title="Study Hours vs Exam Score",
            xaxis_title="Study Hours",
            yaxis_title="Exam Score",
            template="plotly_white",
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    # ===== MODEL EXPLANATION SECTION =====
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 How the Model Works</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Algorithm")
        st.markdown("**Linear Regression**")
        st.markdown("Simple yet powerful ML algorithm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Training Data")
        st.markdown(f"**{len(df)} samples**")
        st.markdown("Historical student performance data")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Model Performance")
        st.markdown(f"**MAE:** {metrics['Mean Absolute Error']:.2f}")
        st.markdown(f"**R²:** {metrics['R2 Score']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== DATASET PREVIEW SECTION =====
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)

    st.markdown("First 10 rows of the training dataset:")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        column_config={
            "Hours": st.column_config.NumberColumn(
                "Study Hours",
                format="%.1f"
            ),
            "Score": st.column_config.NumberColumn(
                "Exam Score",
                format="%.1f"
            )
        }
    )

    # ===== FOOTER SECTION =====
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="margin: 0;">Built with ❤️ using Python, Machine Learning & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
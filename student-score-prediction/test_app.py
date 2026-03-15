"""
Test script to validate the core ML functionality before running the Streamlit app.
"""

from pathlib import Path
import pandas as pd
from src.data_loader import load_data
from src.model import split_data, train_linear_regression


def test_core_functionality():
    """Test the core ML pipeline without Streamlit."""
    print("🔍 Testing core ML functionality...")

    # Load data
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "student_scores.csv"

    try:
        df = load_data(data_path)
        print(f"✅ Data loaded successfully: {len(df)} rows")

        # Split data
        X_train, X_test, y_train, y_test = split_data(df)
        print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")

        # Train model
        model = train_linear_regression(X_train, y_train)
        print("✅ Model trained successfully")

        # Test prediction
        test_hours = [5.0, 7.5, 10.0]
        for hours in test_hours:
            prediction = model.predict([[hours]])[0]
            print(f"  {hours} hours → {prediction:.1f} predicted score")
        print("✅ All tests passed! The ML pipeline is working correctly.")
        print("\n🚀 You can now run the Streamlit app with: streamlit run app.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_core_functionality()
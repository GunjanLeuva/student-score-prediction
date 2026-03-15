"""Data loading and exploration utilities for the Student Score Prediction project."""

from pathlib import Path

import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file. If file is missing, create a sample dataset."""

    if not data_path.exists():
        _create_sample_data(data_path)

    df = pd.read_csv(data_path)
    return df


def _create_sample_data(data_path: Path) -> None:
    """Generate a simple sample dataset and save it to the given path."""
    sample_data = {
        "Hours": [
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
        ],
        "Score": [
            40,
            50,
            45,
            50,
            60,
            55,
            65,
            60,
            70,
            65,
            70,
            75,
            75,
            80,
            85,
            90,
            88,
            95,
            93,
            96,
            98,
            100,
            99,
            100,
            100,
        ],
    }

    df = pd.DataFrame(sample_data)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)


def explore_data(df: pd.DataFrame) -> None:
    """Print basic dataset information and statistics."""
    print("\nDataset info:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())

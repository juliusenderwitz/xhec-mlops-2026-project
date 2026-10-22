"""Preprocessing functions to load the abalone dataset and prepare it for training."""

from pathlib import Path

import pandas as pd
from prefect import task
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Rings"
NUMERICAL_COLUMNS = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]
CATEGORICAL_COLUMNS = ["Sex"]


@task(name="Load data")
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the abalone dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        The raw dataset as a DataFrame.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    return pd.read_csv(path)


@task(name="Drop invalid rows")
def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the physically impossible rows found in the EDA.

    A row is invalid if ``Height`` is 0, ``Height`` is above 0.4, or the sum of the
    shucked, viscera and shell weights is above 1.5 times the whole weight.

    Args:
        df: Raw abalone dataset.

    Returns:
        A copy of ``df`` without the invalid rows, with a fresh index.
    """
    parts_sum = df["Shucked weight"] + df["Viscera weight"] + df["Shell weight"]
    invalid = (
        (df["Height"] == 0)
        | (df["Height"] > 0.4)
        | (parts_sum > 1.5 * df["Whole weight"])
    )
    return df[~invalid].reset_index(drop=True)


@task(name="Extract X and y")
def extract_x_y(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target.

    Only the columns in ``NUMERICAL_COLUMNS`` and ``CATEGORICAL_COLUMNS`` are kept
    as features.

    Args:
        df: Dataset containing the feature columns and ``target_column``.
        target_column: Name of the column to predict.

    Returns:
        A tuple ``(X, y)`` with the feature DataFrame and the target Series.

    Raises:
        KeyError: If a feature column or ``target_column`` is not in ``df``.
    """
    X = df[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    y = df[target_column]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the transformer that scales and encodes the features.

    Numerical columns are standardized and ``Sex`` is one-hot encoded. Unknown
    categories are ignored at prediction time instead of raising an error.

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

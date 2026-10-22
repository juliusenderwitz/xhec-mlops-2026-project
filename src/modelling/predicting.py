"""Prediction functions to run a trained abalone model on new data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from utils import load_object


def predict(model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
    """Predict the target for the given features.

    Args:
        model: A fitted regression model.
        X: Raw features, with the same columns as the training data.

    Returns:
        The predicted values, one per row of ``X``.
    """
    return model.predict(X)


def predict_from_pickle(model_path: str | Path, X: pd.DataFrame) -> np.ndarray:
    """Load a pickled model and predict the target for the given features.

    Args:
        model_path: Path to the pickled model.
        X: Raw features, with the same columns as the training data.

    Returns:
        The predicted values, one per row of ``X``.

    Raises:
        FileNotFoundError: If no file exists at ``model_path``.
    """
    model = load_object(model_path)
    return predict(model, X)

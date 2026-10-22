"""Training functions to fit and evaluate the abalone age prediction model."""

import pandas as pd
from preprocessing import build_preprocessor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Chain the preprocessing and the regressor chosen in the modelling notebook.

    Returns:
        An unfitted pipeline with a ``preprocessor`` step and a ``model`` step.
    """
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("model", LinearRegression())]
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the pipeline (preprocessing + regressor) on the training data.

    Args:
        X: Raw training features.
        y: Training target.

    Returns:
        The fitted pipeline.
    """
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute the regression metrics of a fitted model.

    Args:
        model: A fitted model or pipeline.
        X: Raw evaluation features.
        y: True target values.

    Returns:
        A dict with the RMSE and MAE (in rings) and the R² score.
    """
    y_pred = model.predict(X)
    return {
        "rmse": float(root_mean_squared_error(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

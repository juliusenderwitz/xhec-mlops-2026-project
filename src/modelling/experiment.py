"""Run model experiments and log parameters, metrics, and models to MLflow."""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from preprocessing import build_preprocessor, drop_invalid_rows, extract_x_y, load_data
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_LOCATION = (ROOT_DIR / "mlruns").as_uri()
EXPERIMENT_NAME = "abalone-rings"

TEST_SIZE = 0.2
N_FOLDS = 5
RANDOM_STATE = 42

MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "linear_regression": LinearRegression(),
}


def evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Evaluate the predictions of a model run.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        A dict with the RMSE, MAE, and R² score.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Cross-validates a pipeline and returns the mean metrics across folds.

    Args:
        pipeline: A scikit-learn pipeline with a preprocessor and a model.
        X: Raw features, with the same columns as the training data.
        y: True target values.

    Returns:
        A dict with the mean RMSE, MAE, and R² score across folds.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
    )
    return {
        "rmse": float(-scores["test_rmse"].mean()),
        "mae": float(-scores["test_mae"].mean()),
        "r2": float(scores["test_r2"].mean()),
    }


def run_experiments(data_path: Path) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    raw_df = load_data(data_path)
    clean_df = drop_invalid_rows(raw_df)
    n_dropped = len(raw_df) - len(clean_df)

    X, y = extract_x_y(clean_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                [("preprocessor", build_preprocessor()), ("model", model)]
            )

            mlflow.log_params(
                {
                    "model_name": name,
                    "test_size": TEST_SIZE,
                    "n_folds": N_FOLDS,
                    "random_state": RANDOM_STATE,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_dropped_rows": n_dropped,
                }
            )
            mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})

            cv_metrics = cross_validate_pipeline(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)
            test_metrics = evaluate(y_test, pipeline.predict(X_test))

            metrics = {f"cv_{k}": v for k, v in cv_metrics.items()}
            metrics |= {f"test_{k}": v for k, v in test_metrics.items()}
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=infer_signature(X_train, pipeline.predict(X_train)),
                input_example=X_train.head(5),
            )
            print(
                f"Logged {name} - CV RMSE: {cv_metrics['rmse']:.3f} | Test RMSE: {test_metrics['rmse']:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLflow experiments on the abalone dataset."
    )
    parser.add_argument(
        "data_path", type=Path, nargs="?", default=ROOT_DIR / "data" / "abalone.csv"
    )
    args = parser.parse_args()
    run_experiments(args.data_path)

"""Paths and metadata for the abalone prediction API."""

from pathlib import Path

# Training saves one fitted pipeline containing both preprocessing and regression.
PATH_TO_MODEL = Path(__file__).resolve().parent / "local_objects" / "model.pkl"

APP_TITLE = "Abalone Age Prediction API"
APP_DESCRIPTION = (
    "Predict an abalone's number of shell rings from its sex and physical measurements."
)
APP_VERSION = "0.0.1"

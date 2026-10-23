"""HTTP API for abalone ring count predictions."""

from fastapi import FastAPI

from src.web_service.app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    PATH_TO_MODEL,
)
from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import InputData, PredictionOut
from src.web_service.utils import load_object

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: InputData) -> PredictionOut:
    """Predict the number of rings from one abalone's measurements."""
    model = load_object(PATH_TO_MODEL)
    prediction = run_inference([payload], model)
    return PredictionOut(age=float(prediction[0]))

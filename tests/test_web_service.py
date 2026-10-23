"""Integration checks for the prediction API and artifact cache."""

import pickle

import pytest
from pydantic import ValidationError

from src.web_service.lib.models import InputData
from src.web_service.main import app, home, predict
from src.web_service.utils import load_object

PAYLOAD = {
    "Sex": "M",
    "Length": 0.455,
    "Diameter": 0.365,
    "Height": 0.095,
    "Whole weight": 0.514,
    "Shucked weight": 0.2245,
    "Viscera weight": 0.101,
    "Shell weight": 0.15,
}


def test_prediction_api_uses_saved_pipeline():
    assert home() == {"health_check": "App up and running!"}
    assert "/predict" in app.openapi()["paths"]
    result = predict(InputData.model_validate(PAYLOAD))

    assert isinstance(result.age, float)


def test_prediction_api_rejects_invalid_input():
    with pytest.raises(ValidationError):
        InputData.model_validate({**PAYLOAD, "Height": 0})


def test_artifact_loader_caches_by_path(tmp_path):
    path = tmp_path / "artifact.pkl"
    path.write_bytes(pickle.dumps({"version": 1}))
    load_object.cache_clear()

    first = load_object(path)
    path.write_bytes(pickle.dumps({"version": 2}))

    assert load_object(path) is first
    assert first == {"version": 1}
    load_object.cache_clear()

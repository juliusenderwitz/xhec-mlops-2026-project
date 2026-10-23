"""Convert validated API input into predictions from the fitted pipeline."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.web_service.lib.models import InputData


def run_inference(input_data: list[InputData], model: Pipeline) -> np.ndarray:
    """Predict ring counts for a batch of abalone measurements."""
    features = pd.DataFrame([item.model_dump(by_alias=True) for item in input_data])
    return model.predict(features)

import pandas as pd
import pytest
from predicting import predict, predict_from_pickle
from sklearn.linear_model import LinearRegression
from utils import pickle_object

# Simple data where y = 2 * x, so the expected predictions are easy to check
X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
y = pd.Series([2.0, 4.0, 6.0])


@pytest.fixture
def model():
    return LinearRegression().fit(X, y)


class TestPredict:
    def test_returns_one_prediction_per_row(self, model):
        assert len(predict(model, X)) == len(X)

    def test_returns_expected_values(self, model):
        assert predict(model, X) == pytest.approx([2.0, 4.0, 6.0])


class TestPredictFromPickle:
    def test_same_predictions_as_original_model(self, model, tmp_path):
        path = tmp_path / "model.pkl"
        pickle_object(model, path)
        assert predict_from_pickle(path, X) == pytest.approx(predict(model, X))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            predict_from_pickle(tmp_path / "missing.pkl", X)

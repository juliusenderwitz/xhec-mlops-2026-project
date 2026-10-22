import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from training import build_pipeline, evaluate_model, train_model


@pytest.fixture
def X():
    # Four abalone rows with the feature columns used by the preprocessor
    return pd.DataFrame(
        {
            "Length": [0.5, 0.6, 0.4, 0.55],
            "Diameter": [0.4, 0.5, 0.3, 0.45],
            "Height": [0.1, 0.15, 0.08, 0.12],
            "Whole weight": [0.5, 0.8, 0.3, 0.6],
            "Shucked weight": [0.2, 0.3, 0.1, 0.25],
            "Viscera weight": [0.1, 0.2, 0.05, 0.15],
            "Shell weight": [0.15, 0.25, 0.1, 0.2],
            "Sex": ["M", "F", "I", "M"],
        }
    )


@pytest.fixture
def y():
    return pd.Series([10, 12, 7, 11])


class TestBuildPipeline:
    def test_returns_pipeline(self):
        assert isinstance(build_pipeline(), Pipeline)

    def test_has_preprocessor_and_model_steps(self):
        assert list(build_pipeline().named_steps) == ["preprocessor", "model"]

    def test_model_is_linear_regression(self):
        assert isinstance(build_pipeline().named_steps["model"], LinearRegression)


class TestTrainModel:
    def test_returns_fitted_pipeline(self, X, y):
        model = train_model(X, y)
        assert isinstance(model, Pipeline)
        check_is_fitted(model)

    def test_can_predict_on_raw_features(self, X, y):
        model = train_model(X, y)
        assert len(model.predict(X)) == len(X)


class TestEvaluateModel:
    def test_returns_expected_keys(self, X, y):
        model = train_model(X, y)
        assert set(evaluate_model(model, X, y)) == {"rmse", "mae", "r2"}

    def test_values_are_floats(self, X, y):
        model = train_model(X, y)
        assert all(isinstance(v, float) for v in evaluate_model(model, X, y).values())

    def test_perfect_predictions(self):
        # Simple data where y = 2 * x, so a linear regression fits it exactly
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pd.Series([2.0, 4.0, 6.0])
        model = LinearRegression().fit(X, y)
        metrics = evaluate_model(model, X, y)
        assert metrics["rmse"] == pytest.approx(0)
        assert metrics["mae"] == pytest.approx(0)
        assert metrics["r2"] == pytest.approx(1)

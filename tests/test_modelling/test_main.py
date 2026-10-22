import main
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from utils import load_object


@pytest.fixture
def trainset_path(tmp_path):
    # Ten valid abalone rows saved as a CSV, enough for the train/test split
    df = pd.DataFrame(
        {
            "Sex": ["M", "F", "I"] * 3 + ["M"],
            "Length": [0.4 + 0.02 * i for i in range(10)],
            "Diameter": [0.3 + 0.02 * i for i in range(10)],
            "Height": [0.08 + 0.01 * i for i in range(10)],
            "Whole weight": [0.3 + 0.05 * i for i in range(10)],
            "Shucked weight": [0.1 + 0.02 * i for i in range(10)],
            "Viscera weight": [0.05 + 0.01 * i for i in range(10)],
            "Shell weight": [0.1 + 0.02 * i for i in range(10)],
            "Rings": [7 + i for i in range(10)],
        }
    )
    path = tmp_path / "abalone.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    # Save the model in a temporary folder instead of overwriting the real one
    path = tmp_path / "model.pkl"
    monkeypatch.setattr(main, "MODEL_PATH", path)
    return path


class TestMain:
    def test_saves_model(self, trainset_path, model_path):
        main.main(trainset_path)
        assert model_path.exists()

    def test_saved_model_is_pipeline(self, trainset_path, model_path):
        main.main(trainset_path)
        assert isinstance(load_object(model_path), Pipeline)

    def test_prints_metrics(self, trainset_path, model_path, capsys):
        main.main(trainset_path)
        output = capsys.readouterr().out
        assert "Test RMSE" in output
        assert "Test MAE" in output
        assert "Test R2" in output

    def test_missing_file_raises(self, tmp_path, model_path):
        with pytest.raises(FileNotFoundError):
            main.main(tmp_path / "missing.csv")

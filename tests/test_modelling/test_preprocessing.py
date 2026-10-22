import pandas as pd
import pytest
from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    build_preprocessor,
    drop_invalid_rows,
    extract_x_y,
    load_data,
)


@pytest.fixture
def df():
    # Three valid abalone rows with every column of the dataset
    return pd.DataFrame(
        {
            "Sex": ["M", "F", "I"],
            "Length": [0.5, 0.6, 0.4],
            "Diameter": [0.4, 0.5, 0.3],
            "Height": [0.1, 0.15, 0.08],
            "Whole weight": [0.5, 0.8, 0.3],
            "Shucked weight": [0.2, 0.3, 0.1],
            "Viscera weight": [0.1, 0.2, 0.05],
            "Shell weight": [0.15, 0.25, 0.1],
            "Rings": [10, 12, 7],
        }
    )


class TestLoadData:
    def test_loads_csv(self, df, tmp_path):
        path = tmp_path / "abalone.csv"
        df.to_csv(path, index=False)
        pd.testing.assert_frame_equal(load_data(path), df)

    def test_accepts_string_path(self, df, tmp_path):
        path = tmp_path / "abalone.csv"
        df.to_csv(path, index=False)
        assert len(load_data(str(path))) == len(df)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "missing.csv")


class TestDropInvalidRows:
    def test_keeps_valid_rows(self, df):
        pd.testing.assert_frame_equal(drop_invalid_rows(df), df)

    def test_drops_zero_height(self, df):
        df.loc[0, "Height"] = 0
        assert len(drop_invalid_rows(df)) == 2

    def test_drops_height_above_threshold(self, df):
        df.loc[0, "Height"] = 0.5
        assert len(drop_invalid_rows(df)) == 2

    def test_drops_parts_heavier_than_whole(self, df):
        # Parts sum to 0.45, which is above 1.5 * 0.1
        df.loc[0, "Whole weight"] = 0.1
        assert len(drop_invalid_rows(df)) == 2

    def test_resets_index(self, df):
        df.loc[0, "Height"] = 0
        assert list(drop_invalid_rows(df).index) == [0, 1]

    def test_does_not_modify_input(self, df):
        df.loc[0, "Height"] = 0
        drop_invalid_rows(df)
        assert len(df) == 3


class TestExtractXY:
    def test_returns_feature_columns(self, df):
        X, _ = extract_x_y(df)
        assert list(X.columns) == NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

    def test_returns_target(self, df):
        _, y = extract_x_y(df)
        pd.testing.assert_series_equal(y, df["Rings"])

    def test_custom_target_column(self, df):
        df = df.rename(columns={"Rings": "Age"})
        _, y = extract_x_y(df, target_column="Age")
        assert y.name == "Age"

    def test_missing_target_raises(self, df):
        with pytest.raises(KeyError):
            extract_x_y(df.drop(columns="Rings"))

    def test_missing_feature_raises(self, df):
        with pytest.raises(KeyError):
            extract_x_y(df.drop(columns="Sex"))


class TestBuildPreprocessor:
    def test_output_shape(self, df):
        X, _ = extract_x_y(df)
        X_transformed = build_preprocessor().fit_transform(X)
        # 7 numerical columns + 3 one-hot columns for M, F, I
        assert X_transformed.shape == (3, 10)

    def test_numerical_columns_are_scaled(self, df):
        X, _ = extract_x_y(df)
        X_transformed = build_preprocessor().fit_transform(X)
        numerical = X_transformed[:, : len(NUMERICAL_COLUMNS)]
        assert numerical.mean(axis=0) == pytest.approx(0)

    def test_unknown_category_is_ignored(self, df):
        X, _ = extract_x_y(df)
        preprocessor = build_preprocessor().fit(X)
        X_new = X.head(1).assign(Sex="unknown")
        one_hot = preprocessor.transform(X_new)[:, len(NUMERICAL_COLUMNS) :]
        assert one_hot.sum() == 0

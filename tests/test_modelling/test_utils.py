import pickle

import pytest
from utils import load_object, pickle_object


class TestPickleObject:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "obj.pkl"
        pickle_object({"a": 1}, path)
        assert path.exists()

    def test_file_contains_object(self, tmp_path):
        path = tmp_path / "obj.pkl"
        pickle_object({"a": 1}, path)
        with open(path, "rb") as f:
            assert pickle.load(f) == {"a": 1}

    def test_creates_missing_parent_folders(self, tmp_path):
        path = tmp_path / "new_folder" / "obj.pkl"
        pickle_object({"a": 1}, path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "obj.pkl"
        pickle_object({"a": 1}, str(path))
        assert path.exists()


class TestLoadObject:
    def test_loads_pickled_object(self, tmp_path):
        path = tmp_path / "obj.pkl"
        with open(path, "wb") as f:
            pickle.dump([1, 2, 3], f)
        assert load_object(path) == [1, 2, 3]

    def test_round_trip_with_pickle_object(self, tmp_path):
        path = tmp_path / "obj.pkl"
        pickle_object({"a": 1}, path)
        assert load_object(path) == {"a": 1}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_object(tmp_path / "missing.pkl")

"""Utility functions to pickle and unpickle objects (model, encoder, ...)."""

import pickle
from pathlib import Path
from typing import Any

from prefect import task


@task(name="Pickle object")
def pickle_object(obj: Any, path: str | Path) -> None:
    """Serialize an object to disk using pickle.

    Parent directories of ``path`` are created if they do not exist.

    Args:
        obj: The Python object to serialize (e.g., a trained model or an encoder).
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str | Path) -> Any:
    """Load a pickled object from disk.

    Args:
        path: Path to the pickle file to load.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    with open(Path(path), "rb") as f:
        return pickle.load(f)

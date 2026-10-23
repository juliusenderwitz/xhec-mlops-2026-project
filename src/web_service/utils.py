"""Load the trained pipeline once per artifact path."""

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=2)
def load_object(filepath: Path) -> Any:
    """Load a pickle from disk and reuse it for subsequent requests."""
    with filepath.open("rb") as file:
        return pickle.load(file)

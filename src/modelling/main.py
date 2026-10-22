"""Training flow: read the data, preprocess it, train a model and save it."""

import argparse
from pathlib import Path

from preprocessing import drop_invalid_rows, extract_x_y, load_data
from sklearn.model_selection import train_test_split
from training import evaluate_model, train_model
from utils import pickle_object

LOCAL_OBJECTS_DIR = (
    Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
)
MODEL_PATH = LOCAL_OBJECTS_DIR / "model.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the training set CSV file.
    """
    # Read data
    df = load_data(trainset_path)

    # Preprocess data
    df = drop_invalid_rows(df)
    X, y = extract_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train model
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(
        " | ".join(
            f"Test {name.upper()}: {value:.3f}" for name, value in metrics.items()
        )
    )

    # Pickle model
    pickle_object(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)

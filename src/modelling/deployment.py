"""Deploy the training flow so the model is retrained every week."""

from pathlib import Path

from main import main
from prefect import serve

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "abalone.csv"

if __name__ == "__main__":
    train_model_deployment = main.to_deployment(
        name="Model training Deployment",
        version="0.1.0",
        tags=["training", "model"],
        cron="0 0 * * 0",
        parameters={"trainset_path": str(DATA_PATH)},
    )
    serve(train_model_deployment)

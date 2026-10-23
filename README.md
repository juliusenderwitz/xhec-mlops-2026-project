# Abalone ring prediction

[![CI](https://github.com/juliusenderwitz/xhec-mlops-2026-project/actions/workflows/ci.yaml/badge.svg)](https://github.com/juliusenderwitz/xhec-mlops-2026-project/actions/workflows/ci.yaml)

This X-HEC MLOps project trains a regression model on the [Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and serves predictions through FastAPI. It also includes MLflow experiments and a Prefect training flow with an optional weekly deployment. The API response field is named `age`, but its value is a **predicted shell ring count**, a floating-point estimate rather than a rounded chronological age.

A fitted model is committed at `src/web_service/local_objects/model.pkl`, so the API works without downloading the dataset or retraining. The CSV is needed for training, experiments, and scheduled retraining.

## Start here with Make

Install Python **3.10 or 3.11**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and GNU Make. From the repository root:

On Windows, install GNU Make with `winget install --id ezwinports.make --exact --scope user`, then open a new terminal so the updated PATH takes effect.

```bash
make setup       # install dependencies from uv.lock
make check       # lint, check formatting, and run tests
make api         # serve the committed model at http://localhost:8000/docs
```

`make check` starts an isolated Prefect test backend automatically; you do not need to start the API or Prefect server first.

To reproduce the model comparison and training results, download the [Abalone CSV](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset), save it as `data/abalone.csv`, then run `make reproduce`. This runs the checks, logs the baseline and linear regression results to MLflow, prints the training metrics, and saves the trained model. If your Prefect profile points to a local server, start it first with `make prefect` in another terminal. **Training overwrites** `src/web_service/local_objects/model.pkl`; restart the API to serve the new model. Use `make mlflow` in a separate terminal to inspect the experiment at [http://localhost:5000](http://localhost:5000).

Run `make` to see the other shortcuts. You can run `make train` or `make experiment` separately, and override the CSV path with `make train DATA=path/to/abalone.csv`. The scheduled deployment uses `data/abalone.csv`. Docker remains optional: `make docker-build` followed by `make docker-run` starts the containerized API and Prefect server. The detailed commands below work directly without Make.

## Quick start: API in Docker

Install [Docker](https://docs.docker.com/get-docker/) and start its daemon. From the repository root:

```bash
docker build -f Dockerfile.app -t abalone-api .
docker run --rm --name abalone-api -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

The container starts the prediction API and a Prefect server. Open the [interactive API documentation](http://localhost:8000/docs) to send a request. The [Prefect UI](http://localhost:4200) is also available. Stop both services with Ctrl+C. The image includes the committed model, but no dataset or scheduled training deployment; the Prefect UI has no training runs until you configure and run a flow.

### Make a prediction

Send a JSON object for **one** abalone to `POST /predict`. Field names with spaces must be quoted exactly as shown. In a macOS/Linux shell:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Sex":"M","Length":0.455,"Diameter":0.365,"Height":0.095,"Whole weight":0.514,"Shucked weight":0.2245,"Viscera weight":0.101,"Shell weight":0.15}'
```

In PowerShell:

```powershell
$body = @{
  Sex = 'M'; Length = 0.455; Diameter = 0.365; Height = 0.095
  'Whole weight' = 0.514; 'Shucked weight' = 0.2245
  'Viscera weight' = 0.101; 'Shell weight' = 0.15
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The response has the form `{"age": 10.5}`; the number depends on the model artifact. `GET /` returns a service check and `GET /docs` shows the schemas. `Sex` accepts `M`, `F`, or `I`; all measurements must be positive and `Height` must be at most `0.4`. Missing, extra, or invalid fields receive a validation error. The endpoint does not accept a batch of records.

## Run locally

Use Python **3.10 or 3.11**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Git. Clone the repository and install locked dependencies:

```bash
git clone https://github.com/juliusenderwitz/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
uv sync --locked
```

`uv` creates `.venv/`; use `uv run` for the commands below. To serve the committed model locally:

```bash
uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8000
```

Use the [prediction example](#make-a-prediction) and URLs above. The local API does not start Prefect automatically. To use a newly trained model, restart the API: each process caches the loaded artifact.

### Dataset and training

Download the [Abalone CSV](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv`. Create `data/` if necessary. Its required header is:

```text
Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings
```

The dataset is ignored by Git and omitted from the Docker image. From the repository root, train once with:

```bash
uv run python src/modelling/main.py data/abalone.csv
```

The Prefect flow loads the CSV, removes invalid rows, splits training and test data, fits a pipeline that scales numeric columns, one-hot encodes `Sex`, and applies linear regression. It prints test RMSE, MAE, and R², then writes the fitted pipeline to `src/web_service/local_objects/model.pkl`. This **overwrites the committed model file** in your checkout. The API reads that path for predictions.

To see flow runs in Prefect, start a server in a separate terminal before training:

```bash
uv run prefect server start --host 127.0.0.1
```

Open [http://localhost:4200](http://localhost:4200). In the training terminal, point the Prefect client at that server and run the flow:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python src/modelling/main.py data/abalone.csv
```

To serve the scheduled deployment, keep the server running and use another terminal:

```bash
uv run python src/modelling/deployment.py
```

The deployment schedules training for **Sunday at 00:00** according to Prefect's configured timezone. Keep its process running for scheduled runs. It reads `data/abalone.csv` from the repository and overwrites the model artifact. Restart the API afterward. A plain Docker run does not start this deployment or include the CSV.

### Compare models with MLflow

With `data/abalone.csv` in place:

```bash
uv run python src/modelling/experiment.py data/abalone.csv
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000). The `abalone-rings` experiment compares a mean baseline and linear regression using five-fold cross-validation and a held-out test split. It logs parameters, RMSE, MAE, R², and fitted pipelines. Local tracking data in `mlflow.db` and `mlruns/` is ignored by Git. Experiments do not replace the API's model; use the training command for that.

## Development and checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pre-commit install
uv run pre-commit run --all-files
```

Pre-commit checks file hygiene and configuration syntax, runs Ruff, and checks the lockfile. GitHub Actions runs lint, formatting, hooks, and tests on Python 3.10 and 3.11 for pushes and pull requests. Add dependencies with `uv add <package>` or development dependencies with `uv add --dev <package>`, then commit both `pyproject.toml` and `uv.lock`.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/modelling/` | Data cleaning, training, inference helpers, Prefect flow and deployment, MLflow experiments |
| `src/web_service/` | FastAPI routes, validated schemas, inference, and the committed model artifact |
| `bin/run_services.sh` | Starts Prefect and Uvicorn in the container |
| `Dockerfile.app` | Builds the API and Prefect image from locked runtime dependencies |
| `notebooks/` | Exploratory analysis and model research |
| `tests/` | Modelling and API checks |
| `.github/workflows/ci.yaml` | Continuous integration |

## Team

| Name | GitHub |
| --- | --- |
| Karin Yamakawa | [@im-on-gh](https://github.com/im-on-gh) |
| Kenza Chaabouni | [@chKenza](https://github.com/chKenza) |
| Julius Enderwitz | [@juliusenderwitz](https://github.com/juliusenderwitz) |
| Carl Aaron Heimann | [@aaron3636](https://github.com/aaron3636) |

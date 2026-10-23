.DEFAULT_GOAL := help

UV ?= uv
DATA ?= data/abalone.csv

.PHONY: help setup test lint format-check check hooks api train experiment reproduce mlflow prefect deploy docker-build docker-run

help:
	@echo make setup       Install locked dependencies
	@echo make check       Run lint, format checks, and tests
	@echo make reproduce   Check, compare models, and train from $(DATA)
	@echo make api         Start the local prediction API
	@echo make mlflow      Open the local MLflow UI
	@echo make prefect     Start the local Prefect server
	@echo make deploy      Serve the weekly training deployment
	@echo make docker-build / docker-run  Build or run the API image

setup:
	$(UV) sync --locked

test:
	$(UV) run --locked pytest

lint:
	$(UV) run --locked ruff check .

format-check:
	$(UV) run --locked ruff format --check .

check: lint format-check test

hooks:
	$(UV) run --locked pre-commit run --all-files --show-diff-on-failure

api:
	$(UV) run --locked uvicorn src.web_service.main:app --host 127.0.0.1 --port 8000

train: $(DATA)
	$(UV) run --locked python src/modelling/main.py "$(DATA)"

experiment: $(DATA)
	$(UV) run --locked python src/modelling/experiment.py "$(DATA)"

reproduce: $(DATA) check experiment train

mlflow:
	$(UV) run --locked mlflow ui --backend-store-uri sqlite:///mlflow.db

prefect:
	$(UV) run --locked prefect server start --host 127.0.0.1

deploy: data/abalone.csv
	$(UV) run --locked python src/modelling/deployment.py

docker-build:
	docker build -f Dockerfile.app -t abalone-api .

docker-run:
	docker run --rm --name abalone-api -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api

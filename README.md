<div align="center">

# MLOps Project: Abalone Age Prediction

[![CI](https://github.com/juliusenderwitz/xhec-mlops-2026-project/actions/workflows/ci.yaml/badge.svg)](https://github.com/juliusenderwitz/xhec-mlops-2026-project/actions/workflows/ci.yaml)
[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
</div>

## Project Overview

This repository is our X-HEC MLOps group project. The goal is to predict the **age of abalone** (a type of sea snail) from physical measurements such as shell weight, diameter and height.

Today, an abalone's age is found by cutting the shell, staining it and counting the rings under a microscope, which takes a long time. A model trained on easy-to-collect measurements can estimate the age much faster.

We are building this as a production-style ML system, step by step:

| Step | Content | Status |
| --- | --- | --- |
| 0 | Environment setup: dependencies, pre-commit hooks, CI | ✅ Done |
| 1 | Model training and inference code | ⏳ To do |
| 2 | Automated training pipeline with Prefect | ⏳ To do |
| 3 | Prediction REST API (FastAPI + Pydantic) in Docker | ⏳ To do |

### Dataset

We use the [Abalone dataset from Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset). Download it and put the CSV file in a `data/` folder at the root of the repository. The `data/` folder is not committed to Git.

## Team

| Name | GitHub |
| --- | --- |
| Karin Yamakawa | [@im-on-gh](https://github.com/im-on-gh) |
| Kenza Chaabouni | [@chKenza](https://github.com/chKenza) |
| Julius Enderwitz | [@juliusenderwitz](https://github.com/juliusenderwitz) |
| Carl Aaron Heimann | [@aaron3636](https://github.com/aaron3636) |

## Getting Started

### Prerequisites

- Python **3.10 or 3.11**
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages the Python environment and dependencies
- Git

To install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 1. Clone the repository

```bash
git clone https://github.com/juliusenderwitz/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
```

### 2. Create the environment

```bash
uv sync
```

This command creates a virtual environment in `.venv/` and installs the exact versions pinned in `uv.lock`, including the dev tools (pytest, ruff, pre-commit).

You can run any command inside the environment with `uv run <command>`. You can also activate the environment yourself:

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install the pre-commit hooks

```bash
uv run pre-commit install
```

After this, the hooks run on every `git commit`. They:

- remove trailing whitespace and add a final newline to files
- check that TOML, YAML and JSON files are valid
- block large files from being committed
- lint and auto-fix the code with `ruff check --fix`
- format the code with `ruff format`
- keep `uv.lock` in sync with `pyproject.toml`

If a hook changes a file, the commit stops. Run `git add` on the changed files and commit again.

### 4. Check that everything works

```bash
uv run pytest                          # run the tests
uv run ruff check .                    # lint
uv run ruff format --check .           # check formatting
uv run pre-commit run --all-files      # run all hooks on the whole repo
```

## Project Structure

```
.
├── .github/workflows/ci.yaml   # GitHub Actions CI
├── .pre-commit-config.yaml     # pre-commit hooks
├── pyproject.toml              # project metadata and dependencies
├── uv.lock                     # pinned dependency versions
├── tests/                      # pytest tests
└── README.md
```

## Continuous Integration

GitHub Actions runs [ci.yaml](.github/workflows/ci.yaml) on every push and pull request, with both Python 3.10 and 3.11. The CI:

1. installs the dependencies with `uv sync --locked` (it fails if `uv.lock` is out of date)
2. lints the code with `ruff check`
3. checks the formatting with `ruff format --check`
4. runs all pre-commit hooks
5. runs the tests with `pytest` and measures coverage

## Managing Dependencies

Add a runtime dependency (needed by the model or the API):

```bash
uv add <package>
```

Add a dev-only dependency (testing, linting, notebooks):

```bash
uv add --dev <package>
```

Both commands update `pyproject.toml` and `uv.lock`. Commit both files so that everyone, including the CI, uses the same versions. After pulling changes from a teammate, run `uv sync` again.

## How We Work

The project is split into numbered branches, one per step. For each step we:

1. switch to the branch and merge the latest `main` into it
2. read the `PR_i.md` file for the step's tasks
3. complete the tasks and make sure the CI passes
4. open **one** pull request to `main` on **our** repository (not the original course repository) and merge it

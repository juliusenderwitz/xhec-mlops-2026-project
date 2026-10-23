FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

WORKDIR /app

ENV UV_NO_DEV=1 \
    PATH="/app/.venv/bin:$PATH" \
    PREFECT_API_URL="http://127.0.0.1:4201/api" \
    PREFECT_SERVER_UI_API_URL="http://localhost:4200/api"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY bin/run_services.sh ./bin/run_services.sh

EXPOSE 8001 4201

CMD ["/bin/bash", "bin/run_services.sh"]

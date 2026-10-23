#!/bin/bash
set -e

prefect server start --host 0.0.0.0 --port 4201 &
prefect_pid=$!

uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001 &
api_pid=$!

shutdown() {
    trap - EXIT INT TERM
    kill "$prefect_pid" "$api_pid" 2>/dev/null || true
    wait "$prefect_pid" "$api_pid" 2>/dev/null || true
}

trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Exit and stop the other service if either one stops.
wait -n "$prefect_pid" "$api_pid"

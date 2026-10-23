"""Shared test fixtures for Prefect-backed flows and tasks."""

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(scope="session", autouse=True)
def prefect_backend():
    """Run Prefect tests against an isolated local server and database."""
    with prefect_test_harness():
        yield

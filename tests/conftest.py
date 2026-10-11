"""Global test configuration and environment variable injection."""

import os

import pytest

# Disable LangSmith tracing during test import to avoid sending heartbeat requests to dummy APIs
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

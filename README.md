# RoboWorld-Agent

RoboWorld-Agent tracks robotics world-model papers on arXiv. It fetches recent work, judges whether a paper models environment dynamics, extracts a structured insight, stores the result, and produces a daily brief.

## Overview

The system is organized as a pipeline:

1. Fetch papers from `cs.RO`, `cs.AI`, and `cs.LG`, then drop obvious mismatches with keyword rules.
2. Classify each paper and extract a structured insight.
3. Read selected PDF sections when the abstract is not enough.
4. Store metadata in SQLite and embeddings in a local vector database.
5. Render a Markdown brief and answer questions over the collected papers.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

`pyproject.toml` declares `requires-python = ">=3.11"`. The lockfile records a separate resolution for each supported Python series, so install the same minor version that produced `uv.lock` when you need the same package set.

## Setup

Two files in the repository root define the environment. Both must be present.

| File | Role |
| --- | --- |
| `pyproject.toml` | Declares the project and version ranges for direct dependencies, including the `dev` group (`ruff`, `pre-commit`, `pytest`, `mypy`). |
| `uv.lock` | Pins the full dependency tree: exact versions, hashes, and the wheel for each platform. |

`uv init` writes `pyproject.toml`. `uv.lock` is written the first time dependencies are resolved (`uv add` or `uv lock`). On a new machine, do not regenerate the lockfile. Install from the copy already in the repository.

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, create the virtual environment and install the locked runtime and dev dependencies:

```bash
uv sync --frozen
```

`--frozen` installs `uv.lock` as it is. It does not resolve against PyPI and does not modify the lockfile. `uv sync` includes the `dev` group by default.

To match a specific Python series, install that interpreter and pass it to sync. For 3.11:

```bash
uv python install 3.11
uv sync --frozen --python 3.11
```

Install the git hooks:

```bash
uv run pre-commit install
```

Confirm the base dependencies import:

```bash
uv run python -c "import arxiv, langchain_core, langgraph, pydantic, fitz, dotenv, qdrant_client"
```

## Configuration

Copy the example environment file and fill in the values you use:

```bash
cp .env.example .env
```

`.env` is gitignored. See `.env.example` for the API, tracing, and storage settings.

## Repository layout

```text
pyproject.toml
uv.lock
.env.example
.pre-commit-config.yaml
src/core/
src/fetchers/
src/agents/
src/tools/
src/storage/
src/workflows/
tests/
data/
```

## Development

Run the tools through uv so they use the project environment:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
uv run pre-commit run --all-files
```

Add or upgrade a dependency with `uv add`. That updates `pyproject.toml` and `uv.lock` together. Commit both files so another machine can reproduce the environment with `uv sync --frozen`.

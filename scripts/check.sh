#!/usr/bin/env bash
# Software checks only: no model weights, GPU, provider credentials or SSH needed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync python -m admin.cli selfcheck
uv run --no-sync pytest -q

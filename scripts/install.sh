#!/usr/bin/env bash
# Install the standalone project's CPU development tools and validator.
# Proof verifier dependencies are CPU packages; hardware attestation is not run here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "error: install uv first (python3 -m pip install uv), then rerun scripts/install.sh" >&2
  exit 1
fi
exec uv sync --frozen --extra dev --extra proof --extra validator "$@"

#!/usr/bin/env bash
set -euo pipefail
# Build SparkProof --registry-snapshot from datasets/registry.jsonl.
#   scripts/export_registry_snapshot.sh
#   scripts/export_registry_snapshot.sh --publish
exec uv run python -m eval.export_registry_snapshot "$@"

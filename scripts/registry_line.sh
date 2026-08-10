#!/usr/bin/env bash
# Build (and optionally append) a datasets/registry.jsonl line from a SparkProof bundle.
#
#   scripts/registry_line.sh --bundle ../SparkProof/bundles/run-001 \
#     --miner alice --repo-id alice/sparkproof-triton-v1
#
#   scripts/registry_line.sh --bundle ../SparkProof/bundles/run-001 \
#     --miner alice --hf-url https://huggingface.co/datasets/alice/sparkproof-triton-v1 \
#     --append datasets/registry.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.registry_line "$@"

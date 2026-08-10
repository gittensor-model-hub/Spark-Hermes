#!/usr/bin/env bash
# Compose verified registry datasets into one SFT mix with provenance.
#
#   scripts/mix_registry.sh mix \\
#     --registry datasets/registry.jsonl \\
#     --sha256 <sha-a> --sha256 <sha-b> \\
#     --out data/processed/mix_sft.jsonl \\
#     --manifest-out data/processed/mix_manifest.json \\
#     --sparkproof-root ../SparkProof
#
#   scripts/mix_registry.sh verify \\
#     --manifest data/processed/mix_manifest.json \\
#     --sft data/processed/mix_sft.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.mix_registry "$@"

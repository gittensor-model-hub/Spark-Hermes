#!/usr/bin/env bash
# Export canonical HF mining dataset to local Axolotl messages jsonl and save
# mix_manifest.json for proof.bundle --mix-manifest.
#
#   scripts/prepare_mining_sft.sh
#   scripts/prepare_mining_sft.sh --out data/processed/sparkproof-mining_sft.jsonl
#   SPARKDISTILL_MINING_DATASET_REPO=user/repo scripts/prepare_mining_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.prepare_mining_sft "$@"

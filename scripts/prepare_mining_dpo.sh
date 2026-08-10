#!/usr/bin/env bash
# Export the canonical HF preference dataset to local Axolotl DPO jsonl
# (chosen/rejected pairs), verified against the pinned pref_sha256.
#
#   scripts/prepare_mining_dpo.sh
#   scripts/prepare_mining_dpo.sh --out data/processed/sparkproof-mining_dpo.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.prepare_mining_dpo "$@"

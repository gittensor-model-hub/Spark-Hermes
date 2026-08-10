#!/usr/bin/env bash
# Fold raw teacher trajectories into <think>-tagged SFT records.
#
#   scripts/prepare_sft_data.sh --in data/processed/phase1_trajectories.jsonl --out data/processed/phase1_sft.jsonl --format messages
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run python -m teacher.format "$@"

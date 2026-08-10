#!/usr/bin/env bash
# Generate teacher trajectories from a prompt set.
#
#   scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl [extra python -m teacher.generate args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec uv run python -m teacher.generate "$@"

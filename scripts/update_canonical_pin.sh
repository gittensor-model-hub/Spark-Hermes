#!/usr/bin/env bash
# Refresh datasets/canonical.json from the live HF mining dataset manifest.
#
#   scripts/update_canonical_pin.sh            # refresh file only
#   scripts/update_canonical_pin.sh --publish  # refresh + land on main (PR fallback)
#
# Run after a dataset registry merge republishes gittensor-model-hub/sparkproof-mining.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.update_canonical_pin "$@"

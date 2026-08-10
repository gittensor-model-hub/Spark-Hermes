#!/usr/bin/env bash
# Train a student checkpoint from an Axolotl recipe.
#
#   scripts/install_train.sh
#   scripts/prepare_mining_sft.sh
#   scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml [--dry-run]
#
# Resolves recipe paths from the SparkDistill root, disables sample packing on
# small jsonl mixes, and falls back to SDPA when flash-attn is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

recipe="${1:?usage: scripts/train.sh <recipe.yaml> [--dry-run] [extra axolotl args...]}"
shift

if [ ! -f "$recipe" ]; then
  echo "recipe not found: $recipe" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run scripts/install_train.sh" >&2
  exit 1
fi

uv run --no-sync python -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$recipe"

dry_run=false
args=()
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    dry_run=true
  else
    args+=("$arg")
  fi
done

if ! uv run --no-sync python -c "import axolotl" 2>/dev/null; then
  echo "error: axolotl not installed — run scripts/install_train.sh" >&2
  exit 1
fi

if ! uv run --no-sync python -c "import torchvision" 2>/dev/null; then
  echo "error: torchvision not installed (required for Qwen3.5) — run scripts/install_train.sh" >&2
  exit 1
fi

prep_json="$(uv run --no-sync python -m eval.train_prep --recipe "$recipe" --root "$ROOT")"
prepared_recipe="$(printf '%s' "$prep_json" | uv run --no-sync python -c 'import json,sys; print(json.load(sys.stdin)["prepared_recipe"])')"
printf '%s\n' "$prep_json" | uv run --no-sync python -c 'import json,sys; d=json.load(sys.stdin); [print("train prep:", n) for n in d.get("notes", [])]'

if [ "$dry_run" = true ]; then
  echo "[dry-run] would run: uv run --no-sync axolotl train $prepared_recipe ${args[*]:-}"
  exit 0
fi

if [ "${#args[@]}" -gt 0 ]; then
  exec uv run --no-sync axolotl train "$prepared_recipe" "${args[@]}"
fi
exec uv run --no-sync axolotl train "$prepared_recipe"

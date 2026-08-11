#!/usr/bin/env bash
# Fold a trained LoRA adapter back into its base, after checking it is the right one.
#
#   scripts/train.sh hermes/recipes/spark-hermes-agent-3.8-27b/stage-c-tools.yaml
#   scripts/merge_lora.sh hermes/recipes/spark-hermes-agent-3.8-27b/stage-c-tools.yaml
#     -> outputs/spark-hermes-agent-3.8-27b/stage-c/merged
#
# That path is what stage D starts from, and what gets served for the M1 benchmark.
#
# `hermes.merge` runs first and this script refuses to call axolotl on a non-empty verdict.
# The merges worth stopping all succeed: an adapter with no weights merges to the base model,
# an adapter from another run merges into a model whose weights match no recipe, and both
# serve and benchmark without complaint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

recipe="${1:?usage: scripts/merge_lora.sh <recipe.yaml> [--adapter DIR] [--force] [extra axolotl args...]}"
shift

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run scripts/install_train.sh" >&2
  exit 1
fi

preflight=()
args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --adapter)
      preflight+=("--adapter" "${2:?--adapter needs a directory}")
      shift 2
      ;;
    --force)
      preflight+=("--force")
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Every reason, printed, before anything is written.
uv run --no-sync python -m hermes.merge --recipe "$recipe" "${preflight[@]+"${preflight[@]}"}"

if ! uv run --no-sync python -c "import axolotl" 2>/dev/null; then
  echo "error: axolotl not installed — run scripts/install_train.sh" >&2
  exit 1
fi

# The command the preflight printed, rebuilt from the same source rather than parsed back out
# of its output: two ways of deciding which adapter to merge is one more than can be kept in step.
plan_json="$(uv run --no-sync python -m hermes.merge --recipe "$recipe" --json "${preflight[@]+"${preflight[@]}"}")"
adapter="$(printf '%s' "$plan_json" | uv run --no-sync python -c 'import json,sys; print(json.load(sys.stdin)["adapter"])')"
merged="$(printf '%s' "$plan_json" | uv run --no-sync python -c 'import json,sys; print(json.load(sys.stdin)["merged_dir"])')"

echo "merging $adapter -> $merged"
if [ "${#args[@]}" -gt 0 ]; then
  uv run --no-sync axolotl merge-lora "$recipe" "--lora-model-dir=$adapter" "${args[@]}"
else
  uv run --no-sync axolotl merge-lora "$recipe" "--lora-model-dir=$adapter"
fi

# Axolotl reports success on a merge that wrote no weights, so the result is checked rather
# than assumed. A directory that exists is not a model.
if ! ls "$merged"/*.safetensors >/dev/null 2>&1 && ! ls "$merged"/*.bin >/dev/null 2>&1; then
  echo "error: $merged holds no weight shards; the merge reported success and produced nothing" >&2
  exit 1
fi
echo "merged: $merged"

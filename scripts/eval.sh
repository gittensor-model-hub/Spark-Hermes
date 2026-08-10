#!/usr/bin/env bash
# Score a checkpoint against the quality benchmark basket, optionally
# comparing against the current frontier checkpoint's scores.
#
#   scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 [--compare-frontier]
#       [--frontier-scores eval/results/frontier.json | --repo-frontier]
#       [--gpu-architecture blackwell|hopper]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

checkpoint=""
compare_frontier=false
frontier_scores=""
repo_frontier=false
gpu_architecture=""
extra_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --compare-frontier) compare_frontier=true; shift ;;
    --frontier-scores) frontier_scores="$2"; shift 2 ;;
    --repo-frontier) repo_frontier=true; shift ;;
    --gpu-architecture) gpu_architecture="$2"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

if [ -z "$checkpoint" ]; then
  echo "usage: scripts/eval.sh --checkpoint <path> [--compare-frontier] [--frontier-scores <path> | --repo-frontier] [--gpu-architecture blackwell|hopper]" >&2
  exit 1
fi

candidate_scores="eval/results/candidate.json"
if [ "${#extra_args[@]}" -gt 0 ]; then
  uv run python -m eval.harness --checkpoint "$checkpoint" --out "$candidate_scores" "${extra_args[@]}"
else
  uv run python -m eval.harness --checkpoint "$checkpoint" --out "$candidate_scores"
fi

if [ "$compare_frontier" = true ]; then
  score_args=(--candidate "$candidate_scores" --out eval/results/report.json)
  if [ -n "$gpu_architecture" ]; then
    score_args+=(--gpu-architecture "$gpu_architecture")
  fi
  if [ "$repo_frontier" = true ]; then
    score_args+=(--frontiers runs/frontiers.json)
  else
    if [ -z "$frontier_scores" ]; then
      frontier_scores="eval/results/frontier.json"
    fi
    if [ ! -f "$frontier_scores" ]; then
      echo "no frontier scores found at $frontier_scores — run eval.harness on the current frontier checkpoint first, or pass --repo-frontier" >&2
      exit 1
    fi
    score_args+=(--frontier "$frontier_scores")
  fi
  uv run python -m eval.score "${score_args[@]}"
  cat eval/results/report.json
fi

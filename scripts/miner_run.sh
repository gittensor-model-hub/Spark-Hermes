#!/usr/bin/env bash
# Delegate to SparkProof/scripts/miner_run.sh (Blackwell CC one-shot pipeline).
set -euo pipefail

distill_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sparkproof_root="${SPARKPROOF_ROOT:-$distill_root/../SparkProof}"

if [ ! -x "$sparkproof_root/scripts/miner_run.sh" ]; then
  echo "error: SparkProof not found at $sparkproof_root — clone beside SparkDistill or set SPARKPROOF_ROOT" >&2
  exit 1
fi

exec "$sparkproof_root/scripts/miner_run.sh" --sparkdistill "$distill_root" "$@"

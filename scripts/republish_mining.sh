#!/usr/bin/env bash
# Re-aggregate datasets/registry.jsonl with exact dedupe and republish sparkproof-mining.
#
#   HF_TOKEN=... scripts/republish_mining.sh
#
# Use after changing SPARKDISTILL_MINING_DEDUPE policy or to recover rows dropped by
# the older near-dedupe default. Then run scripts/update_canonical_pin.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

export SPARKDISTILL_MINING_DEDUPE="${SPARKDISTILL_MINING_DEDUPE:-exact}"

uv run python - <<'PY'
import json
from pathlib import Path

from eval.mining_dataset import aggregate_and_publish_mining_dataset, mining_dataset_repo

registry_path = Path("datasets/registry.jsonl")
entries = [json.loads(line) for line in registry_path.read_text().splitlines() if line.strip()]
report = aggregate_and_publish_mining_dataset(entries, repo_id=mining_dataset_repo())
print(json.dumps(report, indent=2))
if not report.get("published"):
    raise SystemExit(1)
PY

exec scripts/update_canonical_pin.sh

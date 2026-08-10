"""Append a merged run's result to the immutable `runs/ledger.jsonl` log.

Maintainer/eval-bot-owned: this is the last step of merging a proof-of-training PR,
run after `eval.verify` (or a full retrain-verification) has produced a report.

    python -m eval.ledger --pr https://github.com/gittensor-ai-lab/sparkdistill/pull/123 \\
        --hf-proof-url https://huggingface.co/miner/sparkdistill-run-001 \\
        --report eval/results/report.json --run-id 2026-07-09-qwen3.5-4b-001 \\
        [--attestation runs/2026-07-09-qwen3.5-4b-001/attestation.json] \\
        --out runs/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class LedgerEntry:
    date: str
    run_id: str
    pr: str
    hf_proof_url: str
    label: str
    best_benchmark: str | None
    best_pct_delta: float | None
    regressions: list[str]
    attested: bool


def build_entry(run_id: str, pr: str, hf_proof_url: str, report: dict, attestation: dict | None) -> LedgerEntry:
    return LedgerEntry(
        date=datetime.now(UTC).isoformat(),
        run_id=run_id,
        pr=pr,
        hf_proof_url=hf_proof_url,
        label=report["label"],
        best_benchmark=report.get("best_benchmark"),
        best_pct_delta=report.get("best_pct_delta"),
        regressions=report.get("regressions", []),
        attested=bool(attestation and attestation.get("passed")),
    )


def append_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--hf-proof-url", required=True)
    parser.add_argument("--report", type=Path, required=True, help="report json from eval.score or eval.verify")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attestation", type=Path, default=None, help="attestation json from eval.attestation")
    parser.add_argument("--out", type=Path, default=Path("runs/ledger.jsonl"))
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    attestation = json.loads(args.attestation.read_text()) if args.attestation else None
    entry = build_entry(args.run_id, args.pr, args.hf_proof_url, report, attestation)

    append_entry(args.out, entry)
    print(f"appended {entry.run_id} ({entry.label}) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

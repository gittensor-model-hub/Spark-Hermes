"""Persist the labelled two-round learning CLI self-check in a fresh directory."""

import argparse
import json
from pathlib import Path

from test_learning_boundary import test_two_round_cli_replay_and_repeated_sft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        parser.error("fixture root must be new; existing evidence is never overwritten")
    root.mkdir(parents=True)
    test_two_round_cli_replay_and_repeated_sft(root)
    records = [json.loads(line) for line in (root / "commands.jsonl").read_text().splitlines()]
    print(
        json.dumps(
            {
                "mode": "fixture",
                "namespace": "cpu-learning",
                "commands": len(records),
                "expected_refusals": sum(r["exit_code"] == 2 for r in records),
                "evidence": str(root),
                "model_inference": False,
                "live_actions": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

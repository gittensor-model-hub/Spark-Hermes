"""Run a complete isolated CPU fixture through the operator entry points."""

import argparse
import json
from pathlib import Path

from settlement_support import cli, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        parser.error("fixture root must be new; existing state is never overwritten")
    prepare(root)
    cli(
        root,
        "validator.crown",
        "select",
        "--store",
        root / "rounds",
        "--settlement-root",
        root / "settlement",
        "--scorecards",
        root / "cards",
        "--episodes",
        root / "episodes",
        "--out",
        root / "settled.json",
    )
    cli(
        root,
        "validator.crown",
        "actions",
        "--store",
        root / "rounds",
        "--settlement-root",
        root / "settlement",
        "--out",
        root / "actions.json",
    )
    cli(
        root,
        "validator.settlement",
        "deliver",
        "--settlement-root",
        root / "settlement",
        "--out",
        root / "dry-run.json",
    )
    record = json.loads((root / "settled.json").read_text())
    print(
        json.dumps(
            {
                "mode": record["mode"],
                "winner": record["outcome"]["winner"]["miner_id"],
                "refused": record["outcome"]["ineligible"],
                "settled_artifact": str(root / "settled.json"),
                "actions": len(record["actions"]),
                "external_mutations": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

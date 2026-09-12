"""Actual installed operator commands on labelled CPU release artifacts; no inference."""

import argparse
import json
from pathlib import Path

from learning_support import cli
from release_support import setup_release

from admin.artifacts import write_record
from hermes.cotraining import DEFAULT_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    authority, candidates, _, _ = setup_release(root)
    candidate = candidates[1]["payload"]
    registered = json.loads(
        cli(
            root,
            "admin.cli",
            "candidates",
            "register",
            "--root",
            authority.candidates().store.root,
            "--workspace",
            candidate["workspace"],
            "--merged-record",
            candidate["model"]["record"],
            "--agent",
            candidate["agent"]["path"],
            "--workload",
            candidate["workload"]["path"],
            "--parent",
            candidate["parent"]["path"],
        ).stdout
    )
    assert registered["id"] == candidates[1]["id"]
    config = root / "release-config.json"
    write_record(
        config,
        {
            "candidates": str(authority.candidates().store.root),
            "incumbent": candidates[0]["id"],
            "data_policy": str(root / "confirmation.json"),
            "policy": DEFAULT_POLICY,
        },
    )
    cli(root, "admin.cli", "release", "init", "--root", authority.store.root, "--config", config)
    plan = json.loads(
        cli(
            root, "admin.cli", "cotraining", "freeze", "--root", authority.store.root, "--config", root / "freeze.json"
        ).stdout
    )
    evaluated = json.loads(
        cli(
            root,
            "admin.cli",
            "cotraining",
            "run",
            "--root",
            authority.store.root,
            "--id",
            plan["id"],
            "--allow-unsandboxed",
        ).stdout
    )
    cli(root, "admin.cli", "cotraining", "show", "--root", authority.store.root, "--id", evaluated["id"])
    decision = json.loads(
        cli(root, "admin.cli", "release", "decide", "--root", authority.store.root, "--id", evaluated["id"]).stdout
    )
    assert decision["payload"]["result"] == "accepted"
    activated = json.loads(
        cli(root, "admin.cli", "release", "activate", "--root", authority.store.root, "--id", decision["id"]).stdout
    )
    cli(root, "admin.cli", "release", "status", "--root", authority.store.root)
    path = Path(candidate["agent"]["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        cli(
            root,
            "admin.cli",
            "candidates",
            "show",
            "--root",
            authority.candidates().store.root,
            "--id",
            candidates[1]["id"],
            expected=2,
        )
        cli(
            root, "admin.cli", "release", "activate", "--root", authority.store.root, "--id", decision["id"], expected=2
        )
    finally:
        path.write_bytes(original)
    assert authority.status() == activated
    report = {
        "fixture_only": True,
        "model_inference_executed": False,
        "policy": DEFAULT_POLICY,
        "candidate_id": candidates[1]["id"],
        "plan_id": plan["id"],
        "evaluation_id": evaluated["id"],
        "approval_id": decision["id"],
        "report": evaluated["payload"]["report"],
        "incumbent": activated,
        "commands": str(root / "commands.jsonl"),
    }
    write_record(root / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

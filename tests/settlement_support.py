"""CPU fixture producers; only isolated fixture stores, never production approvals."""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from competition_support import GitHubTransport, rows, window

from eval.strategy_track import Commitment
from validator.intake import Intake
from validator.pr_admission import main as admission_main
from validator.settlement import SettlementStore
from validator.store import RoundStore

REPOSITORY = "example/spark"


def cli(root, module, *args, expected=0):
    command = [sys.executable, "-m", module, *map(str, args)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    with (root / "commands.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {"argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            )
            + "\n"
        )
    assert result.returncode == expected, result.stderr + result.stdout
    return result


def prepare(root: Path, *, round_id="r-1", tokens=None, pr_numbers=None):
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GH_TOKEN"] = "fixture-credential"
    store = RoundStore(root / "rounds", mode="fixture", namespace="cpu-settlement")
    win = window(store, round_id=round_id, spread=False)
    intake = Intake(root / "bundles", root / "receipts.jsonl", mode="fixture", namespace="cpu-settlement")
    tokens = tokens or {"alice": 60000, "bob": 82650, "carol": 60000}
    remote = {"prs": {}, "mutations": []}
    for number, (miner, cost) in enumerate(tokens.items(), start=7):
        number = (pr_numbers or {}).get(miner, number)
        receipt = intake.accept(
            round_id=round_id, miner_id=miner, files={"SOUL.md": f"Fixture strategy {miner}"}, now=10
        )
        record = Commitment(round_id, miner, receipt.bundle_sha256, (win.task_id,)).to_record()
        transport = GitHubTransport(win.assignment.to_record(reveal_seed=True), record)
        transport.number = number
        with redirect_stdout(io.StringIO()):
            assert (
                admission_main(
                    [
                        "--repository",
                        REPOSITORY,
                        "--pr",
                        str(number),
                        "--head",
                        transport.head,
                        "--round",
                        round_id,
                        "--store",
                        str(store.root),
                        "--intake-root",
                        str(intake.root),
                        "--receipts",
                        str(intake.receipts),
                    ],
                    transport=transport,
                )
                == 0
            )
        # Tie timestamps are derived from genuine admissions, not forged scorecard fields.
        remote["prs"][str(number)] = {
            "number": number,
            "base": {"repo": {"full_name": REPOSITORY}},
            "head": {"sha": transport.head},
            "user": {"login": miner},
            "state": "open",
            "merged": False,
            "draft": False,
            "labels": [],
            "reviews": [],
        }
        data = rows(
            tokens=cost,
            epoch=win.challenge.epoch,
            origin=store.identity,
            round_id=round_id,
            digest=receipt.bundle_sha256,
        )
        source = root / "fixture-input" / round_id / f"{miner}.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("".join(json.dumps(r) + "\n" for r in data))
    remote_path = root / "github-fixture.json"
    if remote_path.exists():
        previous = json.loads(remote_path.read_text())
        previous["prs"].update(remote["prs"])
        remote = previous
    remote_path.write_text(json.dumps(remote))
    cli(
        root,
        "validator.judge",
        "freeze",
        "--round",
        round_id,
        "--store",
        store.root,
        "--reason",
        "isolated CPU fixture",
    )
    cli(
        root,
        "validator.judge",
        "judge",
        "--round",
        round_id,
        "--store",
        store.root,
        "--intake-root",
        intake.root,
        "--receipts",
        intake.receipts,
        "--workspace",
        root / "episodes",
        "--scorecards",
        root / "cards",
        "--fixture-episodes",
        root / "fixture-input",
        "--no-settle",
    )
    cli(
        root,
        "validator.settlement",
        "activate",
        "--round",
        round_id,
        "--store",
        store.root,
        "--settlement-root",
        root / "settlement",
        "--repository",
        REPOSITORY,
        "--mode",
        "fixture",
        "--namespace",
        "cpu-settlement",
    )
    return store, SettlementStore(root / "settlement")


def settle(root, round_id=None, hook=None):
    return SettlementStore(root / "settlement").settle_round(
        RoundStore(root / "rounds"),
        scorecards=root / "cards",
        episodes=root / "episodes",
        round_id=round_id,
        hook=hook,
    )


class ActionTransport:
    """Persistent controlled gh subprocess responses, usable after a process crash."""

    def __init__(self, path, *, fail=False, lost_response=False):
        self.path = path
        self.fail = fail
        self.lost_response = lost_response

    def __call__(self, argv, **kwargs):
        assert argv[:4] == ["gh", "api", "--hostname", "github.com"]
        assert kwargs["env"]["GH_TOKEN"] == "fixture-credential"
        method, endpoint = argv[5:7]
        state = json.loads(self.path.read_text())
        if endpoint == "user":
            value = {"login": "validator-service"}
        else:
            parts = endpoint.split("?")[0].split("/")
            pr = state["prs"][parts[4]]
            if method == "GET":
                value = pr[parts[5]] if len(parts) > 5 else pr
                if isinstance(value, list) and "page=" in endpoint:
                    page = int(endpoint.split("page=")[-1])
                    value = value[(page - 1) * 100 : page * 100]
            else:
                if self.fail:
                    return SimpleNamespace(returncode=1, stdout="", stderr="controlled failure")
                payload = json.loads(kwargs["input"])
                state["mutations"].append({"method": method, "endpoint": endpoint, "payload": payload})
                if parts[3] == "issues":
                    if method == "POST":
                        pr["labels"] = [{"name": name} for name in payload["labels"]]
                    else:
                        pr["labels"] = [x for x in pr["labels"] if x["name"] != parts[6]]
                elif len(parts) > 5:
                    pr["reviews"].append(
                        {
                            "id": len(pr["reviews"]) + 1,
                            "user": {"login": "validator-service"},
                            "body": payload["body"],
                            "commit_id": payload["commit_id"],
                            "state": "COMMENTED",
                        }
                    )
                else:
                    pr["state"] = payload["state"]
                self.path.write_text(json.dumps(state))
                if self.lost_response:
                    raise subprocess.TimeoutExpired(argv, 30)
                value = {}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

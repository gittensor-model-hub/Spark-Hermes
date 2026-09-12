"""Labelled CPU fixtures using the production GitHub adapter, intake and round store."""

import base64
import hashlib
import json
from types import SimpleNamespace

from hermes.challenge import Attempt, Baseline, open_challenge
from hermes.round import open_round
from hermes.seed import OPEN, Round
from validator.pr_admission import GitHubSource, admit
from validator.score import policy_record

TASK = "tc-log-rotation-order"
EPOCH = {
    "model_revision": "a" * 40,
    "harness_digest": "b" * 64,
    "epoch_id": "fixture-epoch-1",
    "attempt_ids": [str(i) for i in range(10)],
    "score_policy": policy_record(),
}
VERIFY = "sha256:" + "c" * 64


def rows(
    n=10,
    *,
    tokens=30000,
    public=True,
    hidden=True,
    malformed=0,
    epoch=None,
    origin=None,
    round_id="r-1",
    digest="",
    task=TASK,
):
    epoch = epoch or EPOCH
    return [
        {
            "task_id": task,
            "public_passed": public,
            "hidden_passed": hidden,
            "success": public and hidden is not False,
            "tokens_used": tokens,
            "tool_calls": 6,
            "wall_time_s": 1.0,
            "steps": 20,
            "malformed_turns": malformed,
            "protocol_clean": malformed == 0,
            "setup_failed": False,
            "max_steps_hit": False,
            "disqualified": False,
            "integrity_clean": True,
            "integrity_fully_checked": True,
            "verify_digest": VERIFY,
            "attempt_id": str(i),
            "epoch_id": epoch["epoch_id"],
            "model_revision": epoch["model_revision"],
            "harness_digest": epoch["harness_digest"],
            "origin": origin,
            "round_id": round_id,
            "bundle_sha256": digest,
        }
        for i in range(n)
    ]


def challenge(*, epoch=None, origin=None, spread=True, n=10, private=True):
    evidence = rows(n, tokens=87000, public=False, hidden=False if private else None, epoch=epoch, origin=origin)
    attempts = []
    for i, row in enumerate(evidence):
        row["public_passed"] = i < 4
        row["hidden_passed"] = (i < 4) if private else None
        row["success"] = i < 4
        row.update(tool_calls=11, wall_time_s=317.0, steps=34)
        row["tokens_used"] = 78000 + i * 2000 if spread else 87000
        attempts.append(
            Attempt(
                public_passed=row["public_passed"],
                hidden_passed=row["hidden_passed"],
                tokens=row["tokens_used"],
                tool_calls=11,
                wall_time_s=317.0,
                steps=34,
                evidence=row,
            )
        )
    return open_challenge(
        Baseline(task_id=TASK, attempts=tuple(attempts)),
        epoch=epoch or EPOCH,
        task_pins={
            "task_id": TASK,
            "verify_digest": VERIFY,
            "private_check_required": private,
            "hidden_verify_commitment": "sha256:" + "c" * 64 if private else "",
        },
    )


def window(store, *, round_id="r-1", private=True, spread=True):
    assignment = Round(
        round_id=round_id,
        seed="a" * 64,
        task_ids=(TASK,),
        miner_ids=("alice", "bob", "carol", "dave"),
        replicas=4,
        state=OPEN,
    )
    result = open_round(
        challenge(origin=store.identity, private=private, spread=spread),
        round_id=round_id,
        opened_at=0.0,
        deadline=1e12,
        assignment=assignment,
    )
    store.save(result)
    return store.load(round_id)


class GitHubTransport:
    """Authenticated HTTP response fixture carried through the real gh adapter boundary."""

    def __init__(self, round_record, record, *, author=None, base_text=""):
        self.repository = "example/spark"
        self.head = "1" * 40
        self.base = "2" * 40
        self.author = author or record["miner_id"]
        self.number = 7
        self.base_text = base_text
        self.head_text = base_text + ("\n" if base_text else "") + json.dumps(record)
        self.round_record = round_record
        self.calls = []
        self.state = "open"
        self.extra = None

    def __call__(self, argv, **kwargs):
        assert argv[:7] == ["gh", "api", "--hostname", "github.com", "--method", "GET", argv[6]]
        assert kwargs["env"]["GH_TOKEN"] == "fixture-credential"
        endpoint = argv[-1]
        self.calls.append(endpoint)
        if endpoint == "user":
            value = {"login": "validator-service"}
        elif "/files?" in endpoint:
            value = [{"filename": "datasets/strategies.jsonl", "status": "modified"}]
            if self.extra:
                value.append({"filename": self.extra, "status": "modified"})
        elif "/contents/" in endpoint:
            path = endpoint.split("/contents/")[1].split("?")[0]
            body = (
                json.dumps(self.round_record)
                if "/rounds/" in path
                else self.base_text
                if endpoint.endswith(self.base)
                else self.head_text
            )
            raw = body.encode()
            value = {
                "type": "file",
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(raw).decode(),
                "sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
            }
        else:
            value = {
                "number": self.number,
                "user": {"login": self.author},
                "head": {"sha": self.head, "repo": {"full_name": self.repository}},
                "base": {"sha": self.base, "repo": {"full_name": self.repository}},
                "state": self.state,
                "draft": False,
                "merged": False,
                "changed_files": 2 if self.extra else 1,
            }
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")


def admit_receipt(store, intake, receipt, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    win = store.load(receipt.round_id)
    from eval.strategy_track import Commitment

    record = Commitment(
        round_id=receipt.round_id, miner_id=receipt.miner_id, bundle_sha256=receipt.bundle_sha256, task_ids=(TASK,)
    ).to_record()
    transport = GitHubTransport(win.assignment.to_record(reveal_seed=True), record)
    metadata = GitHubSource(transport.repository, transport=transport).collect(7, round_id=receipt.round_id)
    return admit(metadata=metadata, round_id=receipt.round_id, store=store, intake=intake, now=20.0)

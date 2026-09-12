"""Failure-oriented scoring through the persisted round and judge surfaces."""

import dataclasses
import json

import pytest
from competition_support import EPOCH, admit_receipt, rows, window

from validator.intake import Intake
from validator.judge import judge_round
from validator.score import ScoreError, score
from validator.store import RoundStore


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture", namespace="score-evidence")
    window(store)
    intake = Intake(
        tmp_path / "bundles", tmp_path / "receipts.jsonl", mode="fixture", namespace=store.identity["namespace"]
    )
    receipt = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "CPU fixture"}, now=10)
    admit_receipt(store, intake, receipt, monkeypatch)
    return store, intake, receipt


def evaluate(store, receipt, evidence):
    return score(
        window=store.load("r-1"),
        miner_id="alice",
        rows=evidence,
        model_revision=EPOCH["model_revision"],
        harness_digest=EPOCH["harness_digest"],
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("public_passed", "true"),
        ("hidden_passed", "true"),
        ("hidden_passed", None),
        ("success", "true"),
        ("disqualified", True),
        ("disqualified", "false"),
        ("integrity_clean", False),
        ("integrity_fully_checked", False),
        ("setup_failed", True),
        ("max_steps_hit", True),
        ("protocol_clean", False),
        ("malformed_turns", 1),
        ("malformed_turns", "0"),
        ("tokens_used", -1),
        ("tokens_used", 0),
        ("tokens_used", float("nan")),
        ("tokens_used", float("inf")),
        ("tokens_used", True),
        ("tokens_used", "100"),
        ("tool_calls", -1),
        ("wall_time_s", float("nan")),
        ("wall_time_s", -1),
        ("cost", -1),
        ("task_id", "wrong-task"),
        ("model_revision", "stale"),
        ("harness_digest", "stale"),
        ("epoch_id", "stale"),
        ("round_id", "old-round"),
        ("attempt_id", "1"),
        ("bundle_sha256", "sha256:" + "0" * 64),
        ("verify_digest", "old-verifier"),
        ("origin", {"mode": "production", "namespace": "copied", "issuer": "fake"}),
    ],
)
def test_bad_execution_or_identity_cannot_persist_a_scorecard(admitted, tmp_path, key, value):
    store, intake, receipt = admitted
    evidence = rows(origin=store.identity, digest=receipt.bundle_sha256)
    evidence[0][key] = value
    win = store.load("r-1")
    win.freeze(now=30, reason="fixture")
    store.save(win)

    def run(miner, bundle, workspace):
        path = workspace / "episodes.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in evidence))
        return path

    results = judge_round(
        round_id="r-1",
        run=run,
        model_revision=EPOCH["model_revision"],
        store=store,
        intake=intake,
        workspace=tmp_path / "run",
        scorecard_dir=tmp_path / "cards",
    )
    assert not results[0].ok, (key, value)
    assert not list((tmp_path / "cards").glob("*.json"))
    assert store.load("r-1").verdicts == {}


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "excess",
        "duplicate",
        "missing_flag",
        "baseline_tokens",
        "baseline_identity",
        "baseline_integrity",
        "baseline_private",
        "baseline_count",
    ],
)
def test_schedule_and_baseline_fail_closed(admitted, change):
    store, _, receipt = admitted
    evidence = rows(origin=store.identity, digest=receipt.bundle_sha256)
    win = store.load("r-1")
    if change == "missing":
        evidence.pop()
    elif change == "excess":
        evidence.append({**evidence[0], "attempt_id": "extra"})
    elif change == "duplicate":
        evidence[-1]["attempt_id"] = evidence[0]["attempt_id"]
    elif change == "missing_flag":
        del evidence[0]["integrity_clean"]
    else:
        attempts = list(win.challenge.baseline.attempts)
        first = attempts[0]
        if change == "baseline_tokens":
            attempts[0] = dataclasses.replace(first, tokens=-1)
        elif change == "baseline_identity":
            attempts[0] = dataclasses.replace(first, evidence={**first.evidence, "epoch_id": "old"})
        elif change == "baseline_integrity":
            attempts[0] = dataclasses.replace(first, evidence={**first.evidence, "integrity_fully_checked": False})
        elif change == "baseline_private":
            attempts[0] = dataclasses.replace(first, hidden_passed=None)
        else:
            attempts.pop()
        win.challenge = dataclasses.replace(
            win.challenge, baseline=dataclasses.replace(win.challenge.baseline, attempts=tuple(attempts))
        )
        store.save(win)
    with pytest.raises(ScoreError):
        evaluate(store, receipt, evidence)


def test_explicit_private_free_task_has_valid_baseline_and_candidate(tmp_path, monkeypatch):
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture")
    window(store, private=False)
    intake = Intake(
        tmp_path / "bundles", tmp_path / "receipts.jsonl", mode="fixture", namespace=store.identity["namespace"]
    )
    receipt = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "private-free fixture"}, now=10)
    admit_receipt(store, intake, receipt, monkeypatch)
    card = evaluate(store, receipt, rows(hidden=None, origin=store.identity, digest=receipt.bundle_sha256))
    assert card.accepted
    assert card.candidate.passes == 10
    assert card.policy["bootstrap_resamples"] == 2000
    assert card.policy["bootstrap_seed"] == 20260810
    assert card.policy["min_token_reduction"] == 0.20


def test_valid_control_and_verifier_failure_have_different_credit(admitted):
    store, _, receipt = admitted
    good = rows(origin=store.identity, digest=receipt.bundle_sha256)
    card = evaluate(store, receipt, good)
    assert card.accepted and card.candidate.passes == 10
    failed = [{**r, "hidden_passed": False, "success": False} for r in good]
    card = evaluate(store, receipt, failed)
    assert not card.accepted and card.overfit_attempts == 10 and card.candidate.passes == 0
    assert sum(card.candidate.tokens) == 300000, "failed attempts retain their full costs"


def test_real_cpu_runner_emits_private_free_evidence_and_attempt_ids(tmp_path):
    """Actual subprocess verification and sink; no model/provider or oracle replacement."""
    from hermes.challenge import episode_metrics_of
    from hermes.trajectory import FINAL, TOOL_CALL, Step
    from hermesbench.runner import LocalToolExecutor, run_suite, verify_digest
    from hermesbench.sink import JsonlEpisodeSink, read_episodes
    from hermesbench.tasks import Task
    from validator.score import candidate_arm

    task = Task.from_record(
        {
            "task_id": "cpu-private-free",
            "prompt": "create done.txt",
            "tools": ["terminal"],
            "verify": "test -f done.txt",
            "max_steps": 5,
            "timeout_s": 5,
        }
    )

    class Policy:
        tokens_used = 42

        def __init__(self):
            self.turn = 0

        def next_steps(self, task, history):
            self.turn += 1
            if self.turn == 1:
                return [
                    Step(
                        kind=TOOL_CALL,
                        tool="terminal",
                        args={"command": "touch done.txt && test -f done.txt"},
                        call_id="c1",
                    )
                ]
            return [Step(kind=FINAL, content="done")]

    origin = {"mode": "fixture", "namespace": "cpu-runner", "issuer": "test"}
    context = {"epoch": EPOCH, "origin": origin, "round_id": "r-1", "bundle_sha256": "sha256:" + "e" * 64}
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        _, results = run_suite(
            [task],
            lambda _: Policy(),
            LocalToolExecutor(allow_unsandboxed=True),
            tmp_path / "ws",
            repeats=10,
            sink=sink,
            evaluation_context=context,
            max_concurrency=2,
        )
    measured = [episode_metrics_of(r) for r in read_episodes(log)]
    assert len(results) == 10
    assert {r["attempt_id"] for r in measured} == set(EPOCH["attempt_ids"])
    assert all(r["origin"] == origin and r["verify_digest"] == verify_digest(task) for r in measured)
    arm, _, _ = candidate_arm(measured, private_required=False)
    assert arm.passes == 10 and arm.tokens == (42,) * 10


def test_existing_explicit_private_declaration_is_preserved():
    from types import SimpleNamespace

    from validator.score import private_check_required

    assert private_check_required(SimpleNamespace(task_pins={"declares_hidden_tests": False})) is False
    with pytest.raises(ScoreError, match="contradicts"):
        private_check_required(
            SimpleNamespace(
                task_pins={"declares_hidden_tests": False, "hidden_verify_commitment": "sha256:" + "f" * 64}
            )
        )
    with pytest.raises(ScoreError, match="explicitly declare"):
        private_check_required(SimpleNamespace(task_pins={}))


def test_score_cli_rejects_torn_tail_after_an_otherwise_complete_schedule(admitted, tmp_path, capsys):
    from validator.score import main

    store, _, receipt = admitted
    path = tmp_path / "torn.jsonl"
    evidence = rows(origin=store.identity, digest=receipt.bundle_sha256)
    path.write_text("".join(json.dumps(r) + "\n" for r in evidence) + '{"task_id":')
    output = tmp_path / "card.json"
    assert (
        main(
            [
                "--round",
                "r-1",
                "--miner",
                "alice",
                "--episodes",
                str(path),
                "--store",
                str(store.root),
                "--out",
                str(output),
            ]
        )
        == 2
    )
    assert "truncated episode log" in capsys.readouterr().err
    assert not output.exists()

"""Admitted snapshot authority across real CPU execution and legacy callbacks.

Only completion transport and Git metadata are labelled fixtures. Baseline,
runner, prompt, local tools, sink, admission, scoring and settlement are real.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import competition_support as support
import pytest
import yaml
from test_bundle_bytes import TEXTS, UNICODE, digest, intake_at, runner_setup

from hermes.challenge import from_episode_log
from hermes.profile import compose_system_prompt
from hermes.trajectory import FINAL, TOOL_CALL, Step
from hermesbench import runner
from hermesbench.sink import JsonlEpisodeSink, read_episodes
from hermesbench.tasks import Task
from validator import intake as intake_module
from validator.judge import JudgeError, execution_context, judge_round, runner_for
from validator.judge import main as judge_main
from validator.score import ScoreError, read_metrics
from validator.settlement import SettlementStore
from validator.store import RoundStore


def setup_round(tmp_path, monkeypatch, text="Admitted B\nsecond line\n"):
    files = {"SOUL.md": text, "skills/é-😀/SKILL.md": "\ufeffCheck\r\nthen finish\r"}
    _, tasks, initial, _, _ = runner_setup(tmp_path, monkeypatch, files)
    task = Task.from_record(yaml.safe_load((tasks / "v0/task.yaml").read_text()))
    epoch = {**initial["epoch"], "attempt_ids": [str(i) for i in range(10)], "task_root": str(tasks)}
    monkeypatch.setattr(support, "EPOCH", epoch)
    monkeypatch.setattr(support, "TASK", task.task_id)
    monkeypatch.setattr(support, "VERIFY", "sha256:" + hashlib.sha256(task.verify.encode()).hexdigest())
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture", namespace="bundle-bytes")
    win = support.window(store, private=False)
    attempts = []

    class Policy:
        tokens_used = 200

        def __init__(self, success):
            self.success = success

        def next_steps(self, task, history):
            if history:
                return [Step(kind=FINAL, content="CPU baseline fixture finished.")]
            command = "touch result" if self.success else "touch attempted"
            return [Step(kind=TOOL_CALL, tool="terminal", call_id="fixture", args={"command": command})]

    def policy(task):
        attempts.append(True)
        return Policy(len(attempts) <= 3)

    baseline = tmp_path / "baseline.jsonl"
    with JsonlEpisodeSink(baseline, keep_trajectories=True) as sink:
        runner.run_suite(
            [task],
            policy,
            runner.LocalToolExecutor(allow_unsandboxed=True),
            tmp_path / "baseline-work",
            repeats=10,
            max_concurrency=1,
            sink=sink,
            evaluation_context={"epoch": epoch, "origin": store.identity, "round_id": "r-1", "bundle_sha256": ""},
        )
    opened, refused = from_episode_log(
        read_episodes(baseline),
        epoch=epoch,
        task_pins={task.task_id: {"task_id": task.task_id, "verify": task.verify, "private_check_required": False}},
    )
    assert len(opened) == 1 and not refused
    win.challenge = opened[0]
    store.save(win)
    intake = intake_at(tmp_path)
    intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "Earlier A must never run"}, now=9)
    receipt = intake.accept(round_id="r-1", miner_id="alice", files=files, now=10)
    support.admit_receipt(store, intake, receipt, monkeypatch)
    win = store.load("r-1")
    win.freeze(now=30, reason="CPU handoff fixture")
    store.save(win)
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace="bundle-bytes")
    settlement.activate(store, "r-1", "example/spark")
    context = {"epoch": epoch, "round_id": "r-1", "origin": store.identity}
    clients, calls = [], []

    def factory(**config):
        clients.append(config)
        assert config["base_url"] == "http://127.0.0.1:1"

        def complete(messages, **kwargs):
            calls.append(copy.deepcopy(messages))
            text = (
                '<tool_call>{"name":"terminal","arguments":{"command":"touch result"}}</tool_call>'
                if len(calls) % 2
                else "CPU fixture complete."
            )
            return text, {"prompt_tokens": 20, "completion_tokens": 10}

        return complete

    monkeypatch.setattr("hermesbench.policy.openai_completion", factory)
    kwargs = dict(
        round_id="r-1",
        base_url="http://127.0.0.1:1",
        model="scripted-cpu",
        api_key_env="NONE",
        task_id=task.task_id,
        repeats=10,
        repo_root=None,
        allow_unsandboxed=True,
        task_root=tasks,
        evaluation_context=context,
        dialect="hermes-4",
    )
    return SimpleNamespace(
        root=tmp_path,
        files=files,
        store=store,
        intake=intake,
        receipt=receipt,
        bundle=intake.bundle_dir(receipt).resolve(),
        context=context,
        kwargs=kwargs,
        clients=clients,
        calls=calls,
        settlement=settlement,
    )


def judge(w, run):
    return judge_round(
        round_id="r-1",
        run=run,
        model_revision=w.context["epoch"]["model_revision"],
        store=w.store,
        intake=w.intake,
        workspace=w.root / "episodes/r-1",
        scorecard_dir=w.root / "cards",
        settle=False,
    )[0]


def no_credit(w):
    assert not list((w.root / "cards").glob("*.json"))
    assert not w.store.load("r-1").verdicts and not w.settlement.actions()


@pytest.mark.parametrize("context_mode", ["configured", "inherited"])
@pytest.mark.parametrize(
    "timing",
    [
        "none",
        "before-verify",
        "after-verify",
        "before-context",
        "after-context",
        "changed-capture-restored",
        "after-intact-capture",
    ],
)
def test_admitted_identity_across_every_capture_boundary(tmp_path, monkeypatch, timing, context_mode):
    w = setup_round(tmp_path, monkeypatch)
    soul = w.bundle / "SOUL.md"

    def mutate():
        soul.write_bytes(w.files["SOUL.md"].replace("\n", "\r").encode())

    if timing == "before-verify":
        mutate()
    if timing == "after-verify":
        verify = w.intake.verify

        def boundary_verify(*args, **kwargs):
            result = verify(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(w.intake, "verify", boundary_verify)
    configured = runner_for(**{**w.kwargs, "evaluation_context": w.context if context_mode == "configured" else None})
    contexts, captured = [], []
    actual_main = runner.main
    capture = intake_module.capture_surface

    def main(argv):
        contexts.append(json.loads(Path(argv[argv.index("--evaluation-context") + 1]).read_text()))
        if timing in {"after-context", "changed-capture-restored"}:
            mutate()
        return actual_main(argv)

    def observe_capture(path, **kwargs):
        result = capture(path, **kwargs)
        if kwargs.get("allow_empty"):
            captured.append(dict(result))
        return result

    def compose(base, surface):
        result = compose_system_prompt(base, surface)
        if timing == "changed-capture-restored":
            soul.write_bytes(w.files["SOUL.md"].encode())
        if timing == "after-intact-capture":
            mutate()
        return result

    monkeypatch.setattr(runner, "main", main)
    monkeypatch.setattr(intake_module, "capture_surface", observe_capture)
    monkeypatch.setattr("hermes.profile.compose_system_prompt", compose)

    def compatible_wrapper(miner, path, workspace):
        assert path == w.bundle and miner == "alice"
        bound = execution_context(miner, path)
        assert bound["bundle_sha256"] == w.receipt.bundle_sha256
        # Modifying a returned copy cannot replace inherited admission authority.
        bound["bundle_sha256"] = digest({"SOUL.md": "wrong"})
        if timing == "before-context":
            mutate()
        return configured(miner, path, workspace)

    result = judge(w, compatible_wrapper)
    assert execution_context("alice", w.bundle) is None
    log = w.root / "episodes/r-1/alice.jsonl"
    rows = read_metrics(log) if log.exists() else []
    assert all(c["bundle_sha256"] == w.receipt.bundle_sha256 for c in contexts)
    if timing in {"none", "after-intact-capture"}:
        assert len(w.clients) == 1 and len(w.calls) == 20 and len(rows) == 10
        assert captured == [w.files]
        assert all(w.files["SOUL.md"].strip() in call[0]["content"] for call in w.calls)
        assert all("Earlier A" not in call[0]["content"] for call in w.calls)
        assert all(
            row["bundle_sha256"] == w.receipt.bundle_sha256 and row["tool_calls"] == 1 and row["public_passed"] is True
            for row in rows
        )
        assert len(list((w.root / "episodes/r-1/ws-alice").rglob("result"))) == 10
        if timing == "none":
            assert result.ok and result.scorecard.accepted
        else:
            assert not result.ok
            no_credit(w)
    else:
        assert not result.ok and not w.clients and not w.calls and not rows
        assert not list((w.root / "episodes").rglob("result"))
        no_credit(w)
        if timing == "changed-capture-restored":
            assert captured[0]["SOUL.md"] != w.files["SOUL.md"]
            assert soul.read_bytes() == w.files["SOUL.md"].encode()


@pytest.mark.parametrize("text", TEXTS + UNICODE)
def test_intact_admitted_text_reaches_real_prompt_and_sink(tmp_path, monkeypatch, text):
    w = setup_round(tmp_path, monkeypatch, text)
    result = judge(w, runner_for(**w.kwargs))
    assert result.ok and result.scorecard.accepted
    assert text.strip() in w.calls[0][0]["content"]
    assert w.files["skills/é-😀/SKILL.md"].strip() in w.calls[0][0]["content"]
    assert all(row["bundle_sha256"] == digest(w.files) for row in read_metrics(w.root / "episodes/r-1/alice.jsonl"))


@pytest.mark.parametrize("key", ["bundle_sha256", "round_id", "origin", "epoch"])
def test_conflicting_configured_context_cannot_override_admission(tmp_path, monkeypatch, key):
    w = setup_round(tmp_path, monkeypatch)
    configured = {**w.context, key: "different"}
    result = judge(w, runner_for(**{**w.kwargs, "evaluation_context": configured}))
    assert not result.ok and not w.clients and not w.calls
    no_credit(w)


@pytest.mark.parametrize("missing", [None, "", "invalid", 123])
def test_unbound_explicit_context_requires_expected_commitment(tmp_path, monkeypatch, missing):
    w = setup_round(tmp_path, monkeypatch)
    context = {**w.context}
    if missing is not None:
        context["bundle_sha256"] = missing
    run = runner_for(**{**w.kwargs, "evaluation_context": context})
    with pytest.raises(JudgeError, match="expected bundle commitment"):
        run("alice", w.bundle, tmp_path)
    assert not w.clients and not w.calls


def test_unbound_context_preserves_expected_and_exploration_remains_explicit(tmp_path, monkeypatch):
    w = setup_round(tmp_path, monkeypatch)
    context = {**w.context, "bundle_sha256": w.receipt.bundle_sha256}
    run = runner_for(**{**w.kwargs, "evaluation_context": context})
    (w.bundle / "SOUL.md").write_bytes(b"Changed after context configuration")
    # Caller mutation after adapter construction cannot silently change its pin.
    context["bundle_sha256"] = digest({**w.files, "SOUL.md": "Changed after context configuration"})
    with pytest.raises(ScoreError, match="bundle does not match"):
        run("alice", w.bundle, tmp_path)
    assert not w.clients and not w.calls
    exploratory = runner_for(**{**w.kwargs, "evaluation_context": None})
    log = exploratory("alice", w.bundle, tmp_path)
    assert len(w.calls) == 20
    assert "bundle_sha256" not in read_metrics(log)[0]
    assert "Changed after context configuration" in w.calls[0][0]["content"]


@pytest.mark.parametrize("adapter", ["recording", "replay"])
def test_legacy_callbacks_refuse_post_verify_changes(tmp_path, monkeypatch, adapter):
    w = setup_round(tmp_path, monkeypatch)
    real_verify = intake_module.Intake.verify
    calls = []

    def verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        (w.bundle / "SOUL.md").write_bytes(b"Changed before callback")
        return result

    monkeypatch.setattr(intake_module.Intake, "verify", verify)

    def run(miner, path, workspace):
        calls.append(adapter)
        return tmp_path / "replay.jsonl"

    if adapter == "recording":
        result = judge(w, run)
        assert not result.ok and not calls
    else:
        source = tmp_path / "replay/r-1/alice.jsonl"
        source.parent.mkdir(parents=True)
        source.write_bytes((tmp_path / "baseline.jsonl").read_bytes())
        assert (
            judge_main(
                [
                    "judge",
                    "--round",
                    "r-1",
                    "--store",
                    str(w.store.root),
                    "--intake-root",
                    str(w.intake.root),
                    "--receipts",
                    str(w.intake.receipts),
                    "--workspace",
                    str(tmp_path / "episodes"),
                    "--scorecards",
                    str(tmp_path / "cards"),
                    "--fixture-episodes",
                    str(tmp_path / "replay"),
                    "--no-settle",
                ]
            )
            == 1
        )
        # This real replay route would copy a present source if reached, even if
        # its baseline rows subsequently refused scoring. Refusal must precede it.
        assert not list((tmp_path / "episodes").rglob("*.jsonl"))
    assert not w.clients and not w.calls
    no_credit(w)

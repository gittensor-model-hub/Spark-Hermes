"""Exact UTF-8 artifacts through real upload, runner and miner consumers (CPU fixtures)."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from competition_support import EPOCH, admit_receipt, rows, window
from settlement_support import cli

from hermes.base_model import load as model_pin
from hermes.profile import compose_system_prompt
from hermesbench import runner
from hermesbench.tasks import Task
from miner.evaluate import runner_argv
from miner.search import Candidate, SearchError, ablations, read_surface
from validator.intake import Intake, IntakeError, bundle_digest, capture_surface
from validator.judge import judge_round, runner_for
from validator.score import ScoreError, read_metrics
from validator.settlement import SettlementStore
from validator.store import RoundStore

TEXTS = ["one\ntwo\n", "one\r\ntwo\r\n", "one\rtwo\r", "one\r\ntwo\rthree\nfour", "one\ntwo"]
UNICODE = ["Ω 😀", "e\u0301", "é", "\ufeffBOM"]


def digest(files):
    # Independent oracle: intentionally do not use product canonical/digest helpers.
    raw = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def put(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


def intake_at(root):
    return Intake(root / "bundles", root / "receipts.jsonl", mode="fixture", namespace="bundle-bytes")


@pytest.mark.parametrize("text", TEXTS + UNICODE)
def test_intact_upload_reopen_retry_and_full_candidate(tmp_path, text):
    files = {"SOUL.md": text, "skills/策略-😀-e\u0301/SKILL.md": " 1. Rule\r\n2. Check\r "}
    intake = intake_at(tmp_path)
    receipt = intake.accept(round_id="r-1", miner_id="alice", files=files, now=10)
    original_receipts = intake.receipts.read_bytes()
    assert receipt.bundle_sha256 == digest(files)
    assert receipt.bytes == sum(len(value.encode("utf-8")) for value in files.values())
    assert receipt.files == 2
    reopened = intake_at(tmp_path)
    source = reopened.verify(receipt)
    assert {name: (source / name).read_bytes() for name in files} == {k: v.encode("utf-8") for k, v in files.items()}
    assert reopened.accept(round_id="r-1", miner_id="alice", files=files, now=99) == receipt
    assert intake.receipts.read_bytes() == original_receipts
    full = ablations(read_surface(source))[0]
    copied = full.write(tmp_path / "candidate")
    assert capture_surface(copied) == files
    assert {name: (copied / name).read_bytes() for name in files} == {k: v.encode("utf-8") for k, v in files.items()}
    assert bundle_digest(read_surface(copied)) == receipt.bundle_sha256


def test_variants_have_independent_digests_and_byte_counts(tmp_path):
    intake = intake_at(tmp_path)
    maps = [{"SOUL.md": text} for text in TEXTS + UNICODE]
    maps += [{f"skills/{name}/SKILL.md": "same"} for name in ["é", "e\u0301", "😀", "\ufeffname"]]
    receipts = [intake.accept(round_id="r-1", miner_id="alice", files=f) for f in maps]
    assert len({r.bundle_sha256 for r in receipts}) == len(maps)
    assert [r.bundle_sha256 for r in receipts] == [digest(f) for f in maps]
    assert [r.bytes for r in receipts[:5]] == [8, 10, 8, 19, 7]
    for receipt, files in zip(receipts, maps):
        assert receipt.bytes == sum(len(v.encode("utf-8")) for v in files.values())
        assert capture_surface(intake.verify(receipt)) == files


@pytest.mark.parametrize(
    "damage",
    [
        "CR",
        "add",
        "remove",
        "rename",
        "symlink",
        "directory-symlink",
        "root-symlink",
        "bom-add",
        "bom-remove",
        "unicode",
        "invalid-utf8",
        "invalid-name",
    ],
)
def test_actual_files_override_sidecar_and_refuse_mutations(tmp_path, damage):
    intake = intake_at(tmp_path)
    files = {"SOUL.md": "one\ntwo\n", "skills/é/SKILL.md": "\ufeffRule"}
    receipt = intake.accept(round_id="r-1", miner_id="alice", files=files)
    root = intake.bundle_dir(receipt)
    sidecar = intake.canonical_path_for(receipt).read_bytes()
    if damage == "CR":
        (root / "SOUL.md").write_bytes(b"one\rtwo\r")
    elif damage == "add":
        put(root, {"skills/new/SKILL.md": "new"})
    elif damage == "remove":
        (root / "SOUL.md").unlink()
    elif damage == "rename":
        (root / "skills/é").rename(root / "skills/e\u0301")
    elif damage == "symlink":
        (root / "SOUL.md").unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(b"one\ntwo\n")
        (root / "SOUL.md").symlink_to(outside)
    elif damage == "directory-symlink":
        (root / "skills").rename(tmp_path / "skills")
        (root / "skills").symlink_to(tmp_path / "skills", target_is_directory=True)
    elif damage == "root-symlink":
        root.rename(tmp_path / "moved")
        root.symlink_to(tmp_path / "moved", target_is_directory=True)
    elif damage == "bom-add":
        (root / "SOUL.md").write_bytes(b"\xef\xbb\xbfone\ntwo\n")
    elif damage == "bom-remove":
        (root / "skills/é/SKILL.md").write_bytes(b"Rule")
    elif damage == "unicode":
        (root / "skills/é/SKILL.md").write_bytes("\ufeffRulé".encode())
    elif damage == "invalid-utf8":
        (root / "SOUL.md").write_bytes(b"one\xfftwo\n")
    else:
        (root / "skills/é").rename(root / "skills/\udcff")
    with pytest.raises(IntakeError):
        intake.verify(receipt)
    with pytest.raises(IntakeError):
        intake.accept(round_id="r-1", miner_id="alice", files=files)
    assert intake.canonical_path_for(receipt).read_bytes() == sidecar
    assert intake.read_receipts() == [receipt]


@pytest.mark.parametrize("files", [{"SOUL.md": "bad\ud800"}, {"skills/\udfff/SKILL.md": "bad"}])
def test_surrogate_upload_rejected_before_any_publication(tmp_path, files):
    intake = intake_at(tmp_path)
    with pytest.raises(IntakeError, match="UTF-8"):
        intake.accept(round_id="r-1", miner_id="alice", files=files)
    assert not intake.root.exists() and not intake.receipts.exists()
    prior = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "valid\r\n"})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(IntakeError, match="UTF-8"):
        intake.accept(round_id="r-1", miner_id="alice", files=files)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert intake.verify(prior)
    with pytest.raises(SearchError, match="UTF-8"):
        Candidate("full", files).write(tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_concurrent_newline_uploads_restart_and_retry(tmp_path):
    intake = intake_at(tmp_path)

    def upload(i):
        return intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": TEXTS[i % 5]}, now=i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(upload, range(40)))
    reopened = intake_at(tmp_path)
    assert len(reopened.read_receipts()) == 5
    for i, receipt in enumerate(results):
        assert receipt.bundle_sha256 == digest({"SOUL.md": TEXTS[i % 5]})
        assert capture_surface(reopened.verify(receipt)) == {"SOUL.md": TEXTS[i % 5]}
        assert reopened.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": TEXTS[i % 5]}, now=999) == receipt


def test_newline_publication_failure_recovers_without_rewriting_bytes(tmp_path, monkeypatch):
    intake = intake_at(tmp_path)
    files = {"SOUL.md": "\ufeffline\r\nsecond\rthird"}

    def fail(_):
        raise OSError("injected receipt publication failure")

    monkeypatch.setattr(intake, "_write_receipts", fail)
    with pytest.raises(OSError, match="injected"):
        intake.accept(round_id="r-1", miner_id="alice", files=files, now=10)
    assert not intake.receipts.exists()
    stored = next(intake.root.rglob("SOUL.md"))
    stat = stored.stat()
    reopened = intake_at(tmp_path)
    receipt = reopened.accept(round_id="r-1", miner_id="alice", files=files, now=11)
    assert stored.read_bytes() == files["SOUL.md"].encode()
    assert stored.stat().st_ino == stat.st_ino and stored.stat().st_mtime_ns == stat.st_mtime_ns
    assert reopened.verify(receipt) and not list(intake.root.rglob(".upload-*"))


def runner_setup(tmp_path, monkeypatch, files):
    # Only the Git metadata source and completion transport are CPU adapters.
    # Keep observed_harness, bundle identity, prompt composition and execution real.
    # This source-bound fixture identifier is never a production commit/approval.
    from hermes import profile
    from validator import intake as intake_module

    fixture_revision = hashlib.sha256(
        b"CPU fixture source identity\x00"
        + b"".join(Path(module.__file__).read_bytes() for module in (runner, profile, intake_module))
    ).hexdigest()[:40]
    monkeypatch.setattr("hermes.pin.git_commit", lambda root=None: fixture_revision)
    for name in ["SPARKDISTILL_WITHHELD_ROOT", "HERMESBENCH_WITHHELD_SALT"]:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "surface"
    put(root, files)
    task = Task.from_record(
        dict(
            task_id="bundle-byte-cpu",
            prompt="Create result.",
            tools=["terminal"],
            verify="test -f result",
            max_steps=5,
            timeout_s=5,
        )
    )
    tasks = tmp_path / "tasks"
    (tasks / "v0").mkdir(parents=True)
    (tasks / "v0/task.yaml").write_text(
        yaml.safe_dump(
            dict(
                task_id=task.task_id,
                prompt=task.prompt,
                tools=list(task.tools),
                verify=task.verify,
                max_steps=5,
                timeout_s=5,
            )
        )
    )
    context = dict(
        epoch={
            **EPOCH,
            "model_revision": model_pin().revision,
            "harness_digest": runner.observed_harness(
                [task], suite_name="all", salt="", executor="local", tool_timeout_s=120
            ),
            "attempt_ids": ["0"],
        },
        origin={"mode": "fixture", "namespace": "byte-cpu", "issuer": "scripted-cpu"},
        round_id="r-1",
        bundle_sha256=digest(files),
    )
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    argv = runner_argv(
        task_id=task.task_id,
        base_url="http://127.0.0.1:1",
        model="cpu-fixture",
        api_key_env="NONE",
        workspace_root=tmp_path / "ws",
        episodes_out=tmp_path / "episodes.jsonl",
        repeats=1,
        miner_dir=root,
        allow_unsandboxed=True,
        task_root=tasks,
        evaluation_context=path,
        dialect="hermes-4",
    )
    messages = []

    def factory(**kwargs):
        def complete(turns, **kwargs):
            messages.append(turns)
            text = (
                '<tool_call>{"name":"terminal","arguments":{"command":"touch result"}}</tool_call>'
                if len(messages) % 2
                else "Fixture done."
            )
            return text, {"prompt_tokens": 20, "completion_tokens": 10}

        return complete

    monkeypatch.setattr("hermesbench.policy.openai_completion", factory)
    return root, tasks, context, argv, messages


@pytest.mark.parametrize("text", TEXTS + UNICODE)
def test_genuine_runner_context_composition_and_episode(tmp_path, monkeypatch, text):
    files = {"SOUL.md": text, "skills/é-😀/SKILL.md": " \ufeff1. Check\r\n2. Finish\r "}
    root, tasks, context, _, messages = runner_setup(tmp_path, monkeypatch, files)
    work = tmp_path / "job"
    work.mkdir()
    run = runner_for(
        round_id="r-1",
        base_url="http://127.0.0.1:1",
        model="cpu-fixture",
        api_key_env="NONE",
        task_id="bundle-byte-cpu",
        repeats=1,
        repo_root=None,
        allow_unsandboxed=True,
        task_root=tasks,
        evaluation_context=context,
        dialect="hermes-4",
    )
    log = run("alice", root, work)
    assert json.loads((work / "alice.context.json").read_text())["bundle_sha256"] == digest(files)
    expected = (
        text.strip()
        + "\n\n"
        + (runner.HARNESS_DIR / "system_prompt.txt").read_text().strip()
        + "\n\n# Strategy: é-😀\n\n"
        + files["skills/é-😀/SKILL.md"].strip()
    )
    assert expected in messages[0][0]["content"]
    row = read_metrics(log)[0]
    assert row["bundle_sha256"] == digest(files) and row["public_passed"] is True
    assert row["tokens_used"] == 60 and row["tool_calls"] == 1
    assert row["origin"]["mode"] == "fixture"


@pytest.mark.parametrize("restore", [False, True])
def test_runner_hashes_the_surface_it_composed_not_a_later_reread(tmp_path, monkeypatch, restore):
    original = {"SOUL.md": "one\ntwo\n"}
    changed = {"SOUL.md": "one\rtwo\r"}
    root, _, _, argv, messages = runner_setup(tmp_path, monkeypatch, original)
    if restore:
        put(root, changed)

    def compose(base, surface):
        # Move disk bytes between composition and identity check. No oracle changes.
        result = compose_system_prompt(base, surface)
        put(root, original if restore else changed)
        return result

    monkeypatch.setattr("hermes.profile.compose_system_prompt", compose)
    if restore:
        with pytest.raises(ScoreError, match="bundle does not match"):
            runner.main(argv)
        assert messages == [] and not (tmp_path / "episodes.jsonl").exists()
    else:
        assert runner.main(argv) == 0
        assert "one\ntwo" in messages[0][0]["content"] and "one\rtwo" not in messages[0][0]["content"]
        assert read_metrics(tmp_path / "episodes.jsonl")[0]["bundle_sha256"] == digest(original)


def test_invalid_utf8_refuses_actual_runner_and_miner_before_execution(tmp_path, monkeypatch):
    root, _, _, argv, messages = runner_setup(tmp_path, monkeypatch, {"SOUL.md": "good"})
    (root / "SOUL.md").write_bytes(b"bad\xff")
    assert runner.main(argv) == 2
    assert messages == [] and not (tmp_path / "episodes.jsonl").exists()
    with pytest.raises(SearchError, match="UTF-8"):
        read_surface(root)


@pytest.mark.parametrize("route", ["recording", "cli"])
@pytest.mark.parametrize("damaged", [b"one\rtwo\r", b"one\xfftwo\n"])
def test_invalid_admitted_bytes_never_reach_execution_card_verdict_or_outbox(tmp_path, monkeypatch, route, damaged):
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture", namespace="bundle-bytes")
    window(store)
    intake = intake_at(tmp_path)
    intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "Bundle A"}, now=9)
    receipt = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "one\ntwo\n"}, now=10)
    admit_receipt(store, intake, receipt, monkeypatch)
    (intake.bundle_dir(receipt) / "SOUL.md").write_bytes(damaged)
    win = store.load("r-1")
    win.freeze(now=30, reason="CPU byte fixture")
    store.save(win)
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace="bundle-bytes")
    settlement.activate(store, "r-1", "example/spark")
    source = tmp_path / "input/r-1/alice.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(r) + "\n" for r in rows(origin=store.identity, digest=receipt.bundle_sha256)))
    if route == "recording":
        called = []

        def run(*args):
            called.append(args)
            pytest.fail("invalid bytes reached runner")

        result = judge_round(
            round_id="r-1",
            run=run,
            model_revision=EPOCH["model_revision"],
            store=store,
            intake=intake,
            workspace=tmp_path / "episodes",
            scorecard_dir=tmp_path / "cards",
            settle=False,
        )
        assert not result[0].ok and called == []
    else:
        cli(
            tmp_path,
            "validator.judge",
            "judge",
            "--round",
            "r-1",
            "--store",
            store.root,
            "--intake-root",
            intake.root,
            "--receipts",
            intake.receipts,
            "--workspace",
            tmp_path / "episodes",
            "--scorecards",
            tmp_path / "cards",
            "--fixture-episodes",
            tmp_path / "input",
            "--no-settle",
            expected=1,
        )
        cli(
            tmp_path,
            "validator.crown",
            "select",
            "--store",
            store.root,
            "--settlement-root",
            settlement.root,
            "--scorecards",
            tmp_path / "cards",
            "--episodes",
            tmp_path / "episodes",
            expected=2,
        )
    assert not list((tmp_path / "cards").glob("*.json"))
    assert not list((tmp_path / "episodes").rglob("*.jsonl"))
    assert store.load("r-1").snapshot()["verdicts"] == []
    assert settlement.actions() == []


@pytest.mark.parametrize(
    "files",
    [
        {"SOUL.md": "bad\ud800"},
        {"skills/\udfff/SKILL.md": "bad"},
        {"SOUL.md": "\ufeffΩ\r\n😀\r", "skills/e\u0301/SKILL.md": "check\r\n"},
    ],
)
def test_upload_api_preserves_valid_text_and_reports_unencodable_text(tmp_path, monkeypatch, files):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from validator import api
    from validator import intake as intake_module

    store = RoundStore(tmp_path / "rounds", mode="fixture", namespace="bundle-bytes")
    win = window(store)
    monkeypatch.setattr(api, "ROUNDS", {win.round_id: win})
    monkeypatch.setattr(intake_module, "SUBMISSION_DIR", tmp_path / "api-bundles")
    monkeypatch.setattr(intake_module, "RECEIPTS", tmp_path / "api-receipts.jsonl")
    # Send original escaped JSON, including unpaired surrogate escapes.
    request = json.dumps({"miner_id": "alice", "files": files}, ensure_ascii=True).encode()
    response = TestClient(api.app).post(
        "/v1/round/r-1/submission", content=request, headers={"content-type": "application/json"}
    )
    intake = Intake()
    invalid = any("\ud800" in v or "\udfff" in k for k, v in files.items())
    assert response.status_code == (400 if invalid else 200)
    if invalid:
        assert "UTF-8" in response.json()["detail"]
        assert not intake.root.exists() and not intake.receipts.exists()
    else:
        receipt = intake.read_receipts()[0]
        assert response.json()["bundle_sha256"] == digest(files)
        assert capture_surface(intake.verify(receipt)) == files

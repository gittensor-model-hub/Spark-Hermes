"""CPU proofs for the final expected identity, independent of generated transport.

Only scripted completion and source-derived fixture Git metadata replace external
prerequisites. The runner, policy, tools, sink, judge and identity checks are real.
"""

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from test_bundle_bytes import digest
from test_execution_handoff import judge, no_credit, setup_round

from hermesbench import runner
from hermesbench.execution import expected_execution
from validator import intake as intake_module
from validator.judge import execution_context, runner_for
from validator.score import read_metrics


def adapter(w, mode):
    configured = runner_for(**{**w.kwargs, "evaluation_context": w.context if mode == "configured" else None})

    def run(miner, path, workspace):
        if mode != "thread":
            return configured(miner, path, workspace)
        forwarded = execution_context(miner, path)
        threaded = runner_for(**{**w.kwargs, "evaluation_context": forwarded})
        # The adapter owns its snapshot even if the caller later edits its copy.
        forwarded.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(threaded, miner, path, workspace).result()

    return run


def assert_refused(w, result):
    assert not result.ok, result
    assert not w.clients and not w.calls
    assert not list((w.root / "episodes").rglob("result"))
    assert not list((w.root / "episodes").rglob("*.jsonl"))
    no_credit(w)
    assert expected_execution() is None
    assert execution_context("alice", w.bundle) is None


FIELDS = [
    ("bundle_sha256",),
    ("round_id",),
    ("task_id",),
    ("origin", "mode"),
    ("origin", "namespace"),
    ("origin", "issuer"),
    ("epoch", "epoch_id"),
    ("epoch", "task_root"),
    ("epoch", "model_revision"),
    ("epoch", "harness_digest"),
    ("epoch", "attempt_ids"),
    ("epoch", "score_policy"),
]


@pytest.mark.parametrize("mode", ["inherited", "thread"])
@pytest.mark.parametrize("field", FIELDS)
def test_full_transport_identity_before_completion(tmp_path, monkeypatch, mode, field):
    w = setup_round(tmp_path, monkeypatch)
    actual_main = runner.main

    def changed(argv):
        path = Path(argv[argv.index("--evaluation-context") + 1])
        context = json.loads(path.read_bytes())
        target = context
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = "changed identity"
        path.write_text(json.dumps(context))
        return actual_main(argv)

    monkeypatch.setattr(runner, "main", changed)
    result = judge(w, adapter(w, mode))
    assert "transport differs" in result.problem
    assert_refused(w, result)


@pytest.mark.parametrize(
    "change",
    [
        "missing-file",
        "missing-flag",
        "missing-miner",
        "other-directory",
        "other-transport",
        "symlink",
        "empty",
        "array",
        "null",
        "truncated",
        "trailing",
        "duplicate",
        "escaped-duplicate",
        "nested-duplicate",
        "nan",
        "infinity",
        "overflow",
        "utf8",
        "bom",
        "removed-epoch",
        "removed-origin",
        "removed-task",
        "removed-round",
        "removed-digest",
        "bool-number",
    ],
)
def test_missing_replaced_or_ambiguous_transport_refuses(tmp_path, monkeypatch, change):
    w = setup_round(tmp_path, monkeypatch)
    actual_main = runner.main

    def changed(argv):
        index = argv.index("--evaluation-context")
        path = Path(argv[index + 1])
        raw = path.read_bytes()
        if change == "missing-file":
            path.unlink()
        elif change == "missing-flag":
            del argv[index : index + 2]
        elif change == "missing-miner":
            index = argv.index("--miner-dir")
            del argv[index : index + 2]
        elif change == "other-directory":
            other = tmp_path / "same-bytes-other-directory"
            shutil.copytree(w.bundle, other)
            argv[argv.index("--miner-dir") + 1] = str(other)
        elif change in {"other-transport", "symlink"}:
            other = path.with_suffix(".replacement")
            other.write_bytes(raw)
            if change == "symlink":
                path.unlink()
                path.symlink_to(other)
            else:
                argv[index + 1] = str(other)
        elif change.startswith("removed-"):
            key = {"task": "task_id", "round": "round_id", "digest": "bundle_sha256"}.get(change[8:], change[8:])
            context = json.loads(raw)
            del context[key]
            path.write_text(json.dumps(context))
        elif change == "bool-number":
            context = json.loads(raw)
            # Python would consider True == 1. The resolved context includes this
            # configured extension; its exact type is still part of the snapshot.
            context["extension"] = True
            path.write_text(json.dumps(context))
        else:
            malformed = {
                "empty": b"",
                "array": b"[]",
                "null": b"null",
                "truncated": raw[:-1],
                "trailing": raw + b"{}",
                "duplicate": b'{"round_id":"wrong",' + raw[1:],
                "escaped-duplicate": b'{"round_\\u0069d":"wrong",' + raw[1:],
                "nested-duplicate": raw.replace(b'"epoch": {', b'"epoch": {"epoch_id":"wrong",', 1),
                "nan": b'{"extra":NaN,' + raw[1:],
                "infinity": b'{"extra":Infinity,' + raw[1:],
                "overflow": b'{"extra":1e999,' + raw[1:],
                "utf8": raw + b"\xff",
                "bom": b"\xef\xbb\xbf" + raw,
            }
            path.write_bytes(malformed[change])
        return actual_main(argv)

    monkeypatch.setattr(runner, "main", changed)
    w.context["extension"] = 1
    assert_refused(w, judge(w, adapter(w, "configured")))


@pytest.mark.parametrize("mode", ["inherited", "configured", "thread"])
@pytest.mark.parametrize("restore", [False, True])
def test_bundle_and_context_rebound_at_final_capture(tmp_path, monkeypatch, mode, restore):
    w = setup_round(tmp_path, monkeypatch)
    capture = intake_module.capture_surface
    seen = []

    def changed(path, **kwargs):
        if kwargs.get("allow_empty"):
            (path / "SOUL.md").write_bytes(w.files["SOUL.md"].replace("\n", "\r").encode())
            ctx = w.root / "episodes/r-1/alice.context.json"
            context = json.loads(ctx.read_bytes())
            context["bundle_sha256"] = digest({**w.files, "SOUL.md": w.files["SOUL.md"].replace("\n", "\r")})
            ctx.write_text(json.dumps(context))
        result = capture(path, **kwargs)
        if kwargs.get("allow_empty"):
            seen.append(dict(result))
            if restore:
                (path / "SOUL.md").write_bytes(w.files["SOUL.md"].encode())
        return result

    monkeypatch.setattr(intake_module, "capture_surface", changed)
    assert_refused(w, judge(w, adapter(w, mode)))
    assert len(seen) == 1 and seen[0] != w.files
    if restore:
        assert (w.bundle / "SOUL.md").read_bytes() == w.files["SOUL.md"].encode()


@pytest.mark.parametrize("mode", ["inherited", "thread"])
@pytest.mark.parametrize("change_bundle", [False, True])
def test_one_checked_snapshot_survives_later_disk_change(tmp_path, monkeypatch, mode, change_bundle):
    w = setup_round(tmp_path, monkeypatch)
    read = Path.read_bytes
    captured = []

    def then_change(path):
        raw = read(path)
        if path.name == "alice.context.json":
            captured.append(json.loads(raw))
            path.write_bytes(b'{"epoch": "changed after snapshot"}')
            if change_bundle:
                (w.bundle / "SOUL.md").write_bytes(b"Changed after immutable capture")
        return raw

    monkeypatch.setattr(Path, "read_bytes", then_change)
    result = judge(w, adapter(w, mode))
    assert len(captured) == 1
    assert len(w.clients) == 1 and len(w.calls) == 20
    assert all(w.files["SOUL.md"].strip() in call[0]["content"] for call in w.calls)
    rows = read_metrics(w.root / "episodes/r-1/alice.jsonl")
    assert len(rows) == 10
    for row in rows:
        assert row["bundle_sha256"] == captured[0]["bundle_sha256"] == w.receipt.bundle_sha256
        assert row["epoch_id"] == captured[0]["epoch"]["epoch_id"]
        assert row["origin"] == captured[0]["origin"] and row["tool_calls"] == 1
    if change_bundle:
        assert not result.ok
        no_credit(w)
    else:
        assert result.ok and result.scorecard.accepted
    assert expected_execution() is None


def test_concurrent_final_scopes_exception_cleanup_and_genuine_execution(tmp_path, monkeypatch):
    w = setup_round(tmp_path, monkeypatch)
    actual_main = runner.main
    barrier = Barrier(2)
    contexts = {}
    observations = []
    # These standalone contexts are operator inputs, not claims of admission.
    for name in ["left", "right"]:
        path = tmp_path / name
        shutil.copytree(w.bundle, path / "bundle")
        context = {**copy.deepcopy(w.context), "bundle_sha256": w.receipt.bundle_sha256, "task_id": w.kwargs["task_id"]}
        context["origin"]["namespace"] = name
        contexts[name] = context

    def interleaved(argv):
        expected = expected_execution()
        name = expected.bundle_path.parent.name
        barrier.wait(timeout=10)
        assert json.loads(expected.context_json) == contexts[name]
        assert expected_execution() is expected
        if name == "left":
            raise RuntimeError("controlled final-consumer exception")
        return actual_main(argv)

    monkeypatch.setattr(runner, "main", interleaved)

    def work(name):
        run = runner_for(**{**w.kwargs, "evaluation_context": contexts[name]})
        try:
            return run("alice", tmp_path / name / "bundle", tmp_path / name)
        finally:
            observations.append((name, expected_execution(), execution_context("alice", w.bundle)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.submit(work, "left"), pool.submit(work, "right")
        with pytest.raises(RuntimeError, match="controlled final-consumer"):
            left.result()
        log = right.result()
    assert len(w.calls) == 20
    assert all(row["origin"]["namespace"] == "right" and row["tool_calls"] == 1 for row in read_metrics(log))
    assert sorted(observations) == [("left", None, None), ("right", None, None)]
    assert expected_execution() is None


@pytest.mark.parametrize("raw", [b"[]", b"null", b'{"epoch":{},"epoch":{}}', b'{"extra":NaN}', b'{"extra":1e999}'])
def test_standalone_operator_context_also_decodes_strictly(tmp_path, monkeypatch, raw):
    from miner.evaluate import runner_argv
    from validator.score import ScoreError

    w = setup_round(tmp_path, monkeypatch)
    path = tmp_path / "operator.json"
    path.write_bytes(raw)
    argv = runner_argv(
        task_id=w.kwargs["task_id"],
        base_url=w.kwargs["base_url"],
        model=w.kwargs["model"],
        api_key_env="NONE",
        workspace_root=tmp_path / "operator-work",
        episodes_out=tmp_path / "operator.jsonl",
        repeats=10,
        miner_dir=w.bundle,
        task_root=w.kwargs["task_root"],
        evaluation_context=path,
        allow_unsandboxed=True,
    )
    with pytest.raises(ScoreError, match="invalid evaluation context"):
        runner.main(argv)
    assert not w.clients and not w.calls

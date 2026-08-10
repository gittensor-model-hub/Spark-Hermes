import json

import eval.record_training_ledger as cli


def test_main_writes_ledger_entry(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": "run-1",
    }
    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (report, None, None, None))

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")
    ledger_path = tmp_path / "ledger.jsonl"

    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(ledger_path),
        ]
    )
    assert rc == 0
    entry = json.loads(ledger_path.read_text().splitlines()[0])
    assert entry["run_id"] == "run-1"


def test_main_returns_nonzero_on_issue(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (None, None, "download failed", None))

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")

    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(tmp_path / "ledger.jsonl"),
        ]
    )
    assert rc == 1


def _ok_report(run_id="run-x"):
    return {
        "verified": True,
        "label": "eval:XS",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": run_id,
    }


def test_main_publishes_runs_when_flag_set(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (_ok_report("run-2"), None, None, None))

    captured: dict = {}

    def fake_publish(pathspec, **kwargs):
        captured["pathspec"] = pathspec
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "publish_paths_to_main", fake_publish)

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")
    ledger_path = tmp_path / "runs" / "ledger.jsonl"

    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/288",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(ledger_path),
            "--publish",
        ]
    )
    assert rc == 0
    assert captured["pathspec"] == (tmp_path / "runs").as_posix()
    assert captured["pr_branch"] == "chore/ledger-pr-288"
    assert "#288" in captured["commit_message"]


def test_main_does_not_publish_without_flag(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (_ok_report("run-3"), None, None, None))
    calls = {"n": 0}
    monkeypatch.setattr(cli, "publish_paths_to_main", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or [])

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")
    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(tmp_path / "runs" / "ledger.jsonl"),
        ]
    )
    assert rc == 0
    assert calls["n"] == 0


def test_main_skips_publish_when_record_fails(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (None, None, "download failed", None))
    calls = {"n": 0}
    monkeypatch.setattr(cli, "publish_paths_to_main", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or [])

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")
    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(tmp_path / "runs" / "ledger.jsonl"),
            "--publish",
        ]
    )
    assert rc == 1
    assert calls["n"] == 0

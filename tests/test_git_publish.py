"""Tests for eval.git_publish.publish_paths_to_main."""

from types import SimpleNamespace

import eval.git_publish as gp


def _fake_run(responses):
    """Build a subprocess.run stub. `responses` is a list of (matcher, (rc, out, err))."""
    calls: list[list[str]] = []

    def run(args, capture_output=True, text=True, check=False):
        calls.append(args)
        for matcher, (rc, out, err) in responses:
            if matcher(args):
                return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run, calls


def _is(*prefix):
    return lambda args: args[: len(prefix)] == list(prefix)


def test_publish_no_changes_is_noop(monkeypatch):
    run, calls = _fake_run([(_is("git", "status"), (0, "", ""))])
    monkeypatch.setattr(gp.subprocess, "run", run)
    assert gp.publish_paths_to_main("runs", commit_message="m", pr_branch="b", pr_title="t", pr_body="body") == []
    assert not any(a[:2] == ["git", "add"] for a in calls)


def test_publish_direct_push_success(monkeypatch):
    run, calls = _fake_run(
        [
            (_is("git", "status"), (0, " M runs/ledger.jsonl\n", "")),
            (lambda a: a[:2] == ["git", "push"] and "HEAD:main" in a, (0, "", "")),
        ]
    )
    monkeypatch.setattr(gp.subprocess, "run", run)
    assert gp.publish_paths_to_main("runs", commit_message="m", pr_branch="b", pr_title="t", pr_body="body") == []
    assert ["git", "push", "origin", "HEAD:main"] in calls
    assert not any(a[0] == "gh" for a in calls)  # no PR needed


def test_publish_pr_fallback_on_branch_protection(monkeypatch):
    run, calls = _fake_run(
        [
            (_is("git", "status"), (0, " M runs/ledger.jsonl\n", "")),
            (lambda a: a[:2] == ["git", "push"] and "HEAD:main" in a, (1, "", "remote: error: GH013 ... pull request")),
            (lambda a: a[:2] == ["git", "push"] and "-u" in a, (0, "", "")),
            (_is("gh", "pr", "create"), (0, "https://github.com/o/r/pull/999\n", "")),
            (_is("gh", "pr", "merge"), (0, "", "")),
        ]
    )
    monkeypatch.setattr(gp.subprocess, "run", run)
    assert gp.publish_paths_to_main("runs", commit_message="m", pr_branch="chore/x", pr_title="t", pr_body="body") == []
    assert ["gh", "pr", "create", "--title", "t", "--body", "body", "--base", "main", "--head", "chore/x"] in calls
    assert ["gh", "pr", "merge", "999", "--squash", "--delete-branch"] in calls


def test_publish_non_protection_push_error_no_pr(monkeypatch):
    run, calls = _fake_run(
        [
            (_is("git", "status"), (0, " M runs/ledger.jsonl\n", "")),
            (lambda a: a[:2] == ["git", "push"], (1, "", "network unreachable")),
        ]
    )
    monkeypatch.setattr(gp.subprocess, "run", run)
    issues = gp.publish_paths_to_main("runs", commit_message="m", pr_branch="b", pr_title="t", pr_body="body")
    assert issues == ["network unreachable"]
    assert not any(a[0] == "gh" for a in calls)  # a non-protection error must not open a PR

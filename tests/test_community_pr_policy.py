"""Tests for eval.community_pr_policy."""

from types import SimpleNamespace

import eval.community_pr_policy as policy

_TRAINING_BODY = "- [x] **Training/evaluation improvement**"
_DATASET_BODY = "- [x] **Dataset track submission**"


def test_is_trusted_author():
    assert policy.is_trusted_author("OWNER", "jane", "User")
    assert policy.is_trusted_author("member", "jane", "User")  # case-insensitive
    assert policy.is_trusted_author("COLLABORATOR", "jane", "User")
    # bots are trusted by login suffix or type
    assert policy.is_trusted_author("NONE", "dependabot[bot]", "Bot")
    assert policy.is_trusted_author("CONTRIBUTOR", "renovate", "Bot")
    # community humans are not
    assert not policy.is_trusted_author("CONTRIBUTOR", "randodev", "User")
    assert not policy.is_trusted_author("FIRST_TIME_CONTRIBUTOR", "randodev", "User")
    assert not policy.is_trusted_author("NONE", "randodev", "User")


def test_is_optimization_pr():
    assert policy.is_optimization_pr(_TRAINING_BODY, None)
    assert policy.is_optimization_pr(_DATASET_BODY, None)
    assert policy.is_optimization_pr(None, ["recipes/qwen3.5-4b-phase1/sft.yaml"])
    assert policy.is_optimization_pr(None, ["datasets/registry.jsonl"])
    # not optimization: harness code, docs, a README under recipes/, empty
    assert not policy.is_optimization_pr("fix a bug", ["eval/verify.py"])
    assert not policy.is_optimization_pr(None, ["recipes/qwen3.5-4b-phase1/README.md"])
    assert not policy.is_optimization_pr(None, ["docs/guide.md"])
    assert not policy.is_optimization_pr(None, [])
    assert not policy.is_optimization_pr(None, None)


def test_should_close_community_pr():
    # community + non-optimization -> close
    assert policy.should_close_community_pr(
        author_association="CONTRIBUTOR",
        author_login="randodev",
        author_type="User",
        pr_body="just a refactor",
        changed_paths=["eval/verify.py"],
    )
    # community + optimization -> keep open (the track gate handles it)
    assert not policy.should_close_community_pr(
        author_association="CONTRIBUTOR",
        author_login="randodev",
        author_type="User",
        pr_body=_TRAINING_BODY,
        changed_paths=["recipes/x/sft.yaml"],
    )
    # trusted author + non-optimization -> keep open (exempt)
    for assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert not policy.should_close_community_pr(
            author_association=assoc,
            author_login="jane",
            author_type="User",
            pr_body="refactor",
            changed_paths=["eval/verify.py"],
        )
    # bot + non-optimization -> keep open (exempt)
    assert not policy.should_close_community_pr(
        author_association="NONE",
        author_login="dependabot[bot]",
        author_type="Bot",
        pr_body="bump dep",
        changed_paths=["pyproject.toml", "uv.lock"],
    )


def _capture_run(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(policy.subprocess, "run", fake_run)
    return calls


def _write(tmp_path, body, paths):
    body_file = tmp_path / "pr_body.md"
    body_file.write_text(body, encoding="utf-8")
    paths_file = tmp_path / "changed.txt"
    paths_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return body_file, paths_file


def test_main_closes_community_non_optimization_pr(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    body_file, paths_file = _write(tmp_path, "docs tweak", ["docs/guide.md"])

    rc = policy.main(
        [
            "--author-association",
            "CONTRIBUTOR",
            "--author-login",
            "randodev",
            "--author-type",
            "User",
            "--pr-body-file",
            str(body_file),
            "--changed-paths-file",
            str(paths_file),
            "--pr-number",
            "7",
            "--apply",
        ]
    )
    assert rc == 0
    assert ["gh", "pr", "close", "7"] in calls
    assert any(c[:3] == ["gh", "pr", "comment"] for c in calls)


def test_main_keeps_trusted_author_pr(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    body_file, paths_file = _write(tmp_path, "harness refactor", ["eval/verify.py"])

    rc = policy.main(
        [
            "--author-association",
            "OWNER",
            "--author-login",
            "jane",
            "--author-type",
            "User",
            "--pr-body-file",
            str(body_file),
            "--changed-paths-file",
            str(paths_file),
            "--pr-number",
            "8",
            "--apply",
        ]
    )
    assert rc == 0
    assert not calls  # nothing closed


def test_main_dry_run_does_not_close(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    body_file, paths_file = _write(tmp_path, "docs tweak", ["docs/guide.md"])

    rc = policy.main(
        [
            "--author-association",
            "CONTRIBUTOR",
            "--author-login",
            "randodev",
            "--author-type",
            "User",
            "--pr-body-file",
            str(body_file),
            "--changed-paths-file",
            str(paths_file),
            "--pr-number",
            "9",
        ]
    )
    assert rc == 0
    assert not calls  # dry-run: no gh calls without --apply

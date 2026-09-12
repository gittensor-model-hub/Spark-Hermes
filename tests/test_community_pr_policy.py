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


def test_strategy_training_and_data_tracks_enforce_their_diff_boundaries():
    cases = [
        ("- [x] **Strategy commitment**", "datasets/strategies.jsonl"),
        (_TRAINING_BODY, "recipes/example/sft.yaml"),
        (_DATASET_BODY, "datasets/registry.jsonl"),
    ]
    for body, artifact in cases:
        assert policy.is_optimization_pr(body, [artifact])
        assert not policy.is_optimization_pr(body, [artifact, "validator/score.py"])
        assert not policy.is_optimization_pr(body, [artifact, "README.md"])
    assert not policy.is_optimization_pr("- [x] **Strategy commitment**", ["datasets/registry.jsonl"])
    assert not policy.is_optimization_pr(_TRAINING_BODY, ["datasets/strategies.jsonl"])
    assert not policy.is_optimization_pr({}, ["datasets/strategies.jsonl"])


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


# --- public CI must never be able to see the withheld half -------------------------------------


def test_no_workflow_references_the_withheld_tree_or_salt():
    """The invariant that makes a PR-based submission channel survivable.

    Miners are identified by the GitHub PR they open, so a submission is public the moment it
    exists and so is every log line CI produces about it. If any workflow ran the withheld
    verifier, its verdict would be published to the miner and to every competitor at once --
    turning the withheld check into a check-your-guess oracle and destroying the only thing
    that makes `overfit_rate` measurable.

    `rollout_track.yml` already set the right precedent in its own comment -- "read the
    submission without executing it" -- and no workflow touches the withheld half today. This
    pins that, because the temptation to grade in CI arrives exactly when submissions start
    coming in as pull requests, and nothing else would notice.

    Public CI may validate the contract and run the PUBLISHED checks, both of which a miner can
    run locally anyway. Grading against withheld checks belongs on a private runner, after the
    deadline.
    """
    from pathlib import Path

    forbidden = (
        "HERMESBENCH_WITHHELD_SALT",
        "SPARKDISTILL_WITHHELD_ROOT",
        "hidden_verify",
        "Spark-Hermes-Withheld",
        "verify_hidden",
    )
    offenders: dict[str, list[str]] = {}
    workflows = sorted(Path(".github/workflows").glob("*.yml"))
    assert workflows, "no workflows found; this test would pass vacuously"
    for wf in workflows:
        text = wf.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        if hits:
            offenders[wf.name] = hits
    assert not offenders, (
        f"public CI references the withheld half: {offenders}. A PR is public, so its CI output "
        "is too -- publishing a withheld verdict hands every competitor the answer key."
    )


def test_a_checkout_with_no_private_tree_can_grade_only_public_checks():
    """The property that makes the workflow above safe, tested in the code rather than in YAML
    prose. `hermesbench.yml` legitimately *discusses* withheld checks in its comments, so
    banning the word would fail an honest file; what matters is that CI cannot reach the
    mechanism.

    Without a configured private tree, `overlay` attaches no withheld bodies, so
    `has_hidden_tests` is False across the suite and `verify_hidden` returns None. Public CI
    therefore grades against published checks only as a consequence of how the split works,
    not because a workflow remembered to avoid something."""
    from hermesbench.tasks import load_suite
    from hermesbench.withheld import overlay

    tasks = overlay(load_suite("all"), root=None)
    assert tasks, "empty suite would make this vacuous"
    assert not any(t.has_hidden_tests for t in tasks)
    # The commitments still travel, so a public checkout knows a withheld check EXISTS -- it
    # just cannot run one. That distinction is what `declares_hidden_tests` is for.
    assert all(t.declares_hidden_tests for t in tasks)

"""Withheld checks split across a public suite and a private tree."""

from pathlib import Path

import pytest
import yaml

from hermes.harness import salted_digest
from hermesbench.split_suite import split
from hermesbench.tasks import Task, load_suite
from hermesbench.withheld import WithheldError, overlay, unscorable

SALT = "test-salt-at-least-16-chars"


def _task(**overrides) -> Task:
    record = {
        "task_id": "t1",
        "prompt": "do it",
        "verify": "true",
        "tools": ["terminal"],
        "hidden_verify": "test -f done.txt",
    }
    record.update(overrides)
    return Task.from_record(record)


def _suite(tmp_path, task_ids=("t1",)):
    """A suite on disk, ready to split. One task by default; more when a test is about the
    difference between them -- a partial private tree needs at least two to be about anything."""
    d = tmp_path / "tasks" / "v0"
    d.mkdir(parents=True)
    for task_id in task_ids:
        (d / f"{task_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "task_id": task_id,
                    "prompt": "do it",
                    "verify": "true",
                    "tools": ["terminal"],
                    "hidden_verify": "test -f done.txt\n",
                }
            ),
            encoding="utf-8",
        )
    return tmp_path / "tasks"


# --- the distinction the split exists for ------------------------------------------------


def test_a_redacted_task_still_declares_a_withheld_check():
    """`has_hidden_tests` asks "can I run it", `declares_hidden_tests` asks "does one exist".
    Collapsing them turns overfit_rate from unavailable into a confident zero, and a
    confident zero on the metric that detects benchmark gaming is the more dangerous
    answer."""
    redacted = _task(hidden_verify="", metadata={"hidden_verify_commitment": "sha256:abc"})
    assert redacted.has_hidden_tests is False
    assert redacted.declares_hidden_tests is True
    assert redacted.withheld_check_missing is True


def test_a_task_that_genuinely_has_no_withheld_check_is_not_missing_one():
    plain = _task(hidden_verify="")
    assert plain.declares_hidden_tests is False
    assert plain.withheld_check_missing is False


def test_unscorable_names_the_tasks_a_public_checkout_cannot_score():
    tasks = [_task(task_id="a", hidden_verify="", metadata={"hidden_verify_commitment": "x"}), _task(task_id="b")]
    assert unscorable(tasks) == ["a"]


# --- splitting ---------------------------------------------------------------------------


def test_split_redacts_the_task_and_writes_the_body_out(tmp_path):
    root = _suite(tmp_path)
    moved, already = split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    assert moved == ["t1"] and already == []

    published = yaml.safe_load((root / "v0" / "t1.yaml").read_text())
    assert "hidden_verify" not in published
    assert published["metadata"]["hidden_verify_commitment"].startswith("sha256:")
    assert (tmp_path / "withheld" / "t1.sh").read_text() == "test -f done.txt\n"


def test_split_is_a_dry_run_under_check(tmp_path):
    root = _suite(tmp_path)
    moved, _ = split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT, dry_run=True)
    assert moved == ["t1"]
    assert "hidden_verify" in yaml.safe_load((root / "v0" / "t1.yaml").read_text())
    assert not (tmp_path / "withheld").exists()


def test_splitting_twice_moves_nothing_the_second_time(tmp_path):
    """A second run must not overwrite the private tree with empty files."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    moved, already = split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    assert moved == [] and already == ["t1"]
    assert (tmp_path / "withheld" / "t1.sh").read_text() == "test -f done.txt\n"


# --- the overlay -------------------------------------------------------------------------


def test_overlay_restores_the_withheld_check(tmp_path):
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    public = load_suite("v0", root=root)
    assert public[0].has_hidden_tests is False

    full = overlay(public, root=tmp_path / "withheld", salt=SALT)
    assert full[0].hidden_verify == "test -f done.txt\n"
    assert unscorable(full) == []


def test_no_private_tree_is_a_legitimate_state_not_an_error(tmp_path):
    """The state most contributors are in. It must load, not raise."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    public = load_suite("v0", root=root)
    assert overlay(public, root=None) == public


def test_a_tampered_withheld_check_is_refused(tmp_path):
    """The maintainer-side failure the commitment exists to prevent: the private tree
    drifting from what was published, with nothing to show it."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    (tmp_path / "withheld" / "t1.sh").write_text("exit 0  # always pass\n", encoding="utf-8")
    with pytest.raises(WithheldError, match="does not match the commitment"):
        overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)


def test_attaching_without_a_salt_is_refused(tmp_path):
    """Attaching unchecked would let the private tree drift silently."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    with pytest.raises(WithheldError, match="cannot be verified"):
        overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt="")


def test_a_committed_check_missing_from_the_tree_is_refused(tmp_path):
    """A suite scored without it reports no overfit signal, which reads as a clean result."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    (tmp_path / "withheld" / "t1.sh").unlink()
    # Unattached and named, not fatal. The raise this used to make meant a partial private tree
    # refused the whole suite -- so nineteen committed tasks could not be re-authored one at a
    # time, because the first check written would abort every run until the last one was.
    attached = overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)
    assert "t1" in unscorable(attached)
    assert not next(t for t in attached if t.task_id == "t1").hidden_verify


def test_a_partial_private_tree_still_attaches_what_it_has(tmp_path):
    """The state every re-authoring pass is in. One body present, the rest not: the present one
    must be usable and the absent ones must be named."""
    root = _suite(tmp_path, task_ids=("t1", "t2", "t3"))
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    present = sorted((tmp_path / "withheld").glob("*.sh"))
    assert len(present) == 3, "this test needs more than one task to be about anything"
    for path in present[1:]:
        path.unlink()
    kept = present[0].stem

    attached = overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)
    scorable = [t.task_id for t in attached if t.hidden_verify]
    assert scorable == [kept]
    assert kept not in unscorable(attached)
    assert len(unscorable(attached)) == len(present) - 1


def test_a_mismatching_body_still_refuses_the_whole_suite(tmp_path):
    """The case that must stay fatal. A missing body is work not done yet; a body that
    disagrees with its commitment is a private tree that drifted from what was published, and
    scoring against it measures a different benchmark than the one named."""
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    next(iter(sorted((tmp_path / "withheld").glob("*.sh")))).write_text("echo drifted", encoding="utf-8")
    with pytest.raises(WithheldError, match="does not match the commitment"):
        overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)


def test_a_missing_private_directory_is_refused_rather_than_ignored(tmp_path):
    root = _suite(tmp_path)
    with pytest.raises(WithheldError, match="not a directory"):
        overlay(load_suite("v0", root=root), root=tmp_path / "nope", salt=SALT)


def test_the_commitment_is_salted_under_a_per_task_salt(tmp_path):
    """A bare digest of a short shell command is a check-your-guess oracle. Two salts over
    the same check must not collide.

    The salt is also per-task, derived from the master rather than being it. `split` writes
    commitments on its own path instead of going through `redact_for_release`, so this pins
    that path to the derivation too: committing under the bare master would mean publishing
    one spent task's salt unseals every task still sealed."""
    from hermes.harness import derive_task_salt

    body = "test -f done.txt\n"
    a = salted_digest(body, SALT)
    b = salted_digest(body, "a-completely-different-salt-value")
    assert a != b

    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    published = yaml.safe_load((root / "v0" / "t1.yaml").read_text())["metadata"]["hidden_verify_commitment"]
    assert published == salted_digest(body, derive_task_salt(SALT, "t1"))
    assert published != a, "committed under the bare master, so every task would share one salt"


# --- redaction must not gut the task ------------------------------------------------------

TASK_WITH_COMMENTS = """\
# Why this task exists: a long explanation the maintainer needs.
task_id: t1
prompt: |
  Multi-line prompt with `backticks` and "quotes".
  Second line.
verify: |
  set -e
  test -f done.txt
tools: [terminal]

# Withheld checks. This paragraph names the answer: the count is 72.
# Second line of the same block.
hidden_verify: |
  set -e
  test "$(cat answer.txt)" = "72"

checkpoints:
  - checkpoint_id: c1
    verify: "true"
"""


def test_redaction_preserves_comments_and_formatting():
    """A YAML round-trip dropped 39 comment lines from one real task and folded its prompt
    into an escaped one-liner. The comments are where each trap is explained."""
    from hermesbench.split_suite import redact_text

    public, _ = redact_text(TASK_WITH_COMMENTS)
    assert "# Why this task exists" in public
    assert "Multi-line prompt with `backticks`" in public
    assert "test -f done.txt" in public


def test_redaction_moves_the_comment_that_explains_the_check():
    """One real task's comment names 'the last diverging row and the count of stable
    half-cent rows'. Publishing the prose while hiding the command withholds nothing."""
    from hermesbench.split_suite import redact_text

    public, notes = redact_text(TASK_WITH_COMMENTS)
    assert "the count is 72" not in public
    assert "the count is 72" in notes


def test_redaction_keeps_keys_that_follow_the_withheld_check():
    """`checkpoints:` follows `hidden_verify` in two real tasks. Cutting to end-of-file
    would silently drop a field the public task needs."""
    from hermesbench.split_suite import redact_text

    public, _ = redact_text(TASK_WITH_COMMENTS)
    assert "checkpoints:" in public
    assert "hidden_verify:" not in public


def test_the_written_check_is_what_the_commitment_covers(tmp_path):
    """The .sh must hold the body exactly as `hidden_verify` parsed it. Writing the comment
    block into it too made the overlay's recomputed digest disagree with the commitment."""
    from hermesbench.split_suite import split

    d = tmp_path / "tasks" / "v0"
    d.mkdir(parents=True)
    (d / "t1.yaml").write_text(TASK_WITH_COMMENTS, encoding="utf-8")
    split(withheld_out=tmp_path / "w", tasks_root=tmp_path / "tasks", salt=SALT)

    public = load_suite("v0", root=tmp_path / "tasks")
    restored = overlay(public, root=tmp_path / "w", salt=SALT)
    assert restored[0].hidden_verify.strip().endswith('= "72"')
    assert (tmp_path / "w" / "t1.notes.md").read_text().startswith("# Withheld checks")


# --- reporting the state, because finding it out took a filesystem search ------------------------


def test_status_reports_a_public_checkout_as_unscorable_rather_than_clean(monkeypatch):
    """The state this whole module exists for. 19 tasks commit to a withheld check, a public
    checkout has none of them, and the failure of the old arrangement was that this was reported
    nowhere at all."""
    from hermesbench.tasks import load_suite
    from hermesbench.withheld import status

    monkeypatch.delenv("SPARKDISTILL_WITHHELD_ROOT", raising=False)
    report = status(load_suite("all"))
    assert len(report.committed) == report.tasks == 19
    assert report.attached == ()
    assert len(report.unscorable) == 19
    assert report.root == ""


def test_status_reports_a_complete_private_tree(tmp_path):
    """Measured after attaching. The first version of `status` computed `unscorable` on the input
    tasks, so a complete tree and no tree at all produced the same answer -- the one distinction it
    exists to draw."""
    from hermesbench.tasks import load_suite
    from hermesbench.withheld import status

    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    report = status(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)
    assert report.attached, "a complete tree attaches its bodies"
    assert report.unscorable == ()
    assert report.salt_long_enough is True


def test_status_never_reports_the_salt_itself(monkeypatch):
    """The output is meant to be pasteable into an issue. A length is a fact about a secret; the
    secret is not."""
    from hermesbench.tasks import load_suite
    from hermesbench.withheld import status

    monkeypatch.setenv("HERMESBENCH_WITHHELD_SALT", "a-secret-long-enough-to-pass")
    report = status(load_suite("all"))
    assert report.salt_length == len("a-secret-long-enough-to-pass")
    assert "a-secret-long-enough-to-pass" not in repr(report.to_record())


def test_status_reports_a_broken_tree_as_a_problem_not_as_an_absence(tmp_path, monkeypatch):
    """A configured tree that does not match is a different state from no tree, and collapsing the
    two is how a drifted private tree reads as a public checkout."""
    from hermesbench.tasks import load_suite
    from hermesbench.withheld import status

    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    for path in (tmp_path / "withheld").iterdir():
        path.write_text("echo not the committed check", encoding="utf-8")
    report = status(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)
    assert report.problem, "a mismatch has to surface as a problem"


def test_the_cli_exit_status_can_gate_a_run(monkeypatch, capsys):
    """1 when something committed cannot be scored, so a script can refuse to spend a suite."""
    from hermesbench.withheld import main

    monkeypatch.delenv("SPARKDISTILL_WITHHELD_ROOT", raising=False)
    assert main(["--suite", "all"]) == 1
    assert "UNSCORABLE" in capsys.readouterr().out


# --- sealing a check twice, which is now the normal case -------------------------------------------
#
# `split` was written as a once-ever operation: author the checks inline, run it, commit the two
# halves. Every task in the suite is already past that point, so re-authoring a withheld check and
# sealing it again is the path -- and the sealing step appended a second `metadata:` block.


def test_resealing_does_not_leave_two_commitments(tmp_path):
    """PyYAML takes the last duplicate key, so the value came out right and the file was wrong.
    Another parser takes the first, which is the stale commitment -- and a task committing to a
    body nobody has is exactly the state this whole exercise is digging out of."""
    from hermesbench.split_suite import split

    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    task_file = root / "v0" / "t1.yaml"
    first = yaml.safe_load(task_file.read_text(encoding="utf-8"))["metadata"]["hidden_verify_commitment"]

    # Author a new check over the redacted task, the way a re-authoring pass does.
    task_file.write_text(
        task_file.read_text(encoding="utf-8") + "\nhidden_verify: |\n  test -f rewritten.txt\n",
        encoding="utf-8",
    )
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)

    text = task_file.read_text(encoding="utf-8")
    assert text.count("metadata:") == 1, "a second block leaves the stale commitment ahead of the new one"
    second = yaml.safe_load(text)["metadata"]["hidden_verify_commitment"]
    assert second != first, "a new body must move the commitment"
    assert (tmp_path / "withheld" / "t1.sh").read_text(encoding="utf-8").strip() == "test -f rewritten.txt"


def test_resealing_leaves_the_task_loadable_and_committed(tmp_path):
    """The failure a bare `metadata:` with no children would produce: it parses as None, and
    `Task.from_record` reads that as a task declaring no withheld check at all -- which is a
    clean overfit rate rather than an unavailable one."""
    from hermesbench.split_suite import split

    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    task = next(t for t in load_suite("v0", root=root) if t.task_id == "t1")
    assert task.hidden_verify_commitment
    assert task.withheld_check_missing is True


def test_dropping_the_commitment_keeps_the_comments(tmp_path):
    """`redact_text` is a text operation because `yaml.safe_dump` dropped 39 comment lines from one
    task, and those comments are where each trap is explained. The same constraint applies here."""
    from hermesbench.split_suite import drop_commitment

    text = Path("hermesbench/tasks/v1/migrate-and-keep-green.yaml").read_text(encoding="utf-8")
    cleaned = drop_commitment(text)
    assert cleaned.count("#") == text.count("#")
    assert "hidden_verify_commitment" not in cleaned
    assert yaml.safe_load(cleaned)["task_id"] == "migrate-and-keep-green"

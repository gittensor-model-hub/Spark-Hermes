"""Withheld checks split across a public suite and a private tree."""

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


def _suite(tmp_path):
    """A one-task suite on disk, ready to split."""
    d = tmp_path / "tasks" / "v0"
    d.mkdir(parents=True)
    (d / "t1.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "t1",
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
    with pytest.raises(WithheldError, match="does not exist"):
        overlay(load_suite("v0", root=root), root=tmp_path / "withheld", salt=SALT)


def test_a_missing_private_directory_is_refused_rather_than_ignored(tmp_path):
    root = _suite(tmp_path)
    with pytest.raises(WithheldError, match="not a directory"):
        overlay(load_suite("v0", root=root), root=tmp_path / "nope", salt=SALT)


def test_the_commitment_is_salted(tmp_path):
    """A bare digest of a short shell command is a check-your-guess oracle. Two salts over
    the same check must not collide."""
    body = "test -f done.txt\n"
    a = salted_digest(body, SALT)
    b = salted_digest(body, "a-completely-different-salt-value")
    assert a != b
    root = _suite(tmp_path)
    split(withheld_out=tmp_path / "withheld", tasks_root=root, salt=SALT)
    assert yaml.safe_load((root / "v0" / "t1.yaml").read_text())["metadata"]["hidden_verify_commitment"] == a


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

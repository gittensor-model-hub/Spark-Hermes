"""Accepting a surface the validator will run, and refusing the ways of cheating with one.

The question this module answers is "can we verify the miner only touched the allowed surface and
is not cheating". Under miner-side generation the answer was no: the attestation proves genuine
confidential hardware and binds to the exported files, but no approved guest measurement is pinned,
so a miner on hardware they own can boot any image and the quote still verifies. Nothing about the
surface was submitted, so `hermes.miner_contract` was unenforced code.

Submitting the surface and having the validator run it removes the whole problem. The files are in
the pull request, so the contract is checked against what was actually committed; the model,
environment and runtime are the validator's own; and the withheld verifiers stay withheld.

Every test below drives the real `gate`, and the assignment is recomputed from a **fixed** seed.
An unseeded round reassigns the task on every run -- the first draft of these checks used
`new_seed()` and every case failed with a path mismatch rather than the reason under test, which
looked like eight bugs in the gate and was one bug in the fixture.
"""

import json
import os
import shutil

import pytest

from eval.strategy_track import (
    STRATEGY_REGISTRY,
    StrategyError,
    StrategySubmission,
    check_append_only,
    check_shape,
    digest_surface,
    gate,
)
from hermes.seed import OPEN, Round

SEED = "a" * 64
TASK = "tc-log-rotation-order"
MINERS = ("alice", "bob")


@pytest.fixture
def round_():
    return Round(round_id="r-001", seed=SEED, task_ids=(TASK,), miner_ids=MINERS, replicas=1, state=OPEN)


@pytest.fixture
def owner(round_):
    return round_.assignees(TASK)[0]


@pytest.fixture
def surface(tmp_path, owner):
    """A valid surface committed at the path the record's own fields derive."""
    from miner.cli import scaffold

    root = tmp_path / "submissions" / "r-001" / owner
    root.mkdir(parents=True)
    scaffold(root, skill="step-budget")
    return tmp_path


def _record(owner, surface_root, **kw):
    root = surface_root / "submissions" / "r-001" / owner
    record = {
        "schema_version": 1,
        "round_id": "r-001",
        "miner_id": owner,
        "task_ids": [TASK],
        "surface_digests": digest_surface(root),
    }
    record.update(kw)
    return record


def _run(record, round_, head_root, *, base_text="", changed=None):
    own = f"submissions/r-001/{record.get('miner_id')}/SOUL.md"
    head_text = (base_text + "\n" if base_text else "") + json.dumps(record)
    return gate(
        record=record,
        round_record=round_.to_record(reveal_seed=True),
        head_root=head_root,
        base_text=base_text,
        head_text=head_text,
        changed_paths=[STRATEGY_REGISTRY.as_posix(), own] if changed is None else changed,
    )


# --- the accepting case --------------------------------------------------------------------------


def test_a_valid_surface_from_the_assigned_miner_is_accepted(round_, owner, surface):
    assert _run(_record(owner, surface), round_, surface) == []


def test_the_assignment_is_recomputed_rather_than_trusted(round_, owner, surface):
    """The record says which miner; the announcement decides whether they were owed the work. A
    gate that believed the record would let anyone claim any task."""
    other = next(m for m in MINERS if m != owner)
    issues = _run(
        _record(owner, surface, miner_id=other),
        round_,
        surface,
        changed=[STRATEGY_REGISTRY.as_posix(), f"submissions/r-001/{other}/SOUL.md"],
    )
    assert any("were not assigned to" in i for i in issues)


# --- the ways of cheating with a surface ---------------------------------------------------------


def test_a_file_present_but_not_claimed_is_refused(round_, owner, surface):
    """The one that matters most, because it is the quiet one. The validator loads the directory,
    so an unclaimed file runs undeclared and the registry line is an incomplete description of what
    actually ran."""
    root = surface / "submissions" / "r-001" / owner
    record = _record(owner, surface)
    (root / "extra.md").write_text("undeclared\n", encoding="utf-8")
    issues = _run(record, round_, surface)
    assert any("extra.md" in i and "not claimed" in i for i in issues)


def test_a_claimed_file_that_is_absent_is_refused(round_, owner, surface):
    root = surface / "submissions" / "r-001" / owner
    record = _record(owner, surface)
    (root / "SOUL.md").unlink()
    issues = _run(record, round_, surface)
    assert any("SOUL.md" in i and "not present" in i for i in issues)


def test_citing_one_surface_and_shipping_another_is_refused(round_, owner, surface):
    record = _record(owner, surface)
    record["surface_digests"]["SOUL.md"] = "sha256:" + "0" * 64
    issues = _run(record, round_, surface)
    assert any("the registry line claims" in i for i in issues)


def test_an_executable_in_the_surface_is_refused(round_, owner, surface):
    """Load-bearing here in a way it is not on the rollout track: the validator is about to run
    these files."""
    root = surface / "submissions" / "r-001" / owner
    (root / "run_agent.py").write_text("import os\n", encoding="utf-8")
    issues = _run(_record(owner, surface), round_, surface)
    assert any("run_agent.py" in i for i in issues)


def test_a_symlink_named_like_prose_is_refused(round_, owner, surface):
    """The contract matches names, so `notes.md -> /etc/passwd` satisfies every rule while
    resolving to whatever the validator's filesystem holds."""
    root = surface / "submissions" / "r-001" / owner
    os.symlink("/etc/passwd", root / "notes.md")
    issues = _run(_record(owner, surface), round_, surface)
    assert any("notes.md" in i and "symlink" in i for i in issues)


def test_editing_another_miners_surface_is_refused(round_, owner, surface):
    """Not a submission -- an attack on theirs. Scoped to this submission's own directory rather
    than to `submissions/` as a whole."""
    other = next(m for m in MINERS if m != owner)
    issues = _run(
        _record(owner, surface),
        round_,
        surface,
        changed=[STRATEGY_REGISTRY.as_posix(), f"submissions/r-001/{other}/SOUL.md"],
    )
    assert any("Editing another miner" in i for i in issues)


def test_a_miner_id_cannot_escape_the_submissions_root():
    """The miner id becomes a directory component and the validator loads whatever is at that
    path, so `..` would point the run outside the submissions tree.

    Built without touching disk on purpose: the first version reused the on-disk fixture with a
    hardcoded miner name, so when the seeded assignment picked the other miner the record came out
    with no digests and the failure was "missing surface_digests" -- a true error, and not the one
    under test.
    """
    record = {
        "schema_version": 1,
        "round_id": "r-001",
        "miner_id": "../../hermes",
        "task_ids": [TASK],
        "surface_digests": {"SOUL.md": "sha256:" + "5" * 64},
    }
    assert any("single path segment" in i for i in check_shape(record))
    assert any("single path segment" in i for i in check_shape({**record, "miner_id": "a/b"}))
    assert any("single path segment" in i for i in check_shape({**record, "round_id": ".."}))


def test_a_second_surface_in_one_round_is_refused(round_, owner, surface):
    """One standing surface per miner per round. Replacements go through the round window, which
    records what replaced what; a second registry line would not."""
    record = _record(owner, surface)
    issues = _run(record, round_, surface, base_text=json.dumps(record))
    assert any("already has a surface standing" in i for i in issues)


def test_the_registry_is_append_only():
    """A rewrite is refused even when it is an improvement. Checked directly, because building a
    "rewrite" by concatenating onto the base produces an honest append -- which is how the first
    version of this test passed while asserting the opposite."""
    prior = json.dumps({"schema_version": 1, "round_id": "r-000", "miner_id": "zed", "task_ids": ["t"]})
    new = json.dumps({"schema_version": 1, "round_id": "r-001", "miner_id": "alice", "task_ids": ["t"]})
    assert check_append_only(prior, new), "dropping the prior line is a rewrite"
    assert check_append_only(prior, prior + "\n" + new) == [], "preserving it is an append"


def test_an_empty_surface_is_refused(round_, owner, surface):
    """It would run as the unmodified baseline while occupying a submission slot."""
    root = surface / "submissions" / "r-001" / owner
    shutil.rmtree(root)
    root.mkdir()
    issues = _run({**_record(owner, surface), "surface_digests": {"SOUL.md": "sha256:" + "2" * 64}}, round_, surface)
    assert issues


def test_a_missing_surface_directory_is_refused(round_, owner, surface):
    record = _record(owner, surface)
    shutil.rmtree(surface / "submissions" / "r-001" / owner)
    issues = _run(record, round_, surface)
    assert any("nothing to run" in i for i in issues)


# --- the record ----------------------------------------------------------------------------------


def test_the_surface_directory_is_derived_never_read_from_the_record():
    """A path read out of the submission is a path the submitter chose, and the one thing this gate
    must not let a miner choose is which files the validator is about to load."""
    submission = StrategySubmission(
        round_id="r-001", miner_id="alice", task_ids=(TASK,), surface_digests={"SOUL.md": "sha256:" + "3" * 64}
    )
    assert submission.surface_dir.as_posix() == "submissions/r-001/alice"


def test_a_record_missing_required_fields_is_refused():
    with pytest.raises(StrategyError, match="missing"):
        StrategySubmission.from_record({"round_id": "r-001"})


def test_surface_digests_must_be_a_mapping():
    with pytest.raises(StrategyError, match="mapping"):
        StrategySubmission.from_record(
            {
                "schema_version": 1,
                "round_id": "r-001",
                "miner_id": "alice",
                "task_ids": ["t"],
                "surface_digests": ["SOUL.md"],
            }
        )


def test_a_symlink_is_digested_as_a_link_not_as_its_target(tmp_path):
    """Following it would digest a file that is not in the submission, so the recorded digest would
    describe something the pull request does not contain."""
    root = tmp_path / "s"
    root.mkdir()
    os.symlink("/etc/passwd", root / "notes.md")
    assert digest_surface(root)["notes.md"].startswith("symlink:")


def test_the_record_round_trips_through_its_canonical_form():
    """`to_record` sorts `task_ids` and `surface_digests`, so the round trip returns the canonical
    ordering rather than the input ordering. That is deliberate and load-bearing: the record is
    digested, and a record whose byte form depends on dict insertion order has no stable digest."""
    submission = StrategySubmission(
        round_id="r-001", miner_id="alice", task_ids=(TASK, "b"), surface_digests={"SOUL.md": "sha256:" + "4" * 64}
    )
    canonical = StrategySubmission.from_record(submission.to_record())
    assert canonical.task_ids == tuple(sorted(submission.task_ids))
    assert StrategySubmission.from_record(canonical.to_record()) == canonical
    assert json.dumps(canonical.to_record(), sort_keys=True) == json.dumps(submission.to_record(), sort_keys=True)

"""Publishing a round announcement, and the gate being able to read it.

`eval.rollout_track_cli.load_round` read `datasets/rounds/<id>.json` and nothing wrote it --
the directory did not exist. So every rollout submission failed with "round ... has no
announcement in the base ref; it was never opened": the gate failing closed, correctly, on a
record no code path could produce. These tests drive the writer and then hand its output to
the real `check_scope`, because a writer whose output the gate rejects has not fixed anything.
"""

import json

import pytest

from eval.rollout_track import check_scope
from hermes.announce import close_round, commit_round, describe, open_seed
from hermes.seed import CLOSED, COMMITTED, OPEN, SeedError

TASKS = ["tc-log-rotation-order", "sv-retry-request-budget", "tc-nested-archive-manifest"]
MINERS = ["alice", "bob", "carol"]


def _commit(root, round_id="r-1", **kw):
    kw.setdefault("task_ids", TASKS)
    kw.setdefault("miner_ids", MINERS)
    return commit_round(round_id=round_id, root=root, **kw)


# --- commit withholds the seed ----------------------------------------------------------------


def test_the_commitment_announcement_does_not_carry_the_seed(tmp_path):
    """A seed a miner can predict is one a miner can grind against: register identities until
    the assignment hands you the tasks you already solved. Publishing the digest first fixes
    the assignment before anyone knows what it is."""
    path, round_ = _commit(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["state"] == COMMITTED
    assert "seed" not in record
    assert record["commitment"] == round_.commitment


def test_a_committed_round_cannot_have_its_assignment_computed_yet(tmp_path):
    _commit(tmp_path)
    described = describe("r-1", tmp_path)
    assert "not computable yet" in described["assignment"]


def test_re_committing_is_refused_rather_than_overwriting(tmp_path):
    """Rewriting an announcement changes an assignment miners may already be working against."""
    _commit(tmp_path)
    with pytest.raises(SeedError, match="already exists"):
        _commit(tmp_path)


# --- reveal is checked against the commitment -------------------------------------------------


def test_a_seed_that_does_not_match_the_commitment_is_refused(tmp_path):
    """Without this the commit-reveal proves nothing: a validator could publish one digest and
    later reveal whichever seed produced a convenient assignment."""
    _commit(tmp_path)
    with pytest.raises(SeedError, match="does not match the published commitment"):
        open_seed(round_id="r-1", seed="0" * 64, root=tmp_path)


def test_the_right_seed_opens_the_round_and_publishes_the_assignment(tmp_path):
    _, round_ = _commit(tmp_path)
    path, opened = open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["state"] == OPEN
    assert record["seed"] == round_.seed
    assert opened.commitment == round_.commitment

    described = describe("r-1", tmp_path)
    assert len(described["assignments"]) == len(TASKS)
    assert sum(described["coverage"].values()) == len(TASKS)


def test_a_round_cannot_be_reopened(tmp_path):
    """Re-revealing would let an assignment change after miners saw it."""
    _, round_ = _commit(tmp_path)
    open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    with pytest.raises(SeedError, match="only a COMMITTED round"):
        open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)


def test_opening_a_round_that_was_never_committed_is_refused(tmp_path):
    with pytest.raises(SeedError, match="commit the round first"):
        open_seed(round_id="nope", seed="0" * 64, root=tmp_path)


# --- closing ----------------------------------------------------------------------------------


def test_closing_an_open_round_marks_it_closed(tmp_path):
    _, round_ = _commit(tmp_path)
    open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    path, record = close_round(round_id="r-1", root=tmp_path)
    assert record["state"] == CLOSED
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == CLOSED


def test_closing_is_idempotent(tmp_path):
    _, round_ = _commit(tmp_path)
    open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    close_round(round_id="r-1", root=tmp_path)
    _, again = close_round(round_id="r-1", root=tmp_path)
    assert again["state"] == CLOSED


def test_closing_a_committed_round_is_refused(tmp_path):
    """It would record a window nobody could submit to as though it had run."""
    _commit(tmp_path)
    with pytest.raises(SeedError, match="never opened"):
        close_round(round_id="r-1", root=tmp_path)


# --- the gate can read what this writes -------------------------------------------------------


def test_the_gate_accepts_an_assigned_miner_and_rejects_everyone_else(tmp_path):
    """The point of the whole module. A writer whose output `check_scope` rejects has fixed
    nothing, so the record goes through the real gate rather than an assertion about its shape."""
    _, round_ = _commit(tmp_path)
    path, opened = open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))

    task = TASKS[0]
    owner = opened.assignees(task)[0]
    stranger = next(m for m in MINERS if m != owner)

    assert check_scope({"round_id": "r-1", "miner_id": owner, "task_ids": [task]}, record) == []
    refused = check_scope({"round_id": "r-1", "miner_id": stranger, "task_ids": [task]}, record)
    assert refused and "not assigned" in refused[0]


def test_the_gate_refuses_a_committed_round_because_no_window_existed(tmp_path):
    """The seed is out for nobody, so nobody was owed this work yet.

    It refuses one layer earlier than `check_scope`'s own COMMITTED branch: `Round.from_record`
    raises on a record with no seed, so the branch reading
    "committed but not open; there was no window to work in" is unreachable for a correctly
    withheld announcement. It would need a record carrying state=committed AND a seed, which
    `commit_round` never writes. Asserted on the message that actually arrives, because a test
    matching the unreachable one would pass only until someone deleted dead code.
    """
    _commit(tmp_path)
    record = json.loads((tmp_path / "r-1.json").read_text(encoding="utf-8"))
    issues = check_scope({"round_id": "r-1", "miner_id": "alice", "task_ids": [TASKS[0]]}, record)
    assert issues and "seed is not revealed yet" in issues[0]


def test_the_gate_refuses_a_closed_round(tmp_path):
    _, round_ = _commit(tmp_path)
    open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    _, record = close_round(round_id="r-1", root=tmp_path)
    issues = check_scope({"round_id": "r-1", "miner_id": "alice", "task_ids": [TASKS[0]]}, record)
    assert issues and "no longer owes anyone work" in issues[0]


def test_replicas_produce_cross_checkable_assignments(tmp_path):
    """`replicas > 1` is deliberate duplication: two independent runs of a deterministic
    verifier must agree, so a disagreement is evidence about the miners rather than the task."""
    _, round_ = _commit(tmp_path, replicas=2)
    open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    described = describe("r-1", tmp_path)
    assert described["cross_checked"], "replicas=2 must report which tasks are cross-checked"
    assert all(len(entry["miners"]) == 2 for entry in described["cross_checked"])

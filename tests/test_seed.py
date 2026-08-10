"""Validator seeding: assignment anyone can recompute, and nobody can grind against."""

import json

import pytest

from hermes.seed import (
    OPEN,
    Round,
    SeedError,
    coverage,
    duplicated,
    new_seed,
    open_round,
    score,
    seed_commitment,
    verify_submission_scope,
)

TASKS = [f"task-{i:03d}" for i in range(30)]
MINERS = ["miner-alpha", "miner-bravo", "miner-charlie"]
SEED = "a" * 64


def _round(**overrides) -> Round:
    kwargs = {"round_id": "r1", "seed": SEED, "task_ids": tuple(TASKS), "miner_ids": tuple(MINERS)}
    kwargs.update(overrides)
    return Round(**kwargs)


# --- the assignment is a pure function ------------------------------------------------


def test_the_same_round_assigns_the_same_way_every_time():
    """A miner has to be able to recompute this without asking the validator."""
    assert _round().assignees("task-000") == _round().assignees("task-000")


def test_a_different_seed_assigns_differently():
    a = _round().assignments()
    b = _round(seed="b" * 64).assignments()
    assert a != b


def test_every_task_is_assigned_to_somebody():
    round_ = _round()
    assert all(round_.assignees(t) for t in TASKS)


def test_work_is_spread_across_the_miners():
    """Rendezvous balances in expectation; a wildly skewed round is worth seeing early."""
    counts = coverage(_round())
    assert set(counts) == set(MINERS)
    assert sum(counts.values()) == len(TASKS)
    assert min(counts.values()) > 0


def test_a_miner_can_list_exactly_its_own_tasks():
    round_ = _round()
    mine = round_.assigned_to("miner-alpha")
    assert all(round_.owns("miner-alpha", t) for t in mine)
    assert not any(round_.owns("miner-alpha", t) for t in set(TASKS) - set(mine))


# --- losing a miner must not reshuffle the round ----------------------------------------


def test_removing_a_miner_only_reassigns_that_miners_tasks():
    """A modulo scheme would reshuffle everything and invalidate work already in progress."""
    before = _round()
    after = _round(miner_ids=("miner-alpha", "miner-bravo"))
    for task in TASKS:
        if before.assignees(task)[0] != "miner-charlie":
            assert after.assignees(task) == before.assignees(task), task


def test_adding_a_miner_only_moves_tasks_to_the_newcomer():
    before = _round()
    after = _round(miner_ids=(*MINERS, "miner-delta"))
    moved = [t for t in TASKS if after.assignees(t) != before.assignees(t)]
    assert all(after.assignees(t)[0] == "miner-delta" for t in moved)


# --- the seed is committed before it is known --------------------------------------------


def test_a_commitment_hides_the_seed():
    assert SEED not in seed_commitment(SEED)


def test_a_commitment_announcement_does_not_reveal_the_assignment():
    """Publishing the digest first stops a validator picking the seed after seeing who registered."""
    record = _round().to_record(reveal_seed=False)
    assert "seed" not in record
    assert record["commitment"].startswith("sha256:")
    with pytest.raises(SeedError, match="not revealed yet"):
        Round.from_record(record)


def test_a_seed_that_does_not_match_its_commitment_is_refused():
    """A swapped seed means the assignment was not fixed before the round opened."""
    record = _round().to_record()
    record["commitment"] = seed_commitment("c" * 64)
    with pytest.raises(SeedError, match="not fixed before the round opened"):
        Round.from_record(record)


def test_a_weak_seed_cannot_be_committed_to():
    with pytest.raises(SeedError, match="128 bits"):
        seed_commitment("short")


def test_generated_seeds_differ():
    assert new_seed() != new_seed()


# --- malformed rounds ---------------------------------------------------------------------


def test_a_round_with_no_tasks_is_refused():
    with pytest.raises(SeedError, match="assigns nothing"):
        _round(task_ids=())


def test_a_round_with_no_miners_is_refused():
    with pytest.raises(SeedError, match="nobody to assign to"):
        _round(miner_ids=())


def test_duplicate_miner_ids_are_refused():
    with pytest.raises(SeedError, match="duplicate miner id"):
        _round(miner_ids=("a", "a"))


def test_duplicate_task_ids_are_refused():
    with pytest.raises(SeedError, match="duplicate task id"):
        _round(task_ids=("t", "t"))


def test_more_replicas_than_miners_is_refused():
    with pytest.raises(SeedError, match="more distinct miners than exist"):
        _round(replicas=4)


def test_zero_replicas_is_refused():
    with pytest.raises(SeedError, match="assigned to nobody"):
        _round(replicas=0)


def test_an_unknown_round_state_is_refused():
    with pytest.raises(SeedError, match="unknown round state"):
        _round(state="whenever")


# --- deliberate duplication for cross-checking ---------------------------------------------


def test_one_replica_duplicates_nothing():
    assert duplicated(_round()) == ()


def test_two_replicas_send_each_task_to_two_distinct_miners():
    """The only way to catch a miner that fabricates: two runs of a deterministic verifier
    must agree, and a miner cannot know whether it is the duplicate."""
    round_ = _round(replicas=2)
    for dup in duplicated(round_):
        assert len(dup.miners) == 2 and len(set(dup.miners)) == 2


def test_replication_doubles_the_work():
    assert sum(coverage(_round(replicas=2)).values()) == 2 * len(TASKS)


# --- scope checking on submission ------------------------------------------------------------


def test_a_submission_for_assigned_tasks_is_in_scope():
    round_ = _round()
    mine = list(round_.assigned_to("miner-bravo"))
    assert verify_submission_scope(round_.to_record(), miner_id="miner-bravo", task_ids=mine) == []


def test_a_submission_for_someone_elses_task_is_out_of_scope():
    round_ = _round()
    theirs = [t for t in TASKS if not round_.owns("miner-bravo", t)]
    out = verify_submission_scope(round_.to_record(), miner_id="miner-bravo", task_ids=theirs[:3])
    assert out == theirs[:3]


def test_a_submission_for_a_task_outside_the_round_is_out_of_scope():
    out = verify_submission_scope(_round().to_record(), miner_id="miner-alpha", task_ids=["not-in-the-pool"])
    assert out == ["not-in-the-pool"]


def test_partial_scope_returns_only_the_rows_to_drop():
    """A partly out-of-scope submission is a partial acceptance, not a rejection."""
    round_ = _round()
    mine = list(round_.assigned_to("miner-alpha"))
    theirs = [t for t in TASKS if not round_.owns("miner-alpha", t)]
    out = verify_submission_scope(round_.to_record(), miner_id="miner-alpha", task_ids=mine[:2] + theirs[:1])
    assert out == theirs[:1]


def test_an_unregistered_miner_is_refused_rather_than_silently_empty():
    with pytest.raises(SeedError, match="not registered"):
        _round().assigned_to("miner-nobody")


# --- round trip -----------------------------------------------------------------------------


def test_a_round_survives_publication():
    round_ = _round(replicas=2)
    rebuilt = Round.from_record(json.loads(json.dumps(round_.to_record())))
    assert rebuilt.assignments() == round_.assignments()


def test_open_round_generates_a_seed():
    round_ = open_round("r2", TASKS, MINERS)
    assert len(round_.seed) == 64 and round_.state == OPEN


def test_scores_are_hex_so_ordering_is_language_independent():
    """A validator and a miner disagreeing about integer width would produce different
    assignments from one seed, and the disagreement would look like cheating."""
    value = score(SEED, "r1", "task-000", "miner-alpha")
    assert len(value) == 64 and int(value, 16) >= 0


def test_the_commitment_covers_the_whole_announcement_not_only_the_seed():
    """The pool and the roster decide the assignment as much as the seed does."""
    base = _round()
    assert base.commitment != _round(task_ids=(*TASKS, "task-999")).commitment
    assert base.commitment != _round(miner_ids=(*MINERS, "miner-delta")).commitment
    assert base.commitment != _round(replicas=2).commitment


def test_swapping_the_pool_after_committing_is_refused():
    """A validator could otherwise publish a digest, watch who registers, then change the
    task set and still match its own commitment."""
    published = _round().to_record()
    published["task_ids"] = [*TASKS, "task-999"]
    with pytest.raises(SeedError, match="not fixed before the round opened"):
        Round.from_record(published)


def test_an_announcement_with_no_commitment_is_refused():
    record = _round().to_record()
    del record["commitment"]
    with pytest.raises(SeedError, match="never fixed"):
        Round.from_record(record)


def test_a_weak_seed_is_refused_on_construction_not_only_on_publish():
    """Otherwise a round rebuilt from a record uses it happily on every read path."""
    with pytest.raises(SeedError, match="128 bits"):
        _round(seed="short")

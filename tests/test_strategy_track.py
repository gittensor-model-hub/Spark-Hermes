"""Binding a miner to a bundle before anything runs.

The surface is private and the pull request carries only its digest, so this gate does the one
thing a public, timestamped, attributable record can do that a private upload cannot: fix which
bundle a miner is judged on, before any of them has been run.

An earlier version of this file tested a gate that took the surface *in the pull request*. That
made the surface public, which destroys the thing a miner competes with, and every file-shaped
check moved to `validator.intake`. What remains is attribution and commitment.

The assignment is recomputed from a **fixed** seed throughout. An unseeded round reassigns the task
on every call, and the first draft of the older gate's tests used `new_seed()` -- eight cases then
failed on a path mismatch rather than the reason under test, which read as eight bugs in the gate
and was one bug in the fixture.
"""

import json

import pytest

from eval.strategy_track import (
    SCHEMA_VERSION,
    STRATEGY_REGISTRY,
    Commitment,
    StrategyError,
    added_lines,
    check_append_only,
    check_author,
    check_commitment,
    check_shape,
    gate,
)
from hermes.seed import OPEN, Round
from validator.intake import Intake

TASK = "tc-log-rotation-order"
SEED = "a" * 64
BUNDLE = {
    "SOUL.md": "# Operating identity\nOne call per turn.\n",
    "skills/p/SKILL.md": "---\nname: p\ndescription: d\n---\n# P\n\n1. Close the tag.\n",
}


@pytest.fixture
def world(tmp_path):
    """A round, an uploaded bundle, and the receipt it produced."""
    intake = Intake(root=tmp_path / "store", receipts=tmp_path / "receipts.jsonl")
    round_ = Round(round_id="r-1", seed=SEED, task_ids=(TASK,), miner_ids=("alice", "bob"), replicas=1, state=OPEN)
    owner = round_.assignees(TASK)[0]
    receipt = intake.accept(round_id="r-1", miner_id=owner, files=BUNDLE, now=10.0)
    return intake, round_, owner, receipt


def _commit(owner, receipt, **kw):
    fields = {
        "round_id": "r-1",
        "miner_id": owner,
        "bundle_sha256": receipt.bundle_sha256,
        "task_ids": (TASK,),
    }
    fields.update(kw)
    return Commitment(**fields)


def _run(world, commitment, *, author, base_text="", changed=None):
    intake, round_, _, _ = world
    return gate(
        record=commitment.to_record(),
        round_record=round_.to_record(reveal_seed=True),
        receipts=intake.read_receipts(),
        base_text=base_text,
        head_text=(base_text + "\n" if base_text else "") + json.dumps(commitment.to_record()),
        changed_paths=[STRATEGY_REGISTRY.as_posix()] if changed is None else changed,
        pull_request_author=author,
    )


# --- the accepting case ---------------------------------------------------------------------------


def test_a_miner_committing_to_their_own_bundle_is_accepted(world):
    _, _, owner, receipt = world
    assert _run(world, _commit(owner, receipt), author=owner) == []


# --- attribution: the attack receipts being public creates -------------------------------------------


def test_a_miner_cannot_commit_to_someone_elses_bundle(tmp_path):
    """Every digest the validator has accepted is public, so the obvious move is to name someone
    else's and have their work scored under your name.

    Tested with `replicas=2` so both miners are assigned the task. With one assignee the scope check
    fires first and this one is never reached -- which would leave it dead code passing by proxy.
    """
    intake = Intake(root=tmp_path / "store", receipts=tmp_path / "receipts.jsonl")
    round_ = Round(round_id="r-1", seed=SEED, task_ids=(TASK,), miner_ids=("alice", "bob"), replicas=2, state=OPEN)
    assignees = round_.assignees(TASK)
    assert len(assignees) == 2, "both must be assigned or scope masks the check under test"
    uploader, thief = assignees

    receipt = intake.accept(round_id="r-1", miner_id=uploader, files=BUNDLE, now=10.0)
    stolen = Commitment(round_id="r-1", miner_id=thief, bundle_sha256=receipt.bundle_sha256, task_ids=(TASK,))

    problems = check_commitment(stolen, intake.read_receipts())
    assert problems and "was uploaded by" in problems[0]

    issues = gate(
        record=stolen.to_record(),
        round_record=round_.to_record(reveal_seed=True),
        receipts=intake.read_receipts(),
        base_text="",
        head_text=json.dumps(stolen.to_record()),
        changed_paths=[STRATEGY_REGISTRY.as_posix()],
        pull_request_author=thief,
    )
    assert any("was uploaded by" in i for i in issues)


def test_the_pull_request_author_must_be_the_claimed_miner(world):
    """The record is written by the submitter, so a `miner_id` field alone proves nothing about who
    opened the pull request."""
    _, _, owner, receipt = world
    other = "alice" if owner == "bob" else "bob"
    issues = _run(world, _commit(owner, receipt), author=other)
    assert any("opened by" in i for i in issues)


def test_an_unattributed_commitment_is_refused(world):
    """An unauthenticated line could name any bundle the validator holds, and receipts are public."""
    _, _, owner, receipt = world
    issues = _run(world, _commit(owner, receipt), author=None)
    assert any("cannot be attributed" in i for i in issues)


def test_check_author_is_given_the_author_rather_than_reading_it(world):
    """GitHub is the only party that can say who opened a pull request. A gate that read it from
    the record would be checking the submitter against themselves."""
    _, _, owner, receipt = world
    assert check_author(_commit(owner, receipt), owner) == []
    assert check_author(_commit(owner, receipt), "someone-else")


# --- the commitment names something that exists --------------------------------------------------------


def test_a_digest_nothing_was_uploaded_for_is_refused(world):
    """A claim on a bundle that could be written afterwards to fit whatever result was convenient."""
    _, _, owner, receipt = world
    issues = _run(world, _commit(owner, receipt, bundle_sha256="sha256:" + "0" * 64), author=owner)
    assert any("was uploaded for round" in i for i in issues)


def test_a_digest_from_another_round_does_not_count(tmp_path):
    """Uploads are per round. A bundle accepted in an earlier round is not a commitment in this one."""
    intake = Intake(root=tmp_path / "store", receipts=tmp_path / "receipts.jsonl")
    receipt = intake.accept(round_id="r-0", miner_id="bob", files=BUNDLE, now=10.0)
    stale = Commitment(round_id="r-1", miner_id="bob", bundle_sha256=receipt.bundle_sha256, task_ids=(TASK,))
    assert check_commitment(stale, intake.read_receipts())


# --- scope ----------------------------------------------------------------------------------------------


def test_the_assignment_is_recomputed_rather_than_trusted(world):
    """The line says which miner; the announcement decides whether they were owed the work."""
    _, _, owner, receipt = world
    other = "alice" if owner == "bob" else "bob"
    issues = _run(world, _commit(owner, receipt, miner_id=other), author=other)
    assert any("were not assigned to" in i for i in issues)


# --- the registry ------------------------------------------------------------------------------------------


def test_only_the_registry_may_change(world):
    """Data-only, so auto-merge can never carry code."""
    _, _, owner, receipt = world
    issues = _run(
        world, _commit(owner, receipt), author=owner, changed=[STRATEGY_REGISTRY.as_posix(), "hermes/round.py"]
    )
    assert any("may only change" in i for i in issues)


def test_the_registry_is_append_only():
    """Checked directly. Building a "rewrite" by concatenating onto the base produces an honest
    append, which is how an earlier version of this test passed while asserting the opposite."""
    prior = json.dumps({"schema_version": SCHEMA_VERSION, "round_id": "r-0", "miner_id": "zed"})
    new = json.dumps({"schema_version": SCHEMA_VERSION, "round_id": "r-1", "miner_id": "bob"})
    assert check_append_only(prior, new), "dropping the prior line is a rewrite"
    assert check_append_only(prior, prior + "\n" + new) == []


def test_a_second_commitment_in_one_round_is_refused(world):
    """Uploading twice is allowed and cheap -- the digest decides which is evaluated -- so the moment
    that choice becomes binding has to be a single, dated, public one."""
    _, _, owner, receipt = world
    line = json.dumps(_commit(owner, receipt).to_record())
    issues = _run(world, _commit(owner, receipt), author=owner, base_text=line)
    assert any("already has a commitment standing" in i for i in issues)


def test_added_lines_are_positional(world):
    _, _, owner, receipt = world
    prior = json.dumps({"round_id": "r-0", "miner_id": "zed"})
    line = json.dumps(_commit(owner, receipt).to_record())
    assert added_lines(prior, prior + "\n" + line) == [json.loads(line)]


def test_a_malformed_appended_line_is_an_error():
    with pytest.raises(StrategyError, match="not JSON"):
        added_lines("", "{not json")


@pytest.mark.parametrize("value", [None, [], 3, True, "metadata", {"schema_version": "2"}])
def test_non_object_or_malformed_metadata_is_a_refusal(value):
    assert check_shape(value)


def test_prior_registry_bytes_are_immutable():
    prior = '{"round_id": "old"}'
    assert check_append_only(prior, '{"round_id":"old"}\n{}')


def test_gated_record_must_be_the_exact_appended_delta(world):
    intake, round_, owner, receipt = world
    record = _commit(owner, receipt).to_record()
    issues = gate(
        record=record,
        round_record=round_.to_record(reveal_seed=True),
        receipts=intake.read_receipts(),
        base_text="",
        head_text=json.dumps({**record, "notes": "different"}),
        changed_paths=[STRATEGY_REGISTRY.as_posix()],
        pull_request_author=owner,
    )
    assert issues and "one commitment" in issues[0]


# --- shape ------------------------------------------------------------------------------------------------


def test_a_record_missing_required_fields_is_refused():
    assert check_shape({"round_id": "r-1"})
    with pytest.raises(StrategyError, match="missing"):
        Commitment.from_record({"round_id": "r-1"})


def test_a_version_one_record_is_refused_with_the_reason():
    """Version 1 carried the surface files themselves. A gate that silently accepted one would be
    reading a record whose meaning has changed."""
    issues = check_shape(
        {"schema_version": 1, "round_id": "r-1", "miner_id": "bob", "bundle_sha256": "sha256:" + "1" * 64}
    )
    assert any("surface is private now" in i for i in issues)


def test_a_digest_that_is_not_a_sha256_is_refused():
    for bad in ("deadbeef", "sha256:short", "md5:" + "1" * 32):
        assert check_shape({"schema_version": SCHEMA_VERSION, "round_id": "r-1", "miner_id": "b", "bundle_sha256": bad})


def test_the_record_round_trips(world):
    _, _, owner, receipt = world
    commitment = _commit(owner, receipt)
    assert Commitment.from_record(commitment.to_record()) == commitment


def test_every_reason_is_reported_not_just_the_first(world):
    """A miner who learns one problem per pull request stops opening them."""
    _, _, owner, receipt = world
    other = "alice" if owner == "bob" else "bob"
    issues = _run(
        world,
        _commit(owner, receipt, miner_id=other, bundle_sha256="sha256:" + "0" * 64),
        author="stranger",
        changed=[STRATEGY_REGISTRY.as_posix(), "setup.py"],
    )
    assert len(issues) >= 3

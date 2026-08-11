"""Running, scoring and recording a merged surface, in the order the lifecycle allows.

These were three commands typed by hand in an end-to-end test. The interesting part is not the
sequencing but the refusals: the round object already enforces most of what matters, and the first
version of this driver worked around two of its guards instead of following them.

`run` is injected so every one of these exercises the ordering, the digest check, the skip logic
and the state transitions without a served model. The parts most likely to be wrong are the parts
that need no GPU.
"""

import json

import pytest

from validator.intake import Intake
from validator.judge import JudgeError, accept, bundle_dir_for, judge_round
from validator.round_loop import open_from_packet
from validator.store import RoundStore

TASK = "tc-log-rotation-order"


def _episodes(tmp_path, n=10, *, tokens=78_000, public=False, hidden=None):
    """A baseline log the challenge opener accepts: a task the baseline reliably fails."""
    path = tmp_path / "base.jsonl"
    rows = [
        {
            "task_id": TASK,
            "metrics": {
                "task_id": TASK,
                "public_passed": public,
                "hidden_passed": hidden,
                "tokens_used": tokens + i * 1_500,
                "tool_calls": 11,
                "wall_time_s": 300.0,
                "steps": 34,
                "max_steps_hit": True,
            },
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path):
    """A stored round, plus an intake holding the bundles miners uploaded.

    The bundles live in the intake store, not in a checkout: the surface is uploaded privately and
    the pull request carries only its digest, so there is nothing in the tree to run.
    """
    store = RoundStore(tmp_path / "rounds", require_private=False)
    intake = Intake(root=tmp_path / "bundles", receipts=tmp_path / "receipts.jsonl")
    packet = tmp_path / "packet.json"
    from hermes.challenge import Attempt, Baseline, open_challenge

    attempts = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=78_000 + i * 1_500,
            tool_calls=11,
            wall_time_s=300.0,
            steps=34,
            max_steps_hit=True,
        )
        for i in range(10)
    )
    challenge = open_challenge(
        Baseline(task_id=TASK, attempts=attempts),
        epoch={"model_revision": "a" * 40, "harness_digest": "b" * 64},
        task_pins={"task_id": TASK, "hidden_verify_commitment": "sha256:" + "c" * 64},
    )
    packet.write_text(json.dumps(challenge.to_record()), encoding="utf-8")
    open_from_packet(
        packet_path=packet,
        episodes_path=_episodes(tmp_path),
        round_id="r-1",
        hours=24.0,
        store=store,
        now=0.0,
    )
    return store, intake, challenge.epoch


def _upload(intake, miner, *, body="# id\nOne call per turn.\n"):
    """Upload a bundle the way a miner would, and return its receipt."""
    return intake.accept(
        round_id="r-1",
        miner_id=miner,
        files={"SOUL.md": body, "skills/p/SKILL.md": "---\nname: p\ndescription: d\n---\n# P\n"},
        now=10.0,
    )


def _log(rows):
    def run(miner, workspace):
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / f"{miner}.jsonl"
        path.write_text("\n".join(json.dumps({"task_id": TASK, "metrics": r}) for r in rows) + "\n", encoding="utf-8")
        return path

    return run


def _rows(n=3, *, tokens=57_000, public=True, hidden=True):
    return [
        {
            "task_id": TASK,
            "public_passed": public,
            "hidden_passed": hidden,
            "tokens_used": tokens,
            "tool_calls": 6,
            "steps": 20,
            "malformed_turns": 0,
        }
        for _ in range(n)
    ]


def _judge(setup, run, **kw):
    store, intake, epoch = setup
    return judge_round(
        round_id="r-1",
        run=run,
        model_revision=epoch["model_revision"],
        store=store,
        intake=intake,
        repo_root=intake.root,
        workspace=intake.root.parent / "ws",
        scorecard_dir=intake.root.parent / "cards",
        **kw,
    )


# --- accept, at merge time --------------------------------------------------------------------


def test_accept_records_the_committed_digest(setup):
    """The digest on the receipt is the one the miner published in their pull request. Recording
    anything else would let the thing that runs differ from the thing committed to."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    round_receipt = accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    assert round_receipt.outcome == "accepted"
    assert store.load("r-1").submissions["carol"].payload_digest == receipt.bundle_sha256


def test_a_receipt_belonging_to_another_miner_or_round_is_refused(setup):
    """`accept` takes a receipt, so the one check it must make is that the receipt is the caller's."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    with pytest.raises(JudgeError, match="belongs to"):
        accept(round_id="r-1", miner_id="dave", receipt=receipt, store=store, intake=intake)


# --- the lifecycle refuses out-of-order judging -------------------------------------------------


def test_judging_an_open_round_is_refused_rather_than_freezing_it(setup):
    """Closing a window is a decision about the miners still working in it. A judge that froze as
    a side effect would take that decision by accident."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    with pytest.raises(JudgeError, match="needs a FROZEN round"):
        _judge(setup, _log(_rows()))


def test_judging_a_settled_round_is_refused(setup):
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)
    _judge(setup, _log(_rows()))
    with pytest.raises(JudgeError, match="needs a FROZEN round"):
        _judge(setup, _log(_rows()))


# --- the surface that runs is the surface that was checked ---------------------------------------


def test_a_stored_bundle_that_no_longer_matches_its_commitment_is_not_run(setup):
    """The digest was published before anything ran. If the store no longer agrees with it, "the
    digest was committed" and "this is what ran" are two claims joined by an assumption."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    (bundle_dir_for(receipt, intake=intake) / "SOUL.md").write_text("# id\nTAMPERED\n", encoding="utf-8")
    results = _judge(setup, _log(_rows()))
    assert results[0].scorecard is None
    assert "no longer digests to what was committed" in results[0].problem


def test_an_untouched_bundle_is_run(setup):
    """The other half, and the half that broke once. An earlier version compared against
    `Receipt.submission_digest` -- a digest of the submission *record* -- so the check could never
    match and refused every legitimate run. An always-refusing guard is as broken as an
    always-passing one and looks more responsible."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    results = _judge(setup, _log(_rows()))
    assert results[0].scorecard is not None, results[0].problem


# --- grading only when every standing submission has a verdict -----------------------------------


def test_a_skipped_submission_leaves_the_round_frozen(setup):
    """`RoundWindow.grade` refuses a round with a miner missing, and its reason is the right one: a
    rate over the wrong denominator, with the omission indistinguishable from a miner who never
    submitted. The first version of this driver graded unconditionally and hit exactly that."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)
    (bundle_dir_for(receipt, intake=intake) / "SOUL.md").write_text("# tampered\n", encoding="utf-8")

    _judge(setup, _log(_rows()))
    assert store.load("r-1").state == "frozen"


def test_a_failed_run_does_not_record_a_false_verdict(setup):
    """`passed=False` means the withheld check failed. A run that died was never put to it, and
    writing a verdict anyway makes an infrastructure problem permanently indistinguishable from a
    miner's bad strategy."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    def boom(miner, workspace):
        raise RuntimeError("the model endpoint went away")

    results = _judge(setup, boom)
    assert "the run failed" in results[0].problem
    reloaded = store.load("r-1")
    assert reloaded.state == "frozen"
    assert reloaded.verdicts == {}


def test_one_broken_run_does_not_stop_the_others(setup):
    """A run that dies partway through is the normal way this fails, not an exotic one."""
    store, intake, _ = setup
    for miner in ("carol", "dave"):
        accept(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    good = _log(_rows())

    def flaky(miner, workspace):
        if miner == "carol":
            raise RuntimeError("boom")
        return good(miner, workspace)

    results = _judge(setup, flaky)
    by_miner = {r.miner_id: r for r in results}
    assert by_miner["carol"].scorecard is None
    assert by_miner["dave"].scorecard is not None


# --- the happy path ------------------------------------------------------------------------------


def test_every_submission_judged_grades_and_settles_the_round(setup):
    store, intake, _ = setup
    for miner in ("carol", "dave"):
        accept(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    results = _judge(setup, _log(_rows()))
    assert all(r.scorecard is not None for r in results)

    reloaded = store.load("r-1")
    assert reloaded.state == "settled"
    assert set(reloaded.verdicts) == {"carol", "dave"}
    assert sorted(p.name for p in (intake.root.parent / "cards").glob("*.json")) == ["r-1-carol.json", "r-1-dave.json"]


def test_overfit_is_recorded_as_a_failure_and_counted_apart(setup):
    """Passing the published check and failing the withheld one is not a pass, and the count is
    kept separate so it is not read as a capability gap."""
    store, intake, _ = setup
    accept(round_id="r-1", miner_id="dave", receipt=_upload(intake, "dave"), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    results = _judge(setup, _log(_rows(hidden=False)))
    card = results[0].scorecard
    assert card is not None
    assert card.candidate.passes == 0 and card.overfit_attempts == 3
    assert store.load("r-1").verdicts["dave"].passed is False


def test_no_settle_grades_without_releasing_the_salt(setup):
    """Releasing the per-task salt is what turns the commitment into an audit, and it is separable
    from grading so an operator can publish verdicts before opening the commitment."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    accept(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    _judge(setup, _log(_rows()), settle=False)
    assert store.load("r-1").state == "graded"


def test_a_partially_judged_round_can_be_resumed(setup):
    """Miners already carrying a verdict are skipped rather than re-judged: `record_verdict` allows
    one per miner, so a second pass would raise on the first and abandon everyone after them."""
    store, intake, _ = setup
    for miner in ("carol", "dave"):
        accept(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    good = _log(_rows())

    def only_carol(miner, workspace):
        if miner == "dave":
            raise RuntimeError("not yet")
        return good(miner, workspace)

    _judge(setup, only_carol)
    assert store.load("r-1").state == "frozen"

    results = _judge(setup, good)
    by_miner = {r.miner_id: r for r in results}
    assert by_miner["carol"].problem == "already judged; skipped"
    assert by_miner["dave"].scorecard is not None
    assert store.load("r-1").state == "settled"

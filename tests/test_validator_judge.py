"""Running, scoring and recording a merged surface, in the order the lifecycle allows.

These were three commands typed by hand in an end-to-end test. The interesting part is not the
sequencing but the refusals: the round object already enforces most of what matters, and the first
version of this driver worked around two of its guards instead of following them.

`run` is injected so every one of these exercises the ordering, the digest check, the skip logic
and the state transitions without a served model. The parts most likely to be wrong are the parts
that need no GPU.
"""

import json
from pathlib import Path

import pytest
from competition_support import admit_receipt
from competition_support import rows as evidence_rows
from competition_support import window as fixture_window

from validator.intake import Intake
from validator.judge import JudgeError, bundle_dir_for, judge_round
from validator.judge import accept as direct_accept
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


_ORIGIN = None


@pytest.fixture
def setup(tmp_path):
    global _ORIGIN
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture")
    intake = Intake(root=tmp_path / "bundles", receipts=tmp_path / "receipts.jsonl", mode="fixture")
    window = fixture_window(store)
    _ORIGIN = store.identity
    return store, intake, window.challenge.epoch


def _admit(*, round_id, miner_id, receipt, store, intake):
    if receipt.miner_id != miner_id or receipt.round_id != round_id:
        return direct_accept(round_id=round_id, miner_id=miner_id, receipt=receipt, store=store, intake=intake)
    from types import SimpleNamespace

    with pytest.MonkeyPatch.context() as patch:
        admit_receipt(store, intake, receipt, patch)
    return SimpleNamespace(outcome="accepted", miner=miner_id)


def _upload(intake, miner, *, body="# id\nOne call per turn.\n"):
    """Upload a bundle the way a miner would, and return its receipt."""
    return intake.accept(
        round_id="r-1",
        miner_id=miner,
        files={"SOUL.md": body, "skills/p/SKILL.md": "---\nname: p\ndescription: d\n---\n# P\n"},
        now=10.0,
    )


def _log(rows):
    def run(miner, bundle_path, workspace):
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / f"{miner}.jsonl"
        from validator.intake import bundle_digest

        digest = bundle_digest(
            {p.relative_to(bundle_path).as_posix(): p.read_text() for p in bundle_path.rglob("*") if p.is_file()}
        )
        stamped = [{**r, "origin": _ORIGIN, "bundle_sha256": digest} for r in rows]
        path.write_text(
            "\n".join(json.dumps({"task_id": TASK, "metrics": r}) for r in stamped) + "\n", encoding="utf-8"
        )
        return path

    return run


def _rows(n=10, *, tokens=57_000, public=True, hidden=True):
    return evidence_rows(n, tokens=tokens, public=public, hidden=hidden)


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
    round_receipt = _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    assert round_receipt.outcome == "accepted"
    assert store.load("r-1").submissions["carol"].payload_digest == receipt.bundle_sha256


def test_a_receipt_belonging_to_another_miner_or_round_is_refused(setup):
    """`accept` takes a receipt, so the one check it must make is that the receipt is the caller's."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    with pytest.raises(JudgeError, match="belongs to"):
        _admit(round_id="r-1", miner_id="dave", receipt=receipt, store=store, intake=intake)


# --- the lifecycle refuses out-of-order judging -------------------------------------------------


def test_judging_an_open_round_is_refused_rather_than_freezing_it(setup):
    """Closing a window is a decision about the miners still working in it. A judge that froze as
    a side effect would take that decision by accident."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    with pytest.raises(JudgeError, match="needs a FROZEN round"):
        _judge(setup, _log(_rows()))


def test_judging_a_settled_round_is_refused(setup):
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
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
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
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
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
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
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
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
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    def boom(miner, bundle_path, workspace):
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
        _admit(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    good = _log(_rows())

    def flaky(miner, bundle_path, workspace):
        if miner == "carol":
            raise RuntimeError("boom")
        return good(miner, bundle_path, workspace)

    results = _judge(setup, flaky)
    by_miner = {r.miner_id: r for r in results}
    assert by_miner["carol"].scorecard is None
    assert by_miner["dave"].scorecard is not None


# --- the happy path ------------------------------------------------------------------------------


def test_every_submission_judged_grades_and_settles_the_round(setup):
    store, intake, _ = setup
    for miner in ("carol", "dave"):
        _admit(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
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
    _admit(round_id="r-1", miner_id="dave", receipt=_upload(intake, "dave"), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    results = _judge(setup, _log(_rows(hidden=False)))
    card = results[0].scorecard
    assert card is not None
    assert card.candidate.passes == 0 and card.overfit_attempts == 10
    assert store.load("r-1").verdicts["dave"].passed is False


def test_no_settle_grades_without_releasing_the_salt(setup):
    """Releasing the per-task salt is what turns the commitment into an audit, and it is separable
    from grading so an operator can publish verdicts before opening the commitment."""
    store, intake, _ = setup
    receipt = _upload(intake, "carol")
    _admit(round_id="r-1", miner_id="carol", receipt=receipt, store=store, intake=intake)
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
        _admit(round_id="r-1", miner_id=miner, receipt=_upload(intake, miner), store=store, intake=intake)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    good = _log(_rows())

    def only_carol(miner, bundle_path, workspace):
        if miner == "dave":
            raise RuntimeError("not yet")
        return good(miner, bundle_path, workspace)

    _judge(setup, only_carol)
    assert store.load("r-1").state == "frozen"

    results = _judge(setup, good)
    by_miner = {r.miner_id: r for r in results}
    assert by_miner["carol"].problem == "already judged; skipped"
    assert by_miner["dave"].scorecard is not None
    assert store.load("r-1").state == "settled"


# --- what the judge hands the runner --------------------------------------------------------------
#
# Two defaults that were wrong in opposite directions, and both were invisible until a round was
# actually judged against a served model.


def test_a_judged_round_keeps_its_trajectories():
    """`validator.aggregate` builds every SFT row and preference pair FROM trajectories. Without
    them a judged round yields no training data at all -- and aggregate refuses with "no episode
    carries a trajectory" rather than writing an empty file, so the loop stops dead one step after
    the crown."""
    from miner.evaluate import runner_argv

    argv = runner_argv(
        task_id="t1",
        base_url="http://x/v1",
        model="m",
        api_key_env="K",
        workspace_root=Path("/tmp/ws"),
        episodes_out=Path("/tmp/ws/out.jsonl"),
        repeats=10,
        miner_dir=Path("/tmp/bundle"),
        allow_unsandboxed=True,
    )
    assert "--keep-trajectories" in argv


def test_the_dialect_is_the_pin_unless_the_caller_says_otherwise():
    """Empty is right when the served model IS the pinned one, which is the normal case."""
    from miner.evaluate import runner_argv

    common = dict(
        task_id="t1",
        base_url="http://x/v1",
        model="m",
        api_key_env="K",
        workspace_root=Path("/tmp/ws"),
        episodes_out=Path("/tmp/ws/out.jsonl"),
        repeats=1,
        miner_dir=None,
        allow_unsandboxed=False,
    )
    assert "--dialect" not in runner_argv(**common)
    assert runner_argv(**common, dialect="atem")[-2:] == ["--dialect", "atem"] or "atem" in runner_argv(
        **common, dialect="atem"
    )


def test_the_judge_passes_the_dialect_it_was_given(monkeypatch, tmp_path):
    """A validator serving a model other than the pinned one -- during a migration, or to compare
    two -- otherwise instructs a wire format the model does not speak. That is not a loud failure:
    it surfaces as malformed turns, or as a model that never calls a tool, and both read as the
    model being bad rather than the harness asking wrongly."""
    from validator.judge import runner_for

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        Path(argv[argv.index("--episodes-out") + 1]).write_text("", encoding="utf-8")
        return 0

    import hermesbench.runner as runner_module

    monkeypatch.setattr(runner_module, "main", fake_main)
    run = runner_for(
        round_id="r-1",
        base_url="http://x/v1",
        model="m",
        api_key_env="K",
        task_id="t1",
        repeats=1,
        repo_root=tmp_path,
        allow_unsandboxed=False,
        dialect="atem",
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run("carol", tmp_path / "exact-bundle", workspace)
    assert "--dialect" in captured["argv"] and "atem" in captured["argv"]

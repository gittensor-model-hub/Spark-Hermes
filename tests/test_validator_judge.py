"""Running, scoring and recording a merged surface, in the order the lifecycle allows.

These were three commands typed by hand in an end-to-end test. The interesting part is not the
sequencing but the refusals: the round object already enforces most of what matters, and the first
version of this driver worked around two of its guards instead of following them.

`run` is injected so every one of these exercises the ordering, the digest check, the skip logic
and the state transitions without a served model. The parts most likely to be wrong are the parts
that need no GPU.
"""

import json
import shutil

import pytest

from validator.judge import JudgeError, accept, judge_round, surface_digest, surface_dir
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
    """A stored round over the real challenge packet, plus a repo root to put surfaces in."""
    store = RoundStore(tmp_path / "rounds", require_private=False)
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
    return store, tmp_path / "repo", challenge.epoch


def _surface(repo, miner, *, body="# id\nOne call per turn.\n"):
    root = surface_dir("r-1", miner, root=repo)
    (root / "skills" / "p").mkdir(parents=True, exist_ok=True)
    (root / "SOUL.md").write_text(body, encoding="utf-8")
    (root / "skills" / "p" / "SKILL.md").write_text("---\nname: p\ndescription: d\n---\n# P\n", encoding="utf-8")
    return root


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
    store, repo, epoch = setup
    return judge_round(
        round_id="r-1",
        run=run,
        model_revision=epoch["model_revision"],
        store=store,
        repo_root=repo,
        workspace=repo.parent / "ws",
        scorecard_dir=repo.parent / "cards",
        **kw,
    )


# --- accept, at merge time --------------------------------------------------------------------


def test_accept_records_the_surface_digest_not_a_digest_of_the_line(setup):
    """The registry line describes the submission; the surface is what will be executed, and it is
    what `judge` has to be able to check it is still running."""
    store, repo, _ = setup
    root = _surface(repo, "carol")
    receipt = accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    assert receipt.outcome == "accepted"
    assert store.load("r-1").submissions["carol"].payload_digest == surface_digest(root)


def test_accepting_a_surface_that_is_not_in_the_tree_is_refused(setup):
    store, repo, _ = setup
    with pytest.raises(JudgeError, match="no surface to accept"):
        accept(round_id="r-1", miner_id="ghost", store=store, repo_root=repo)


# --- the lifecycle refuses out-of-order judging -------------------------------------------------


def test_judging_an_open_round_is_refused_rather_than_freezing_it(setup):
    """Closing a window is a decision about the miners still working in it. A judge that froze as
    a side effect would take that decision by accident."""
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    with pytest.raises(JudgeError, match="needs a FROZEN round"):
        _judge(setup, _log(_rows()))


def test_judging_a_settled_round_is_refused(setup):
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)
    _judge(setup, _log(_rows()))
    with pytest.raises(JudgeError, match="needs a FROZEN round"):
        _judge(setup, _log(_rows()))


# --- the surface that runs is the surface that was checked ---------------------------------------


def test_a_surface_edited_after_acceptance_is_not_run(setup):
    """The gate checked the pull request head; this runs the merged tree, and a later commit can
    touch a directory an earlier gate approved."""
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    _surface(repo, "carol", body="# id\nSOMETHING ELSE ENTIRELY\n")
    results = _judge(setup, _log(_rows()))
    assert results[0].scorecard is None
    assert "changed after it was accepted" in results[0].problem


def test_an_unmodified_surface_is_run(setup):
    """The other half, and the half that broke. The first version compared the surface digest
    against `Receipt.submission_digest` -- a digest of the submission *record*, a different value
    entirely -- so the check could never match and refused every legitimate run. An always-refusing
    guard is as broken as an always-passing one and looks more responsible."""
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
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
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)
    _surface(repo, "carol", body="# tampered\n")

    _judge(setup, _log(_rows()))
    assert store.load("r-1").state == "frozen"


def test_a_failed_run_does_not_record_a_false_verdict(setup):
    """`passed=False` means the withheld check failed. A run that died was never put to it, and
    writing a verdict anyway makes an infrastructure problem permanently indistinguishable from a
    miner's bad strategy."""
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
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
    store, repo, _ = setup
    for miner in ("carol", "dave"):
        _surface(repo, miner)
        accept(round_id="r-1", miner_id=miner, store=store, repo_root=repo, now=10.0)
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
    store, repo, _ = setup
    for miner in ("carol", "dave"):
        _surface(repo, miner)
        accept(round_id="r-1", miner_id=miner, store=store, repo_root=repo, now=10.0)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    results = _judge(setup, _log(_rows()))
    assert all(r.scorecard is not None for r in results)

    reloaded = store.load("r-1")
    assert reloaded.state == "settled"
    assert set(reloaded.verdicts) == {"carol", "dave"}
    assert sorted(p.name for p in (repo.parent / "cards").glob("*.json")) == ["r-1-carol.json", "r-1-dave.json"]


def test_overfit_is_recorded_as_a_failure_and_counted_apart(setup):
    """Passing the published check and failing the withheld one is not a pass, and the count is
    kept separate so it is not read as a capability gap."""
    store, repo, _ = setup
    _surface(repo, "dave")
    accept(round_id="r-1", miner_id="dave", store=store, repo_root=repo, now=10.0)
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
    store, repo, _ = setup
    _surface(repo, "carol")
    accept(round_id="r-1", miner_id="carol", store=store, repo_root=repo, now=10.0)
    window = store.load("r-1")
    window.freeze(now=1e9, reason="t")
    store.save(window)

    _judge(setup, _log(_rows()), settle=False)
    assert store.load("r-1").state == "graded"


def test_a_partially_judged_round_can_be_resumed(setup):
    """Miners already carrying a verdict are skipped rather than re-judged: `record_verdict` allows
    one per miner, so a second pass would raise on the first and abandon everyone after them."""
    store, repo, _ = setup
    for miner in ("carol", "dave"):
        _surface(repo, miner)
        accept(round_id="r-1", miner_id=miner, store=store, repo_root=repo, now=10.0)
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


def test_the_surface_digest_moves_when_a_file_is_added_or_removed(tmp_path):
    """A digest of the per-file digest map, not of a concatenation: adding, removing and editing
    must all move it, which concatenating contents in directory order would not reliably do."""
    root = tmp_path / "s"
    (root / "skills" / "p").mkdir(parents=True)
    (root / "SOUL.md").write_text("a\n", encoding="utf-8")
    (root / "skills" / "p" / "SKILL.md").write_text("b\n", encoding="utf-8")
    first = surface_digest(root)

    (root / "skills" / "p" / "references").mkdir()
    (root / "skills" / "p" / "references" / "n.md").write_text("c\n", encoding="utf-8")
    assert surface_digest(root) != first

    shutil.rmtree(root / "skills" / "p" / "references")
    assert surface_digest(root) == first, "removing what was added must return the original digest"

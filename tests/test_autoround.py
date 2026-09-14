"""The round cycle's driver: which command is next, and the one action nothing else performs."""

import json
import time

import pytest

from hermes.challenge import Attempt, Baseline, open_challenge
from hermes.harness import derive_task_salt, salted_digest
from hermes.round import open_round
from validator.autoround import (
    BLOCKED,
    CROWN,
    FREEZE,
    JUDGE,
    OPEN_NEXT,
    STALLED,
    WAIT,
    AutoRoundError,
    Decision,
    choose_task,
    decide,
    next_round_id,
    open_next,
    resolve_episodes,
)
from validator.store import RoundStore

SALT = "0" * 32
EPOCH = {"model_revision": "a" * 40, "harness_digest": "h" * 16}


def _challenge(task_id="tc-log-rotation-order"):
    attempts = tuple(
        Attempt(public_passed=False, hidden_passed=None, tokens=10_000, tool_calls=20, wall_time_s=30.0, steps=8)
        for _ in range(10)
    )
    pins = {
        "task_id": task_id,
        "prompt": "rotate the logs in order",
        "verify": "pytest -q",
        "hidden_verify_commitment": salted_digest("check", derive_task_salt(SALT, task_id)),
    }
    return open_challenge(Baseline(task_id=task_id, attempts=attempts), epoch=EPOCH, task_pins=pins)


def _store(tmp_path):
    return RoundStore(tmp_path / "rounds", require_private=False)


def _open_window(store, *, round_id="r-001", task="tc-log-rotation-order", opened=1000.0, hours=1.0):
    window = open_round(_challenge(task), round_id=round_id, opened_at=opened, deadline=opened + hours * 3600.0)
    store.save(window)
    return window


def _pool(tmp_path, *names):
    root = tmp_path / "challenges"
    root.mkdir(exist_ok=True)
    for name in names:
        (root / f"{name}.json").write_text("{}", encoding="utf-8")
    return root


# --- deciding ----------------------------------------------------------------------


def test_no_round_and_a_stocked_pool_opens_the_next_one(tmp_path):
    d = decide(_store(tmp_path), challenges=_pool(tmp_path, "task-a", "task-b"))
    assert d.action == OPEN_NEXT
    assert d.round_id == "r-001"


def test_an_empty_pool_is_blocked_not_silent(tmp_path):
    """The consumer loop draws from a pool. Nothing here stocks it, so an empty one is a real
    condition an operator has to act on, not a quiet hour."""
    d = decide(_store(tmp_path), challenges=tmp_path / "challenges")
    assert d.action == BLOCKED
    assert "nothing is stocking it" in d.reason


def test_an_open_window_before_its_deadline_waits(tmp_path):
    store = _store(tmp_path)
    _open_window(store, opened=1000.0, hours=1.0)
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=1000.0 + 600)
    assert d.action == WAIT
    assert d.detail["seconds_remaining"] == pytest.approx(3000.0)


def test_past_the_deadline_the_next_action_is_freeze(tmp_path):
    store = _store(tmp_path)
    _open_window(store, opened=1000.0, hours=1.0)
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=1000.0 + 3601)
    assert d.action == FREEZE


def test_a_frozen_round_awaits_judging(tmp_path):
    store = _store(tmp_path)
    window = _open_window(store, opened=1000.0)
    window.freeze(now=5000.0)
    store.save(window)
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=6000.0)
    assert d.action == JUDGE


def test_a_graded_round_awaits_the_crown(tmp_path):
    store = _store(tmp_path)
    window = _open_window(store, opened=1000.0)
    window.freeze(now=5000.0)
    window.grade(now=5100.0)
    store.save(window)
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=5200.0)
    assert d.action == CROWN


def test_a_round_that_has_not_moved_is_reported_stalled(tmp_path):
    """'Invalid or missing evidence leaves the round frozen' produces no crown, no next round and
    no error. Left to silence it is indistinguishable from a quiet hour."""
    store = _store(tmp_path)
    window = _open_window(store, opened=1000.0)
    window.freeze(now=5000.0)
    store.save(window)
    # Frozen at t=5000; 6h later it is stalled, and the clock runs from the FREEZE, not the open.
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=5000.0 + 6 * 3600 + 1, stall_after_s=6 * 3600)
    assert d.action == STALLED
    assert d.detail["state"] == "frozen"
    assert d.detail["since"] == 5000.0
    # Five hours after the freeze is not stalled, even though the round OPENED more than six ago.
    d = decide(store, challenges=_pool(tmp_path, "task-a"), now=5000.0 + 5 * 3600, stall_after_s=6 * 3600)
    assert d.action == JUDGE


def test_two_unsettled_rounds_are_refused_rather_than_chosen_between(tmp_path):
    """Two live windows means miners may be submitting to either. No default here can repair that."""
    store = _store(tmp_path)
    _open_window(store, round_id="r-001", task="task-a")
    _open_window(store, round_id="r-002", task="task-b")
    d = decide(store, challenges=_pool(tmp_path, "task-a"))
    assert d.action == BLOCKED
    assert "more than one round is unsettled" in d.reason
    # And `open` refuses outright: it is the action that would make it three.
    with pytest.raises(AutoRoundError, match="more than one round is unsettled"):
        open_next(store, challenges=_pool(tmp_path, "task-a"), episodes=tmp_path, hours=1.0)


# --- ids, rotation, evidence -------------------------------------------------------


def test_round_ids_continue_the_stored_sequence(tmp_path):
    store = _store(tmp_path)
    assert next_round_id(store) == "r-001"
    _open_window(store, round_id="r-007")
    assert next_round_id(store) == "r-008"


def test_the_pool_rotates_round_robin(tmp_path):
    pool = _pool(tmp_path, "task-a", "task-b", "task-c")
    assert choose_task(pool, current="task-a") == "task-b"
    assert choose_task(pool, current="task-c") == "task-a", "rotation wraps"
    assert choose_task(pool, current="") == "task-a", "a first round starts at the beginning"


def test_a_directory_of_logs_resolves_by_task_id(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "task-a.jsonl").write_text('{"task_id": "task-a"}\n', encoding="utf-8")
    assert resolve_episodes(logs, "task-a").name == "task-a.jsonl"


def test_a_log_naming_the_task_is_found_when_the_filename_does_not_match(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "probe-2026-09.jsonl").write_text('{"task_id": "task-a", "metrics": {}}\n', encoding="utf-8")
    assert resolve_episodes(logs, "task-a").name == "probe-2026-09.jsonl"


def test_no_log_for_the_task_is_an_error_not_a_guess(tmp_path):
    """A round built from the wrong evidence gets a zero token spread, which does not read as
    missing data -- it reads as a perfectly repeatable task."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "other.jsonl").write_text('{"task_id": "task-z"}\n', encoding="utf-8")
    with pytest.raises(AutoRoundError, match="mentions task-a"):
        resolve_episodes(logs, "task-a")


# --- opening -----------------------------------------------------------------------


def test_opening_while_a_round_is_live_is_refused(tmp_path):
    """The store refuses to overwrite a round. This is the case it cannot see: a NEW id opened
    while the previous window is still taking submissions."""
    store = _store(tmp_path)
    _open_window(store, round_id="r-001", task="task-a")
    with pytest.raises(AutoRoundError, match="not settled"):
        open_next(store, challenges=_pool(tmp_path, "task-a"), episodes=tmp_path, hours=1.0)


def test_the_decision_record_round_trips_as_json():
    record = Decision(WAIT, round_id="r-001", reason="x", detail={"k": 1}).to_record()
    assert json.loads(json.dumps(record))["detail"] == {"k": 1}


def test_wait_is_not_an_alerting_condition():
    """A scheduled driver runs every minute; if ordinary waiting alerted, nobody would read it."""
    assert WAIT not in (STALLED, BLOCKED)


def test_decide_defaults_to_now(tmp_path):
    store = _store(tmp_path)
    _open_window(store, opened=time.time(), hours=1.0)
    assert decide(store, challenges=_pool(tmp_path, "task-a")).action == WAIT

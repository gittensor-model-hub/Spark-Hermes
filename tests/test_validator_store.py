"""Rounds surviving a restart, and the two records that must never be confused.

`validator.api` held its rounds in a module dict that nothing filled, so a live server answered
404 to every round and a restart during grading lost the grading. These tests drive the store and
then hand its output to the real API, because a store whose output the server cannot load has
fixed nothing.

The property worth the most attention is which record gets persisted. `RoundWindow.to_record()`
withholds `verdicts` until GRADED -- correctly, since a verdict served during an open round turns
the withheld verifier into a check-your-guess oracle. Saving through it would drop every verdict
recorded before grading finished, the reload would report success, and the round would come back
up publishing `verdicts_recorded: 0` as though none had been made. So there is a second, private
record, and `screen_public_payload` refuses it on sight.
"""

import json
from pathlib import Path

import pytest

from hermes.challenge import Attempt, Baseline, Challenge, ChallengeError, open_challenge
from hermes.round import (
    PUBLIC_VIEW_FIELDS,
    FreezeToken,
    RoundError,
    RoundWindow,
    ScoreLeakError,
    open_round,
    screen_public_payload,
)
from validator.store import RoundStore, StoreError, store_is_private

EPOCH = {"model_revision": "a" * 40, "harness_digest": "b" * 64}
PINS = {"task_id": "t", "hidden_verify_commitment": "sha256:" + "c" * 64}


def _challenge(task_id="t"):
    attempts = tuple(
        Attempt(
            public_passed=False,
            hidden_passed=None,
            tokens=30_000 + i * 900,
            tool_calls=9,
            wall_time_s=40.0,
            steps=12,
            max_steps_hit=True,
        )
        for i in range(10)
    )
    return open_challenge(
        Baseline(task_id=task_id, attempts=attempts),
        epoch=EPOCH,
        task_pins={**PINS, "task_id": task_id},
    )


def _window(round_id="r-1", task_id="t"):
    return open_round(_challenge(task_id), round_id=round_id, opened_at=0.0, deadline=1_000.0)


def _graded(round_id="r-1"):
    """A window carrying a verdict recorded while still FROZEN -- the state that must survive."""
    window = _window(round_id)
    window.submit("alice", paths=["SOUL.md"], payload_digest="sha256:" + "d" * 64, received_at=10.0)
    window.freeze(now=1_001.0)
    window.record_verdict(window.token(), "alice", passed=True, notes="withheld check passed")
    return window


def _store(tmp_path):
    return RoundStore(tmp_path / "rounds", require_private=False)


# --- the private snapshot carries what the public ledger drops ----------------------------------


def test_the_published_ledger_does_not_carry_the_verdicts_it_would_need_to_reload():
    """The reason a second record exists. Not a hypothetical: this is the record a reasonable
    person would have persisted."""
    window = _graded()
    ledger = window.to_record()
    assert "verdicts" not in ledger
    assert ledger["verdicts_recorded"] == 1, "the count is published; the verdicts are not"


def test_the_snapshot_carries_the_verdict_and_the_reload_can_publish_it():
    window = _graded()
    back = RoundWindow.from_snapshot(json.loads(json.dumps(window.snapshot())))
    assert len(back.verdicts) == 1

    back.grade(now=1_002.0)
    published = back.to_record()["verdicts"]
    assert len(published) == 1 and published[0]["passed"] is True


def test_the_publish_screen_refuses_the_snapshot():
    """The safety property, and the reason `snapshot` is safe to write to disk at all: the private
    record cannot go out through the same door as the public view even by mistake."""
    with pytest.raises(ScoreLeakError):
        screen_public_payload(_graded().snapshot(), allowed=PUBLIC_VIEW_FIELDS, where="snapshot")


def test_a_published_ledger_is_refused_as_a_snapshot_source():
    with pytest.raises(RoundError, match="not a round snapshot"):
        RoundWindow.from_snapshot(_graded().to_record())


def test_a_published_challenge_record_is_refused_as_a_snapshot_source():
    """The published packet has an `attempts` key too -- holding a *count*. So the obvious
    "attempts not in baseline" check passes on exactly the input it exists to reject, and the
    failure lands later as `'int' object is not iterable`. Found by feeding it a real record."""
    with pytest.raises(ChallengeError, match="published Challenge.to_record"):
        Challenge.from_snapshot(_challenge().to_record())


def test_the_baseline_attempts_survive_so_the_spread_is_not_silently_zero():
    """A challenge rebuilt from the published record would have no attempts, so `token_spread`
    would be 0.0 -- which does not read as missing data. It reads as a perfectly repeatable task,
    which is the strongest possible evidence that a miner's token margin is real."""
    challenge = _challenge()
    back = Challenge.from_snapshot(json.loads(json.dumps(challenge.snapshot())))
    assert len(back.baseline.attempts) == 10
    assert back.baseline.token_spread == challenge.baseline.token_spread > 0.0
    assert back.digest == challenge.digest


# --- the store ----------------------------------------------------------------------------------


def test_a_round_survives_a_save_and_load(tmp_path):
    store = _store(tmp_path)
    store.save(_graded("r-7"))
    back = store.load("r-7")
    assert back.round_id == "r-7"
    assert back.state == "frozen"
    assert len(back.submissions) == 1
    assert len(back.receipts) == 1
    # `token()` and not `token`: it is a method here, so `back.token is not None` would be true of
    # the bound method and would pass on a round whose token never came back at all.
    assert isinstance(back.token(), FreezeToken), "the token must return, or no verdict can be rebuilt"


def test_the_freeze_token_is_shared_with_the_verdicts_rather_than_copied(tmp_path):
    """`Verdict` refuses to exist without the `FreezeToken` its round minted -- which is what
    stops a verdict being fabricated for an open round. A second equal-but-separate token would
    satisfy the constructor while quietly breaking that identity, so the rebuild hands each
    verdict the window's own token."""
    store = _store(tmp_path)
    store.save(_graded("r-8"))
    back = store.load("r-8")
    assert all(v.token is back.token() for v in back.verdicts.values())


def test_load_all_reports_a_corrupt_round_instead_of_failing_the_whole_load(tmp_path):
    """This runs at startup. Raising would let one unparseable snapshot take the validator down
    and every healthy round with it -- converting a recoverable problem into an outage."""
    store = _store(tmp_path)
    store.save(_window("good"))
    (store.root / "broken.json").write_text("{not json", encoding="utf-8")

    loaded, failed = store.load_all()
    assert set(loaded) == {"good"}
    assert [round_id for round_id, _ in failed] == ["broken"]


def test_a_round_id_cannot_escape_the_store(tmp_path):
    """A round id arrives from a request in some deployments, and `Path / ".."` escapes without
    complaining."""
    store = _store(tmp_path)
    for bad in ("..", "a/b", ""):
        with pytest.raises(StoreError, match="unusable round id"):
            store.path_for(bad)


def test_saving_is_atomic_enough_that_a_reader_never_sees_a_half_file(tmp_path):
    """Written to a temp file and renamed. A crash inside `write_text` would otherwise leave JSON
    no reload can parse, and the round it described is then unrecoverable."""
    store = _store(tmp_path)
    store.save(_window("r-9"))
    assert not list(store.root.glob("*.tmp")), "the temp file must not survive a successful save"
    assert json.loads(store.path_for("r-9").read_text(encoding="utf-8"))["round_id"] == "r-9"


# --- the store must not be publishable ----------------------------------------------------------


def test_the_default_store_location_is_gitignored():
    """Asked of git, not asserted in a comment. A snapshot carries verdicts recorded before
    grading, so committing one publishes what the withheld half exists to withhold -- and the
    commit that reorganises ignore rules will look like housekeeping."""
    assert store_is_private() is True


def test_a_store_in_a_tracked_directory_is_refused():
    """Refused at construction rather than at the first save: a validator that has already graded
    a round and only then cannot store it has done the expensive part twice."""
    with pytest.raises(StoreError, match="not gitignored"):
        RoundStore(Path("datasets/rounds"))


# --- the API loads what the store holds ---------------------------------------------------------


def test_the_api_serves_a_stored_round_instead_of_404(tmp_path):
    """The whole point. Every endpoint was already correct and there was no path by which any of
    them could have had anything to serve."""
    pytest.importorskip("fastapi", reason="the validator API is an optional extra")
    from fastapi.testclient import TestClient

    from validator import api

    store = _store(tmp_path)
    store.save(_window("r-live", task_id="tc-log-rotation-order"))

    api.ROUNDS.clear()
    try:
        loaded, failed = api.load_from_store(store)
        assert (loaded, failed) == (1, [])
        client = TestClient(api.app)
        current = client.get("/v1/round/current")
        assert current.status_code == 200
        assert current.json()["round_id"] == "r-live"
        assert client.get("/v1/round/r-live/results").status_code == 409, "still no verdicts while OPEN"
    finally:
        api.ROUNDS.clear()


def test_the_api_reports_an_unreadable_round_rather_than_refusing_to_start(tmp_path):
    pytest.importorskip("fastapi", reason="the validator API is an optional extra")
    from validator import api

    store = _store(tmp_path)
    store.save(_window("fine"))
    (store.root / "wrecked.json").write_text("{", encoding="utf-8")

    api.ROUNDS.clear()
    try:
        loaded, failed = api.load_from_store(store)
        assert loaded == 1
        assert [round_id for round_id, _ in failed] == ["wrecked"]
    finally:
        api.ROUNDS.clear()


# --- the driver that opens a round over a packet -------------------------------------------------
#
# Nothing called `open_round` outside its own tests, so a live validator had nothing to serve and
# no way to be given anything. These drive the CLI's own functions against the packets and the
# baseline log actually committed to this repository.

REAL_PACKET = Path("datasets/challenges/tc-log-rotation-order.json")


def test_a_round_cannot_be_opened_from_the_packet_alone(tmp_path):
    """The packet publishes the baseline as aggregate statistics. A window built from it would have
    no attempts, so `token_spread` would be 0.0 -- which reads as a perfectly repeatable task, the
    strongest possible evidence that a miner's token margin is real. The efficiency gate would then
    accept margins it should refuse, so this is refused instead of guessed at."""
    from validator.round_loop import LoopError, challenge_from_packet_and_log

    if not REAL_PACKET.is_file():
        pytest.skip("no challenge packets in this checkout")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(LoopError, match="cannot reopen"):
        challenge_from_packet_and_log(REAL_PACKET, empty)


def test_opening_the_same_round_twice_is_refused(tmp_path):
    """Reopening resets a window miners may already have submitted to, and their receipts go with
    it."""
    from validator.round_loop import LoopError, open_from_packet

    store = _store(tmp_path)
    challenge = _challenge("t")
    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": "t",
                    "metrics": {
                        "task_id": "t",
                        "public_passed": False,
                        "tokens_used": a.tokens,
                        "tool_calls": a.tool_calls,
                        "wall_time_s": a.wall_time_s,
                        "steps": a.steps,
                        "max_steps_hit": True,
                    },
                }
            )
            for a in challenge.baseline.attempts
        )
        + "\n",
        encoding="utf-8",
    )
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(challenge.to_record()), encoding="utf-8")

    first = open_from_packet(packet_path=packet, episodes_path=log, round_id="r-x", hours=24.0, store=store, now=0.0)
    assert first.round_id == "r-x"
    assert len(first.challenge.baseline.attempts) == 10, "the log supplied the attempts"

    with pytest.raises(LoopError, match="already stored"):
        open_from_packet(packet_path=packet, episodes_path=log, round_id="r-x", hours=24.0, store=store, now=0.0)


def test_open_round_can_attach_an_assignment_so_scope_is_enforceable():
    """Without this parameter every window built through the documented constructor was permanently
    unscoped: `scope_enforced` reported False and the only way to change it was to assign to the
    field from outside, which is the operator discipline the class docstring says not to rely on."""

    class _Assignment:
        round_id = "r-seed"

        def owns(self, miner, task_id):
            return miner == "alice"

    unscoped = open_round(_challenge(), round_id="r-a", opened_at=0.0, deadline=1_000.0)
    scoped = open_round(_challenge(), round_id="r-b", opened_at=0.0, deadline=1_000.0, assignment=_Assignment())
    assert unscoped.scope_enforced is False
    assert scoped.scope_enforced is True

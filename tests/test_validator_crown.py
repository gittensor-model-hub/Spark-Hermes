"""Who holds a task's crown, and why almost nobody does.

The crown is a ratchet: once set it only rises, and every later challenger has to beat it. That
makes the refusals more important than the acceptances -- a crown awarded on thin evidence sets a
bar no honest strategy can clear and it does not decay.

The numbers here are the real ones. `carol`'s surface scored 3/3 on its first measurement and 6/10
on the next, which is exactly what the attempt floor predicted when it refused the first: 3/3
bounds the true rate only to 43.9%. Neither can hold a crown, and the second is the more
interesting refusal because it is a genuine improvement.
"""

import json

import pytest

from hermes.acceptance import Arm
from validator.crown import (
    CROWN_LABEL,
    Contender,
    Reign,
    contenders_from,
    contest,
    label_actions,
    pr_numbers,
    standings,
)

TASK = "tc-log-rotation-order"


def _arm(passes, attempts, tokens, calls=8):
    return Arm(
        passes=passes,
        attempts=attempts,
        tokens=tuple(tokens if isinstance(tokens, (list, tuple)) else [tokens] * attempts),
        tool_calls=tuple([calls] * attempts),
    )


def _c(miner, arm, *, task=TASK, round_id="r-1", pr=0):
    return Contender(task_id=task, miner_id=miner, round_id=round_id, arm=arm, pr=pr)


# --- an empty throne is not a lower bar ------------------------------------------------------------


def test_a_first_arrival_still_has_to_clear_the_bar():
    """A crown handed to whoever went first means "went first"."""
    took, why = contest(None, _c("carol", _arm(6, 10, 59_024)))
    assert took is False
    assert "must still pass every attempt" in why


def test_a_perfect_run_of_too_few_attempts_cannot_open_a_reign():
    """The real sequence: 3/3 on the first measurement, 6/10 on the next. The floor was right."""
    took, why = contest(None, _c("dave", _arm(3, 3, 50_000)))
    assert took is False
    assert "43.9%" in why and "does not decay" in why


def test_a_clean_run_at_the_floor_opens_a_reign():
    took, why = contest(None, _c("erin", _arm(10, 10, 60_000)))
    assert took is True and why == ""


# --- taking a crown from someone --------------------------------------------------------------------


def test_an_equal_challenger_does_not_take_the_crown():
    """Ties keep the incumbent, so a crown changes hands only on evidence."""
    king = _c("erin", _arm(10, 10, 60_000))
    took, why = contest(king, _c("frank", _arm(10, 10, 60_000)))
    assert took is False
    assert "does not survive the noise" in why


def test_a_genuine_improvement_takes_the_crown():
    king = _c("erin", _arm(10, 10, 60_000))
    took, _ = contest(king, _c("grace", _arm(10, 10, 40_000, calls=6)))
    assert took is True


def test_the_crown_does_not_trade_tool_calls_for_tokens():
    """A bounty on tokens alone pays for collapsing thirty operations into one helper call, which
    is why `dominates` requires both."""
    king = _c("erin", _arm(10, 10, 60_000, calls=8))
    took, why = contest(king, _c("heidi", _arm(10, 10, 30_000, calls=20)))
    assert took is False
    assert "tool calls" in why


def test_a_challenger_that_fails_one_attempt_cannot_take_it():
    king = _c("erin", _arm(10, 10, 60_000))
    took, why = contest(king, _c("ivan", _arm(9, 10, 20_000, calls=4)))
    assert took is False
    assert "every attempt" in why


# --- standings --------------------------------------------------------------------------------------


def test_the_result_does_not_depend_on_the_order_contenders_arrive():
    """A ratchet whose outcome changes with iteration order is not a ratchet."""
    people = [
        _c("erin", _arm(10, 10, 60_000), round_id="r-1"),
        _c("grace", _arm(10, 10, 40_000, calls=6), round_id="r-2"),
        _c("dave", _arm(3, 3, 10_000), round_id="r-3"),
    ]
    first, _ = standings(people)
    second, _ = standings(list(reversed(people)))
    assert first[TASK].miner_id == second[TASK].miner_id == "grace"


def test_every_refusal_is_returned_with_its_reason():
    kings, refused = standings([_c("dave", _arm(3, 3, 10_000)), _c("erin", _arm(10, 10, 60_000))])
    assert kings[TASK].miner_id == "erin"
    assert [c.miner_id for c, _ in refused] == ["dave"]
    assert "too few attempts" in refused[0][1]


def test_crowns_are_per_task():
    """One leaderboard would rank miners by which task they drew: the baselines are not comparable."""
    kings, _ = standings(
        [
            _c("erin", _arm(10, 10, 60_000), task="a"),
            _c("grace", _arm(10, 10, 90_000), task="b"),
        ]
    )
    assert {t: r.miner_id for t, r in kings.items()} == {"a": "erin", "b": "grace"}


def test_no_crown_is_reported_as_no_crown():
    kings, refused = standings([_c("carol", _arm(6, 10, 59_024))])
    assert kings == {}
    assert len(refused) == 1


# --- the label ----------------------------------------------------------------------------------------


def test_a_new_king_removes_the_old_label_and_adds_the_new():
    """A crown only ever added is a crown several people hold at once."""
    kings = {TASK: Reign(TASK, "grace", "r-2", 40_000, 6, 10, pr=22)}
    actions = label_actions(kings, {"crowns": {TASK: {"pr": 11}}})
    assert ("remove", 11, TASK) in actions
    assert ("add", 22, TASK) in actions


def test_an_unchanged_king_moves_nothing():
    """Otherwise the hourly job re-notifies the same miner every hour."""
    kings = {TASK: Reign(TASK, "grace", "r-2", 40_000, 6, 10, pr=22)}
    assert label_actions(kings, {"crowns": {TASK: {"pr": 22}}}) == []


def test_a_vanished_crown_has_its_label_removed():
    """A round rolled back or a scorecard withdrawn leaves a label asserting something no longer
    computed."""
    actions = label_actions({}, {"crowns": {TASK: {"pr": 11}}})
    assert actions == [("remove", 11, TASK)]


def test_a_king_with_no_recorded_pull_request_is_not_labelled():
    """Guessing which pull request belongs to a miner is how the wrong one gets labelled."""
    kings = {TASK: Reign(TASK, "grace", "r-2", 40_000, 6, 10, pr=0)}
    assert label_actions(kings, {"crowns": {}}) == []


def test_pull_request_numbers_come_from_the_registry(tmp_path):
    registry = tmp_path / "strategies.jsonl"
    registry.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"round_id": "r-1", "miner_id": "erin", "pr": 11},
                {"round_id": "r-2", "miner_id": "grace"},
                "not json",
            )
            if isinstance(r, dict)
        )
        + "\nnot json\n",
        encoding="utf-8",
    )
    found = pr_numbers(registry)
    assert found == {("r-1", "erin"): 11}, "a line with no pr and a malformed line are both skipped"


def test_a_missing_registry_is_not_an_error(tmp_path):
    assert pr_numbers(tmp_path / "absent.jsonl") == {}


# --- eligibility ------------------------------------------------------------------------------------


def test_only_graded_rounds_are_eligible(tmp_path):
    """A verdict is not published until grading. Crowning on an ungraded round would rank miners on
    a result the round itself refuses to serve."""
    from hermes.challenge import Attempt, Baseline, open_challenge
    from hermes.round import open_round
    from validator.store import RoundStore

    store = RoundStore(tmp_path / "rounds", require_private=False)
    attempts = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=78_000 + i * 900,
            tool_calls=11,
            wall_time_s=1.0,
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
    window = open_round(challenge, round_id="r-open", opened_at=0.0, deadline=1_000.0)
    # Submit before grading. `record_verdict` refuses a miner with no standing submission -- "a
    # verdict for a miner who never submitted inflates the denominator of every rate the round
    # reports" -- and the first version of this test skipped straight to the verdict.
    window.submit("erin", paths=["SOUL.md"], payload_digest="sha256:" + "d" * 64, received_at=10.0)
    store.save(window)

    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "r-open-erin.json").write_text(
        json.dumps(
            {
                "round_id": "r-open",
                "miner_id": "erin",
                "task_id": TASK,
                "candidate": {"verified_passes": 10, "attempts": 10, "tokens": [60_000] * 10},
            }
        ),
        encoding="utf-8",
    )
    assert contenders_from(cards, store=store) == [], "an OPEN round yields no contender"

    window.freeze(now=2_000.0)
    window.record_verdict(window.token(), "erin", passed=True)
    window.grade(now=2_001.0)
    store.save(window)
    assert len(contenders_from(cards, store=store)) == 1


def test_a_scorecard_for_a_missing_round_is_skipped(tmp_path):
    from validator.store import RoundStore

    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "gone-erin.json").write_text(
        json.dumps({"round_id": "gone", "miner_id": "erin", "task_id": TASK, "candidate": {"tokens": [1]}}),
        encoding="utf-8",
    )
    assert contenders_from(cards, store=RoundStore(tmp_path / "rounds", require_private=False)) == []


def test_the_crown_label_name_is_stable():
    """It is written into a workflow and onto pull requests; renaming it orphans every label
    already applied."""
    assert CROWN_LABEL == "crown"


@pytest.mark.parametrize("state", ["graded", "settled"])
def test_both_published_states_are_eligible(state):
    """`grade` publishes the verdicts and `settle` releases the salt. A crown depends on the first,
    so a settled round must not become ineligible by having gone further."""
    from hermes.round import GRADED, SETTLED

    assert state in (GRADED, SETTLED)


def test_a_scorecard_carries_tool_calls_so_the_crown_guard_can_fire(tmp_path):
    """Found by wiring the crown to real scorecards. `Scorecard.to_record` omitted `tool_calls`, so
    `contenders_from` defaulted them to zeros and `dominates`' tool-call guard -- the one stopping a
    challenger from buying a token win by collapsing thirty operations into one helper call --
    compared 0 against 0 and passed every time.

    The comparison is driven through a serialised scorecard rather than a hand-built `Arm`, because
    a hand-built one has the field by construction and would never have caught this.
    """
    from hermes.acceptance import Decision
    from validator.score import Scorecard

    def card(miner, tokens, calls):
        return Scorecard(
            round_id="r-1",
            miner_id=miner,
            task_id=TASK,
            candidate=Arm(passes=10, attempts=10, tokens=tuple([tokens] * 10), tool_calls=tuple([calls] * 10)),
            baseline=Arm(passes=4, attempts=10, tokens=tuple([78_000] * 10), tool_calls=tuple([11] * 10)),
            decision=Decision(accepted=True, reasons=()),
            interval=(0.1, 0.4),
            repeats_to_settle=None,
            overfit_attempts=0,
            protocol_failures=0,
        ).to_record()

    assert "tool_calls" in card("erin", 60_000, 8)["candidate"]

    def contender(record):
        candidate = record["candidate"]
        return Contender(
            task_id=record["task_id"],
            miner_id=record["miner_id"],
            round_id=record["round_id"],
            arm=Arm(
                passes=candidate["verified_passes"],
                attempts=candidate["attempts"],
                tokens=tuple(candidate["tokens"]),
                tool_calls=tuple(candidate["tool_calls"]),
            ),
        )

    king = contender(card("erin", 60_000, 8))
    greedy = contender(card("heidi", 30_000, 20))
    took, why = contest(king, greedy)
    assert took is False and "tool calls" in why

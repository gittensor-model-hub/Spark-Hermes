"""One crown an hour: score the field, keep one, close the rest.

Each hour is a tournament rather than a standing record. The design pressure is that a single
winner has to be chosen across tasks whose bars are not comparable, and that an hourly reward which
always pays out stops carrying information. Most of what follows is about those two.

The numbers are real where they can be. `carol`'s surface scored 3/3 on its first measurement and
6/10 on ten episodes, which is why perfect-but-few and improved-but-imperfect are both tested as
refusals -- they are the two shapes a promising submission actually took.
"""

import json

import pytest

from hermes.acceptance import Arm
from validator.crown import (
    CROWN_LABEL,
    Contender,
    Standing,
    close_actions,
    contenders_from,
    eligibility,
    next_task,
    pr_numbers,
    render,
    score_of,
    select,
    settle,
)

TASK = "tc-log-rotation-order"
BASE = [78_000 + i * 1_500 for i in range(10)]


def _arm(passes, attempts, tokens, calls=8):
    tokens = list(tokens) if isinstance(tokens, (list, tuple)) else [tokens] * attempts
    return Arm(passes=passes, attempts=attempts, tokens=tuple(tokens), tool_calls=tuple([calls] * attempts))


def _c(miner, candidate, *, task=TASK, base=None, pr=0, at=0.0, round_id="r-1"):
    return Contender(
        task_id=task,
        miner_id=miner,
        round_id=round_id,
        candidate=candidate,
        baseline=_arm(4, 10, base or BASE, 11),
        pr=pr,
        received_at=at,
    )


# --- eligibility is a gate, not a term ------------------------------------------------------------


def test_a_cheaper_wrong_answer_is_not_ranked_at_all():
    """Without this the hourly cadence becomes "cheapest wrong answer wins": correctness has to be
    a gate before tokens are looked at, exactly as `acceptance.decide` orders it."""
    ok, why = eligibility(_c("heidi", _arm(6, 10, 20_000, calls=4)))
    assert ok is False
    assert "before tokens are looked at" in why


def test_a_perfect_run_of_too_few_attempts_is_not_ranked():
    """The real shape: 3/3 first, 6/10 on ten episodes."""
    ok, why = eligibility(_c("dave", _arm(3, 3, 30_000)))
    assert ok is False and "43.9%" in why


def test_a_baseline_with_one_measurement_cannot_bound_a_reduction():
    thin = Contender(TASK, "erin", "r-1", _arm(10, 10, 50_000), _arm(1, 1, [78_000]))
    ok, why = eligibility(thin)
    assert ok is False and "fewer than two measurements" in why


def test_a_correct_submission_over_the_floor_is_ranked():
    assert eligibility(_c("erin", _arm(10, 10, 59_000)))[0] is True


# --- ranking across tasks -----------------------------------------------------------------------


def test_the_score_is_relative_to_each_submissions_own_baseline():
    """Absolute tokens would rank miners by which task they drew."""
    small_bar = _c("erin", _arm(10, 10, [59_000 + i * 900 for i in range(10)]))
    big_bar = _c(
        "grace",
        _arm(10, 10, [160_000 + i * 2_000 for i in range(10)]),
        task="task-b",
        base=[210_000 + i * 4_000 for i in range(10)],
    )
    assert big_bar.median_tokens > small_bar.median_tokens
    assert score_of(big_bar).lower_bound > 0 and score_of(small_bar).lower_bound > 0


def test_a_noisy_large_margin_does_not_outrank_a_tight_smaller_one():
    """The reason the score is the lower bound and not the point estimate: a large margin on a
    wildly variable task is not more known than a smaller one on a tight task -- it is less."""
    tight = _c("erin", _arm(10, 10, [59_000 + i * 400 for i in range(10)]))
    noisy = _c(
        "grace",
        _arm(10, 10, [20_000 + i * 26_000 for i in range(10)]),
        task="task-b",
        base=[210_000 + i * 3_000 for i in range(10)],
    )
    from statistics import median

    nominal = (median(noisy.baseline.tokens) - median(noisy.candidate.tokens)) / median(noisy.baseline.tokens)
    assert nominal > 0.25, "the noisy entry has the larger point estimate"
    outcome = select([tight, noisy])
    assert outcome.winner is not None
    assert outcome.winner.contender.miner_id == "erin"


def test_the_full_ordering_is_published_not_just_the_winner():
    """A miner has to be able to see where they placed, or an hourly loss carries no information."""
    outcome = select([_c("erin", _arm(10, 10, 59_000)), _c("frank", _arm(10, 10, 70_000))])
    assert [s.contender.miner_id for s in outcome.ranked] == ["erin", "frank"]


def test_ties_break_on_tool_calls_then_on_arriving_first():
    """Deterministic, and it rewards the earlier of two identical results rather than the later."""
    early = _c("erin", _arm(10, 10, 59_000, calls=6), at=10.0)
    late = _c("frank", _arm(10, 10, 59_000, calls=6), at=20.0)
    chatty = _c("grace", _arm(10, 10, 59_000, calls=20), at=5.0)
    outcome = select([late, chatty, early])
    assert outcome.winner is not None and outcome.winner.contender.miner_id == "erin"


# --- an hour can have no winner -------------------------------------------------------------------


def test_a_field_that_is_correct_but_not_cheaper_has_no_winner():
    """The crown is for an improvement, not for turning up. An hourly reward that always pays out
    stops carrying information within a week."""
    outcome = select([_c("frank", _arm(10, 10, [79_000 + i * 1_500 for i in range(10)]))])
    assert outcome.winner is None
    assert any("does not clear" in why for _, why in outcome.ineligible)
    assert "NO CROWN" in render(outcome)


def test_an_empty_field_has_no_winner():
    outcome = select([])
    assert outcome.winner is None and outcome.ranked == []
    assert outcome.to_record()["crowned"] is False


def test_the_record_distinguishes_no_entries_from_no_winner():
    """An hour with nothing submitted and an hour whose field was ranked and found wanting are
    different, and only the second says something about the miners."""
    nothing = select([]).to_record()
    lost = select([_c("frank", _arm(10, 10, BASE))]).to_record()
    assert nothing["ranked"] == [] and nothing["ineligible"] == []
    assert lost["ranked"] and lost["crowned"] is False


# --- the label and the closures ---------------------------------------------------------------------


def test_every_pull_request_but_the_winners_is_closed():
    field = [
        _c("erin", _arm(10, 10, 59_000), pr=21),
        _c("frank", _arm(10, 10, 70_000), pr=22),
        _c("heidi", _arm(6, 10, 20_000), pr=23),
    ]
    outcome = select(field)
    assert outcome.winner is not None and outcome.winner.contender.pr == 21
    assert sorted(pr for pr, _ in close_actions(outcome)) == [22, 23]


def test_each_closure_carries_the_reason_it_lost():
    """A pull request closed with no reason is a miner who learns nothing and resubmits the same
    thing."""
    outcome = select([_c("erin", _arm(10, 10, 59_000), pr=21), _c("heidi", _arm(6, 10, 20_000), pr=23)])
    reasons = dict(close_actions(outcome))
    assert "every attempt" in reasons[23]


def test_challengers_close_on_a_winnerless_hour_and_the_incumbent_does_not():
    """This replaced a flag. When a barren hour left the crown vacant, whether to close the field
    was a policy choice; now the crown never vacates, so the answer follows from it -- challengers
    close, the standing holder does not."""
    outcome = select([_c("frank", _arm(10, 10, BASE), pr=22)])
    assert outcome.winner is None
    closed = [pr for pr, _ in close_actions(outcome, Standing.from_record({"winner": {"pr": 21}}))]
    assert closed == [22]


def test_a_submission_with_no_pull_request_number_is_not_closed():
    """Guessing which pull request belongs to a miner is how the wrong one gets closed, and closing
    is not reversible by this job."""
    outcome = select([_c("erin", _arm(10, 10, 59_000), pr=21), _c("frank", _arm(10, 10, 70_000), pr=0)])
    assert [pr for pr, _ in close_actions(outcome)] == []


# --- reading the field --------------------------------------------------------------------------------


def test_pull_request_numbers_and_arrival_come_from_the_registry(tmp_path):
    registry = tmp_path / "strategies.jsonl"
    registry.write_text(
        json.dumps({"round_id": "r-1", "miner_id": "erin", "pr": 11, "received_at": 5.0})
        + "\n"
        + json.dumps({"round_id": "r-1", "miner_id": "grace"})
        + "\nnot json\n",
        encoding="utf-8",
    )
    assert pr_numbers(registry) == {("r-1", "erin"): {"pr": 11, "received_at": 5.0}}


def test_a_missing_registry_is_not_an_error(tmp_path):
    assert pr_numbers(tmp_path / "absent.jsonl") == {}


def test_only_graded_rounds_are_eligible(tmp_path):
    """A verdict is not published until grading; crowning on an ungraded round would rank miners on
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
    # `record_verdict` refuses a miner with no standing submission: a verdict for someone who never
    # submitted inflates the denominator of every rate the round reports.
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
                "candidate": {"verified_passes": 10, "attempts": 10, "tokens": [59_000] * 10, "tool_calls": [8] * 10},
                "baseline": {"verified_passes": 4, "attempts": 10, "tokens": BASE, "tool_calls": [11] * 10},
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


def test_a_scorecard_without_a_baseline_is_skipped(tmp_path):
    """No bar, no relative score. Ranking it would compare an absolute token count against a set of
    reductions."""
    from validator.store import RoundStore

    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "x.json").write_text(
        json.dumps({"round_id": "gone", "miner_id": "erin", "task_id": TASK, "candidate": {"tokens": [1, 2]}}),
        encoding="utf-8",
    )
    assert contenders_from(cards, store=RoundStore(tmp_path / "r", require_private=False)) == []


def test_the_crown_label_name_is_stable():
    """It is written into a workflow and onto pull requests; renaming it orphans every label already
    applied."""
    assert CROWN_LABEL == "crown"


def test_a_scorecard_carries_tool_calls_so_ties_can_break_on_them():
    """Found by wiring this to real scorecards: `Scorecard.to_record` omitted `tool_calls`, so
    everything read zeros and the tie-break compared 0 against 0. Driven through a serialised
    scorecard, because a hand-built `Arm` has the field by construction."""
    from hermes.acceptance import Decision
    from validator.score import Scorecard

    record = Scorecard(
        round_id="r-1",
        miner_id="erin",
        task_id=TASK,
        candidate=Arm(passes=10, attempts=10, tokens=tuple([59_000] * 10), tool_calls=tuple([6] * 10)),
        baseline=Arm(passes=4, attempts=10, tokens=tuple(BASE), tool_calls=tuple([11] * 10)),
        decision=Decision(accepted=True, reasons=()),
        interval=(0.1, 0.4),
        repeats_to_settle=None,
        overfit_attempts=0,
        protocol_failures=0,
    ).to_record()
    assert record["candidate"]["tool_calls"] == [6] * 10
    assert record["baseline"]["tool_calls"] == [11] * 10


@pytest.mark.parametrize("state", ["graded", "settled"])
def test_both_published_states_are_eligible(state):
    from hermes.round import GRADED, SETTLED

    assert state in (GRADED, SETTLED)


def test_a_losing_pull_request_is_closed_exactly_once():
    """A contender appears on both lists when the best of the field does not clear zero: ranked,
    and recorded as ineligible with the reason. Closing it twice reads as two miners losing when it
    is one."""
    outcome = select([_c("frank", _arm(10, 10, BASE), pr=22)])
    assert outcome.winner is None
    assert [pr for pr, _ in close_actions(outcome)] == [22]
    assert [c.miner_id for c in outcome.losers] == ["frank"]


# --- what carries from one hour to the next ---------------------------------------------------------


TASKS = ["task-a", "task-b", "task-c"]


def _standing(winner=None, task="task-a", barren=0):
    return {"winner": winner, "task_id": task, "barren_rounds": barren}


def test_a_winner_takes_the_crown_and_resets_the_counter():
    outcome = select([_c("erin", _arm(10, 10, 59_000), pr=21)])
    standing, labels = settle(_standing(barren=1), outcome, available_tasks=TASKS)
    assert labels == [("add", 21)]
    assert standing.pr == 21 and standing.barren_rounds == 0
    assert standing.task_id == "task-a", "someone beat it, so it is a task worth running again"


def test_a_winner_dethrones_the_previous_holder():
    outcome = select([_c("grace", _arm(10, 10, 50_000), pr=24)])
    standing, labels = settle(_standing(winner={"pr": 21}), outcome, available_tasks=TASKS)
    assert labels == [("remove", 21), ("add", 24)]
    assert standing.pr == 24


def test_a_barren_hour_leaves_the_crown_where_it_is():
    """A barren hour is a fact about this hour's field, not about the miner who last cleared the
    bar. Nothing is removed and nothing is added, so the holder is not re-notified either."""
    outcome = select([_c("frank", _arm(10, 10, BASE), pr=22)])
    assert outcome.winner is None
    standing, labels = settle(_standing(winner={"pr": 21}), outcome, available_tasks=TASKS)
    assert labels == []
    assert standing.pr == 21
    assert standing.barren_rounds == 1
    assert standing.task_id == "task-a", "one barren hour is not evidence about the task"


def test_two_barren_hours_rotate_the_task():
    """Evidence about the task rather than the field: either nobody can beat its baseline or nobody
    is trying, and a third run spends an hour of everyone's GPU time to learn the same thing."""
    standing, labels = settle(_standing(winner={"pr": 21}, barren=1), select([]), available_tasks=TASKS)
    assert standing.task_id == "task-b"
    assert standing.barren_rounds == 0
    assert labels == [], "the rotation does not touch the crown"


def test_the_crown_carries_across_a_rotation():
    """It was won and nothing has taken it. Stripping it because the subject changed would punish
    the holder for other people's failure."""
    standing, _ = settle(_standing(winner={"pr": 21}, barren=1), select([]), available_tasks=TASKS)
    assert standing.pr == 21


def test_rotation_cycles_and_is_deterministic():
    """A rotation nobody can predict is a rotation nobody can prepare for."""
    assert next_task("task-a", TASKS) == "task-b"
    assert next_task("task-c", TASKS) == "task-a"
    assert next_task("unknown", TASKS) == "task-a"
    assert next_task("task-a", []) == "task-a", "nothing to rotate to leaves it alone"


def test_a_barren_hour_with_no_incumbent_still_counts():
    """Otherwise a task with no crown yet never rotates, and the first task runs forever."""
    standing, _ = settle(_standing(), select([]), available_tasks=TASKS)
    assert standing.barren_rounds == 1 and standing.pr == 0


def test_the_crowned_pull_request_is_not_closed_while_it_holds_the_crown():
    """It is the standing result; closing it would leave the label pointing at a closed page."""
    outcome = select([_c("frank", _arm(10, 10, BASE), pr=22)])
    spared = Standing.from_record(_standing(winner={"pr": 21}))
    closed = [pr for pr, _ in close_actions(outcome, spared)]
    assert 21 not in closed
    assert closed == [22]


def test_this_hours_winner_is_not_closed_either():
    outcome = select([_c("erin", _arm(10, 10, 59_000), pr=21), _c("frank", _arm(10, 10, 70_000), pr=22)])
    closed = [pr for pr, _ in close_actions(outcome, Standing())]
    assert closed == [22]


def test_four_hours_end_to_end():
    """The sequence the design describes, walked once: win, barren, barren-and-rotate, win again."""
    tasks = TASKS
    s1, l1 = settle({}, select([_c("erin", _arm(10, 10, 59_000), pr=21)]), available_tasks=tasks)
    assert l1 == [("add", 21)] and s1.barren_rounds == 0

    s2, l2 = settle(_standing(s1.winner, s1.task_id, s1.barren_rounds), select([]), available_tasks=tasks)
    assert l2 == [] and s2.pr == 21 and s2.barren_rounds == 1 and s2.task_id == s1.task_id

    s3, l3 = settle(_standing(s2.winner, s2.task_id, s2.barren_rounds), select([]), available_tasks=tasks)
    assert l3 == [] and s3.pr == 21 and s3.barren_rounds == 0 and s3.task_id != s2.task_id

    grace = _c("grace", _arm(10, 10, 50_000), pr=24, task=s3.task_id)
    s4, l4 = settle(_standing(s3.winner, s3.task_id, s3.barren_rounds), select([grace]), available_tasks=tasks)
    assert l4 == [("remove", 21), ("add", 24)] and s4.pr == 24


def test_available_tasks_comes_from_the_published_packets(tmp_path):
    """A round can only be opened over a task with a challenge packet, so the rotation cannot land
    somewhere there is nothing to run."""
    from validator.crown import available_tasks

    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert available_tasks(tmp_path) == ["a", "b"]
    assert available_tasks(tmp_path / "absent") == []

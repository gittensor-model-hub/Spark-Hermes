"""The acceptance bar: correctness first, then efficiency, and only above the noise."""

import pytest

from hermes.acceptance import (
    MIN_ATTEMPTS,
    MIN_TOKEN_REDUCTION,
    AcceptanceError,
    Arm,
    decide,
    dominates,
)


def _arm(passes, attempts, tokens, calls):
    return Arm(passes=passes, attempts=attempts, tokens=tuple(tokens), tool_calls=tuple(calls))


def _steady(n, token, call):
    """n attempts, all passing, with a tight spread so noise is not the thing under test."""
    return _arm(n, n, [token] * n, [call] * n)


BASELINE = _steady(MIN_ATTEMPTS, 10_000, 20)


# --- gate 1: correctness comes first, and is reported before anything is computed -----------


def test_a_failed_attempt_stops_the_judgement_before_tokens_are_quoted():
    """Reporting a token reduction beside a wrong answer invites someone to quote the number."""
    candidate = _arm(MIN_ATTEMPTS - 1, MIN_ATTEMPTS, [3_000] * MIN_ATTEMPTS, [5] * MIN_ATTEMPTS)
    d = decide(candidate=candidate, baseline=BASELINE)
    assert d.accepted is False
    assert d.token_reduction is None
    assert "before efficiency is considered" in d.reasons[0]


def test_one_hundred_percent_of_one_attempt_is_refused_with_the_bound_it_implies():
    """1/1 bounds the true rate only to 20.7%: a model that truly fails four times in five
    can show 100%."""
    d = decide(candidate=_steady(1, 3_000, 5), baseline=BASELINE)
    assert d.accepted is False
    assert "statement about luck" in d.reasons[0]
    assert d.true_rate_lower_bound is not None and d.true_rate_lower_bound < 0.25


def test_five_of_five_is_still_too_few():
    """5/5 bounds the true rate to 56.6% -- a model failing two in five can show it."""
    d = decide(candidate=_steady(5, 3_000, 5), baseline=BASELINE)
    assert d.accepted is False
    assert d.true_rate_lower_bound is not None and 0.5 < d.true_rate_lower_bound < 0.6


def test_enough_attempts_carries_a_usable_bound():
    d = decide(candidate=_steady(MIN_ATTEMPTS, 3_000, 5), baseline=BASELINE)
    assert d.accepted is True
    assert d.true_rate_lower_bound is not None and d.true_rate_lower_bound > 0.7


# --- gate 2: not cheaper by checking less ----------------------------------------------------


def test_a_verification_regression_blocks_acceptance():
    d = decide(
        candidate=_steady(MIN_ATTEMPTS, 3_000, 5),
        baseline=BASELINE,
        verification_ok=False,
        verification_reason="verified less than the baseline on demonstrated_repairs",
    )
    assert d.accepted is False
    assert "demonstrated_repairs" in d.reasons[0]
    assert d.token_reduction is None


# --- gate 3: the margin must beat the observed spread, not just the constant -----------------


def test_a_reduction_below_the_stated_bar_is_refused():
    d = decide(candidate=_steady(MIN_ATTEMPTS, 9_000, 18), baseline=BASELINE)  # 10%
    assert d.accepted is False
    assert "below the 20% bar" in d.reasons[0]


def test_a_reduction_that_meets_the_bar_but_loses_to_the_noise_is_refused():
    """The interesting refusal: the point estimate clears the bar and the interval's lower
    bound does not, so the margin cannot be told apart from sampling noise."""
    noisy_baseline = _arm(MIN_ATTEMPTS, MIN_ATTEMPTS, [7_000, 10_000, 13_000] * 4, [20] * 12)
    d = decide(candidate=_steady(MIN_ATTEMPTS, 8_000, 10), baseline=noisy_baseline)
    assert d.accepted is False
    assert "cannot be distinguished from noise" in d.reasons[0]
    # Says what to do about it. At high spread the answer is more repeats, not a bigger
    # margin: the interval narrows with n, the raw spread does not.
    assert "more paired repeats" in d.reasons[0]


def test_a_margin_that_clears_the_spread_is_accepted():
    noisy_baseline = _arm(MIN_ATTEMPTS, MIN_ATTEMPTS, [9_500, 10_000, 10_500] * 4, [20] * 12)
    d = decide(candidate=_steady(MIN_ATTEMPTS, 5_000, 8), baseline=noisy_baseline)
    assert d.accepted is True
    assert d.token_reduction == 0.5


def test_a_single_measurement_arm_cannot_bound_a_reduction():
    """One run tells you nothing about its own variability. Reported as its own reason rather
    than as an infinite interval: "there was no second observation" and "the data was noisy"
    call for different actions, and the second misdescribes the first."""
    d = decide(candidate=_steady(MIN_ATTEMPTS, 1_000, 2), baseline=_arm(1, 1, [10_000], [20]))
    assert d.accepted is False
    assert "too few token measurements" in d.reasons[0]
    assert "no information about its own variability" in d.reasons[0]


# --- tool calls are reported, never a purse ---------------------------------------------------


def test_tool_call_reduction_is_reported_and_labelled_as_not_a_bounty():
    """A bounty on tool calls pays for the one metric a helper script trivially collapses:
    thirty operations become one call, and the bounty makes that the best move on the board."""
    d = decide(candidate=_steady(MIN_ATTEMPTS, 5_000, 4), baseline=BASELINE)
    assert d.tool_call_reduction == 16
    assert d.to_record()["tool_calls_are_a_reported_dimension_not_a_bounty"] is True


def test_tool_calls_alone_cannot_carry_a_submission():
    """Same tokens, far fewer calls: refused, because the calls could be one helper script."""
    d = decide(candidate=_steady(MIN_ATTEMPTS, 10_000, 2), baseline=BASELINE)
    assert d.accepted is False
    assert d.tool_call_reduction == 18


# --- the crown ratchets only on what is deterministic -----------------------------------------


def test_the_crown_needs_domination_on_tokens_and_tool_calls():
    incumbent = _steady(MIN_ATTEMPTS, 4_100, 6)
    challenger = _steady(MIN_ATTEMPTS, 3_100, 4)
    assert dominates(challenger, incumbent)[0] is True


def test_a_challenger_that_trades_calls_for_tokens_does_not_take_the_crown():
    incumbent = _steady(MIN_ATTEMPTS, 4_100, 6)
    challenger = _steady(MIN_ATTEMPTS, 3_100, 9)
    ok, reason = dominates(challenger, incumbent)
    assert ok is False
    assert "does not dominate" in reason


def test_an_equal_challenger_does_not_take_the_crown():
    incumbent = _steady(MIN_ATTEMPTS, 4_100, 6)
    assert dominates(incumbent, incumbent)[0] is False


def test_a_challenger_that_fails_an_attempt_cannot_hold_the_crown():
    incumbent = _steady(MIN_ATTEMPTS, 4_100, 6)
    flaky = _arm(MIN_ATTEMPTS - 1, MIN_ATTEMPTS, [1] * MIN_ATTEMPTS, [1] * MIN_ATTEMPTS)
    ok, reason = dominates(flaky, incumbent)
    assert ok is False
    assert "every attempt" in reason


def test_a_single_attempt_cannot_take_the_crown():
    """The crown persists and every later challenger must beat it, so a bar set from one
    lucky sample is worse than a one-off acceptance from one: it does not decay."""
    ok, reason = dominates(_steady(1, 1_000, 2), _steady(MIN_ATTEMPTS, 4_100, 6))
    assert ok is False
    assert "20.7%" in reason and "the crown persists" in reason


def test_a_challenger_cannot_be_crowned_against_a_one_attempt_incumbent():
    """Otherwise the floor is trivially bypassed from the other side: stand up a weak
    incumbent on one run, then beat it."""
    ok, reason = dominates(_steady(MIN_ATTEMPTS, 1_000, 2), _steady(1, 4_100, 6))
    assert ok is False
    assert "incumbent" in reason


def test_the_two_gates_agree_about_what_counts_as_evidence():
    """The defect this closes: dominates() accepted an arm that decide() refused, and the
    asymmetry ran the wrong way -- the persistent gate was the lenient one."""
    thin = _steady(1, 3_000, 5)
    assert decide(candidate=thin, baseline=BASELINE).accepted is False
    assert dominates(thin, BASELINE)[0] is False


def test_the_floor_is_overridable_so_a_calibration_run_can_lower_it():
    """The baseline run has to be able to establish a first incumbent before ten repeats of
    everything exist. Explicit parameter, not a silent default."""
    assert dominates(_steady(3, 1_000, 2), _steady(3, 4_100, 6), min_attempts=3)[0] is True


def test_latency_is_absent_from_the_crown_by_construction():
    """A bar that only rises would lock in whichever run got favourable scheduling,
    permanently, because no later run could legitimately beat it."""
    import inspect

    from hermes import acceptance

    assert "latency" not in inspect.getsource(acceptance.dominates).replace("latency is", "")
    assert not hasattr(Arm, "wall_time")


# --- malformed input -------------------------------------------------------------------------


def test_more_passes_than_attempts_is_refused():
    with pytest.raises(AcceptanceError, match="passes out of"):
        _arm(5, 3, [1], [1])


def test_an_arm_with_no_attempts_is_refused():
    with pytest.raises(AcceptanceError, match="no attempts"):
        _arm(0, 0, [], [])


def test_a_baseline_with_no_tokens_is_refused():
    with pytest.raises(AcceptanceError, match="nothing to improve on"):
        decide(candidate=_steady(MIN_ATTEMPTS, 1, 1), baseline=_steady(MIN_ATTEMPTS, 0, 5))


# --- the gate must be satisfiable, which the spread multiple was not -------------------------


def test_a_large_win_on_a_high_spread_task_is_no_longer_impossible():
    """The regression this replaces. `max(0.20, 2.0 * spread)` demanded 196.6% of
    migrate-and-keep-green, whose measured spread was 98.3% -- and a reduction cannot exceed
    100%, so no candidate could ever clear it. Verified at the time: even a 99.9% reduction was
    refused. The noisiest tasks were silently removed from the competition."""
    noisy = tuple([61_798] * 4 + [1_200, 122_000] + [61_798] * 4)
    baseline = Arm(passes=10, attempts=10, tokens=noisy, tool_calls=(24,) * 10)
    assert baseline.token_spread > 0.9, "this fixture must reproduce the high-spread case"

    candidate = Arm(passes=10, attempts=10, tokens=(30_000,) * 10, tool_calls=(10,) * 10)
    d = decide(candidate=candidate, baseline=baseline)
    assert d.accepted is True, d.reasons


def test_the_required_margin_can_never_exceed_what_is_achievable():
    """A reduction is bounded by 100%. Any rule that can demand more than that is not strict,
    it is broken -- it refuses every possible submission while looking like a threshold."""
    from hermes.acceptance import reduction_interval

    noisy = tuple([61_798] * 4 + [1_200, 122_000] + [61_798] * 4)
    low, high = reduction_interval(noisy, (1,) * 10)
    assert high <= 1.0
    assert low <= 1.0


def test_two_draws_from_one_distribution_do_not_clear_the_bar():
    """The property the gate exists for, and the one the multiple did achieve: no real
    improvement must not read as one."""
    from hermes.acceptance import reduction_interval

    same = (52_000, 61_000, 70_000, 58_000, 66_000, 49_000, 73_000, 60_000, 55_000, 68_000)
    other = (61_000, 52_000, 66_000, 70_000, 49_000, 58_000, 60_000, 73_000, 68_000, 55_000)
    low, _ = reduction_interval(same, other)
    assert low < MIN_TOKEN_REDUCTION


def test_the_interval_narrows_as_attempts_accumulate():
    """What makes a refusal actionable. The raw spread does not shrink with n; the uncertainty
    in the median does, so "run more repeats" is a real remedy rather than a brush-off."""
    from hermes.acceptance import reduction_interval

    base_cycle = (52_000, 61_000, 70_000, 58_000, 66_000, 49_000, 73_000, 60_000, 55_000, 68_000)
    cand_cycle = (36_000, 43_000, 49_000, 41_000, 46_000, 34_000, 51_000, 42_000, 39_000, 48_000)

    def width(reps):
        lo, hi = reduction_interval(base_cycle * reps, cand_cycle * reps)
        return hi - lo

    assert width(8) < width(1), "more paired attempts must tighten the interval"


def test_the_interval_is_deterministic():
    """Two people judging the same submission must reach the same verdict. An unseeded
    bootstrap would make acceptance a coin flip in the fourth decimal place."""
    from hermes.acceptance import reduction_interval

    a = (52_000, 61_000, 70_000, 58_000, 66_000)
    b = (36_000, 43_000, 49_000, 41_000, 46_000)
    assert reduction_interval(a, b) == reduction_interval(a, b)

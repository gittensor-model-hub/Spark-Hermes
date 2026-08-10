"""The acceptance bar: correctness first, then efficiency, and only above the noise."""

import pytest

from hermes.acceptance import MIN_ATTEMPTS, AcceptanceError, Arm, decide, dominates


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
    """The interesting refusal. Two draws from one distribution clear a 20% gate about 15%
    of the time at a 15% spread, so a 20% margin there is indistinguishable from nothing."""
    noisy_baseline = _arm(MIN_ATTEMPTS, MIN_ATTEMPTS, [7_000, 10_000, 13_000] * 4, [20] * 12)
    d = decide(candidate=_steady(MIN_ATTEMPTS, 8_000, 10), baseline=noisy_baseline)
    assert d.accepted is False
    assert "the observed spread demands" in d.reasons[0]
    assert "cannot be distinguished from noise" in d.reasons[0]


def test_a_margin_that_clears_the_spread_is_accepted():
    noisy_baseline = _arm(MIN_ATTEMPTS, MIN_ATTEMPTS, [9_500, 10_000, 10_500] * 4, [20] * 12)
    d = decide(candidate=_steady(MIN_ATTEMPTS, 5_000, 8), baseline=noisy_baseline)
    assert d.accepted is True
    assert d.token_reduction == 0.5


def test_a_single_run_arm_has_infinite_spread_so_no_margin_can_clear_it():
    """One run tells you nothing about its own variability, and a gate that accepted it
    would be accepting an uncalibrated threshold."""
    d = decide(candidate=_steady(MIN_ATTEMPTS, 1_000, 2), baseline=_arm(1, 1, [10_000], [20]))
    assert d.accepted is False
    assert "observed spread demands" in d.reasons[0]


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

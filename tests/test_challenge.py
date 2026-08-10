"""Turning a baseline failure into a challenge, and refusing to when it isn't one."""

import pytest

from hermes.challenge import (
    EXPENSIVE_SUCCESS,
    HEALTHY,
    HIDDEN_VERIFY_FAILED,
    MALFORMED_PROTOCOL,
    NO_PROGRESS_LOOP,
    OVERFIT,
    PUBLIC_VERIFY_FAILED,
    SETUP_FAILED,
    STEP_BUDGET_EXHAUSTED,
    Attempt,
    Baseline,
    ChallengeError,
    classify,
    open_challenge,
)

EPOCH = {"model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "harness_digest": "a" * 64}


def _attempt(**kw):
    base = dict(public_passed=False, hidden_passed=None, tokens=10_000, tool_calls=20, wall_time_s=30.0, steps=8)
    base.update(kw)
    return Attempt(**base)


def _fails(n=10, **kw):
    return Baseline(task_id="t", attempts=tuple(_attempt(**kw) for _ in range(n)))


def _passes(n=10, tokens=10_000):
    return Baseline(
        task_id="t",
        attempts=tuple(_attempt(public_passed=True, hidden_passed=True, tokens=tokens) for _ in range(n)),
    )


# --- the classifier reads the trace ----------------------------------------------------------


def test_a_plain_failure_is_a_public_verify_failure():
    assert classify(_attempt()) == PUBLIC_VERIFY_FAILED


def test_passing_public_and_failing_hidden_is_overfit_not_a_capability_gap():
    """Same exit status, completely different finding: one is a model that could not do the
    task, the other is a model that learned the published check."""
    assert classify(_attempt(public_passed=True, hidden_passed=False)) == OVERFIT
    assert classify(_attempt(public_passed=False, hidden_passed=False)) == HIDDEN_VERIFY_FAILED


def test_running_out_of_steps_is_distinguished_from_getting_it_wrong():
    """A miner optimises differently for the two: one needs a better plan, one needs a
    cheaper one."""
    assert classify(_attempt(max_steps_hit=True)) == STEP_BUDGET_EXHAUSTED


def test_repeating_the_same_action_is_a_loop_not_a_wrong_answer():
    assert classify(_attempt(repeated_actions=4)) == NO_PROGRESS_LOOP


def test_a_malformed_protocol_outranks_the_verdict_it_produced():
    """A run the wire format could not read was never really scored, and the fix is the
    protocol rather than the strategy."""
    assert classify(_attempt(public_passed=True, hidden_passed=True, malformed_turns=2)) == MALFORMED_PROTOCOL


def test_broken_setup_outranks_everything():
    assert classify(_attempt(setup_failed=True, max_steps_hit=True)) == SETUP_FAILED


def test_a_pass_inside_the_envelope_is_healthy():
    assert classify(_attempt(public_passed=True, hidden_passed=True), envelope_tokens=10_000) == HEALTHY


def test_a_pass_far_outside_the_envelope_is_a_challenge():
    a = _attempt(public_passed=True, hidden_passed=True, tokens=30_000)
    assert classify(a, envelope_tokens=10_000) == EXPENSIVE_SUCCESS


def test_a_task_with_no_withheld_check_is_not_treated_as_having_failed_it():
    """hidden_passed is None when the task declares no withheld check. Reading None as
    False would open a challenge on every task in the corpus that has none."""
    assert classify(_attempt(public_passed=True, hidden_passed=None)) == HEALTHY


# --- a challenge is confirmed over repeats, never opened on one run --------------------------


def test_one_failed_run_does_not_open_a_challenge():
    """The whole reason this gate exists: a challenge opened on sampling noise sends every
    miner in the subnet to work on a task the baseline usually solves."""
    with pytest.raises(ChallengeError, match="one unlucky sample"):
        open_challenge(_fails(1), epoch=EPOCH)


def test_the_refusal_quotes_the_interval_rather_than_the_count():
    """'failed once' and 'fails reliably' are the same integer and very different facts."""
    with pytest.raises(ChallengeError) as exc:
        open_challenge(_fails(2), epoch=EPOCH)
    assert "true pass rate anywhere in" in str(exc.value)


def test_a_baseline_that_mostly_passes_is_a_flaky_task_not_a_capability_gap():
    attempts = tuple(
        [_attempt(public_passed=True, hidden_passed=True) for _ in range(7)] + [_attempt() for _ in range(3)]
    )
    with pytest.raises(ChallengeError, match="flaky task"):
        open_challenge(Baseline(task_id="t", attempts=attempts), epoch=EPOCH)


def test_a_mixed_baseline_is_not_dismissed_as_healthy_by_a_majority_vote():
    """The refusal must name the reliability problem. `dominant_class` is a majority vote, so
    7-of-10 passes returns HEALTHY -- and "nothing to improve" would bury a task that fails
    almost a third of the time under a message saying it was fine."""
    attempts = tuple(
        [_attempt(public_passed=True, hidden_passed=True) for _ in range(7)] + [_attempt() for _ in range(3)]
    )
    with pytest.raises(ChallengeError) as exc:
        open_challenge(Baseline(task_id="t", attempts=attempts), epoch=EPOCH, envelope_tokens=10_000)
    assert "nothing for a miner to improve" not in str(exc.value)
    assert "flaky task" in str(exc.value)


def test_an_expensive_success_with_a_mixed_baseline_reports_the_failures_not_the_cost():
    """Correctness outranks cost. A task that also fails 3 of 10 times is not a pure resource
    challenge, and labelling it one would send miners to optimise tokens on a task they
    cannot reliably pass."""
    attempts = tuple(
        [_attempt(public_passed=True, hidden_passed=True, tokens=40_000) for _ in range(7)]
        + [_attempt() for _ in range(3)]
    )
    with pytest.raises(ChallengeError, match="flaky task"):
        open_challenge(Baseline(task_id="t", attempts=attempts), epoch=EPOCH, envelope_tokens=10_000)


def test_a_reliable_failure_opens():
    c = open_challenge(_fails(10), epoch=EPOCH)
    assert c.failure_class == PUBLIC_VERIFY_FAILED
    assert c.baseline.pass_rate == 0.0


def test_a_baseline_with_no_attempts_is_refused():
    with pytest.raises(ChallengeError, match="establishes nothing"):
        Baseline(task_id="t", attempts=())


# --- what must never open --------------------------------------------------------------------


def test_a_healthy_baseline_cannot_become_a_challenge():
    with pytest.raises(ChallengeError, match="nothing for a miner to improve"):
        open_challenge(_passes(10), epoch=EPOCH, envelope_tokens=10_000)


def test_broken_infrastructure_cannot_become_a_challenge():
    """It would send miners to fix a workspace, and every one of them would fail."""
    with pytest.raises(ChallengeError, match="infrastructure breakage"):
        open_challenge(_fails(10, setup_failed=True), epoch=EPOCH)


def test_an_expensive_success_opens_despite_passing_every_attempt():
    """The pass-rate bar is about capability gaps. An expensive success is a resource
    challenge, so it must not be refused for the thing that defines it."""
    c = open_challenge(_passes(10, tokens=40_000), epoch=EPOCH, envelope_tokens=10_000)
    assert c.failure_class == EXPENSIVE_SUCCESS
    assert c.baseline.pass_rate == 1.0


def test_an_expensive_success_still_needs_enough_attempts_for_a_median():
    with pytest.raises(ChallengeError, match="one unlucky sample"):
        open_challenge(_passes(2, tokens=40_000), epoch=EPOCH, envelope_tokens=10_000)


# --- the packet carries the commitment, never the check --------------------------------------


def test_the_withheld_body_is_never_in_the_packet():
    """A challenge is published to miners. Shipping the check hands over the answer key."""
    c = open_challenge(
        _fails(10),
        epoch=EPOCH,
        task_pins={"hidden_verify_commitment": "deadbeef", "hidden_verify": "test -f secret.txt"},
    )
    record = c.to_record()
    assert record["withheld"]["hidden_verify_commitment"] == "deadbeef"
    assert record["withheld"]["body_included"] is False
    assert "secret.txt" not in str(record)


def test_an_unrecognised_task_field_is_dropped_rather_than_published():
    """Deny by default. A denylist filtering `hidden_verify` by name would keep working right
    up until somebody added a second withheld field, and the failure is silent."""
    c = open_challenge(
        _fails(10),
        epoch=EPOCH,
        task_pins={"task_id": "t", "hidden_verify_v2": "test -f secret.txt", "answer_key": "42"},
    )
    record = c.to_record()
    assert record["task"] == {"task_id": "t"}
    assert "secret.txt" not in str(record) and "42" not in str(record["task"])
    # The names travel so a maintainer can see the strip happened; the values do not.
    assert record["withheld"]["dropped_task_keys"] == ["answer_key", "hidden_verify_v2"]


def test_a_challenge_is_built_from_the_real_episode_metrics_dataclass():
    """The seam that breaks in an actual run: `Attempt.from_metrics` reads attributes off
    EpisodeMetrics by name, so a field rename upstream must fail here rather than in Targon."""
    from hermesbench.metrics import EpisodeMetrics

    m = EpisodeMetrics(
        task_id="t",
        success=False,
        tool_calls=14,
        failed_calls=2,
        hit_failure=True,
        recovered=False,
        mutated=True,
        self_checked=False,
        tokens_used=12_345,
        wall_time_s=41.5,
        steps=9,
        public_passed=False,
        hidden_passed=None,
    )
    a = Attempt.from_metrics(m)
    assert (a.tokens, a.tool_calls, a.steps) == (12_345, 14, 9)
    assert a.verified is False
    assert classify(a) == PUBLIC_VERIFY_FAILED


def test_no_acceptance_threshold_is_baked_into_a_challenge():
    """The run that produces challenges is the run that measures the spread. A threshold
    frozen here would be a guess captured at the moment its replacement arrived."""
    record = open_challenge(_fails(10), epoch=EPOCH).to_record()
    assert record["acceptance_thresholds_included"] is False
    assert "0.2" not in str(record.get("why_no_thresholds"))


def test_the_observed_spread_travels_so_the_gate_can_read_it():
    attempts = tuple(_attempt(tokens=t) for t in [9_000, 10_000, 11_000] * 4)
    record = open_challenge(Baseline(task_id="t", attempts=attempts), epoch=EPOCH).to_record()
    assert record["baseline"]["token_spread"] == 0.1


def test_a_single_attempt_reports_an_unusable_spread_rather_than_zero():
    """One run has no variability, and reporting 0.0 would let any margin clear the gate."""
    assert Baseline(task_id="t", attempts=(_attempt(),)).token_spread == float("inf")


# --- content addressing ----------------------------------------------------------------------


def test_two_identical_packets_are_one_challenge():
    a = open_challenge(_fails(10), epoch=EPOCH)
    b = open_challenge(_fails(10), epoch=EPOCH)
    assert a.digest == b.digest


def test_a_different_epoch_is_a_different_challenge():
    """Same task, different model or harness: not comparable, so not the same challenge."""
    a = open_challenge(_fails(10), epoch=EPOCH)
    b = open_challenge(_fails(10), epoch={**EPOCH, "harness_digest": "b" * 64})
    assert a.digest != b.digest


def test_wall_time_is_recorded_and_labelled_as_not_scored():
    record = open_challenge(_fails(10), epoch=EPOCH).to_record()
    assert "median_wall_time_s" in record["baseline"]
    import inspect

    from hermes import challenge

    assert "never scored across nodes" in inspect.getsource(challenge.Challenge.to_record)

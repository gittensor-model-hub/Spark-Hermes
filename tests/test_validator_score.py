"""Scoring a submitted surface against the bar its round published.

`eval.strategy_track` decides whether a surface may merge. This is the step that says whether it
beat anything, and it differs from the miner's local rehearsal in two ways that matter.

The validator does not re-measure the control. The challenge was opened from a measured baseline
and the round's private snapshot carries every one of its attempts, so that baseline *is* the
published bar. Re-running it would replace a published bar with a fresh one and move the target
between submissions. The price is that the comparison only holds while the epoch does, which is
what `epoch_issues` refuses on.

And correctness here counts the withheld check. A miner can only see `public_passed`; the validator
holds the withheld verifiers, and `overfit_attempts` is reported apart from the pass count because
"failed the task" and "passed the one it was shown and failed the one it was not" are different
findings.
"""

import pytest

from hermes.challenge import Attempt, Baseline, open_challenge
from hermes.round import open_round
from validator.score import ScoreError, baseline_arm, candidate_arm, epoch_issues, render, score

EPOCH = {"model_revision": "a" * 40, "harness_digest": "b" * 64}


def _window(*, spread=True, attempts=10):
    # The flat variant keeps the same MEDIAN as the spread one. An earlier version set every token
    # to the spread series' first value, which moved the median as well -- so the comparison it
    # claimed to make about variability was really about two different point estimates.
    flat = 78_000 + (attempts - 1) * 1_000
    ats = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=(78_000 + i * 2_000) if spread else flat,
            tool_calls=11,
            wall_time_s=317.0,
            steps=34,
            max_steps_hit=True,
        )
        for i in range(attempts)
    )
    challenge = open_challenge(
        Baseline(task_id="tc-log-rotation-order", attempts=ats),
        epoch=EPOCH,
        task_pins={"task_id": "tc-log-rotation-order", "hidden_verify_commitment": "sha256:" + "c" * 64},
    )
    return open_round(challenge, round_id="r-001", opened_at=0.0, deadline=1_000.0)


def _rows(n=10, *, tokens=40_000, public=True, hidden=True, malformed=0):
    return [
        {
            "task_id": "tc-log-rotation-order",
            "public_passed": public,
            "hidden_passed": hidden,
            "tokens_used": tokens,
            "tool_calls": 6,
            "steps": 20,
            "malformed_turns": malformed,
        }
        for _ in range(n)
    ]


def _score(window, rows):
    return score(
        window=window,
        miner_id="alice",
        rows=rows,
        model_revision=EPOCH["model_revision"],
        harness_digest=EPOCH["harness_digest"],
    )


# --- the epoch guard ------------------------------------------------------------------------------


def test_a_different_model_is_refused_rather_than_scored():
    """Comparing a candidate measured on one model against a baseline measured on another measures
    the model change. Refused rather than turned into a number."""
    with pytest.raises(ScoreError, match="measures the model change"):
        score(
            window=_window(),
            miner_id="alice",
            rows=_rows(),
            model_revision="deadbeef" * 5,
            harness_digest=EPOCH["harness_digest"],
        )


def test_a_different_harness_is_refused_and_named_separately():
    """A model change and a harness change produce the same wrong number and call for different
    repairs -- re-pin, or re-baseline. A caller told only "epoch mismatch" has to go and find out
    which."""
    with pytest.raises(ScoreError, match="measures the harness change"):
        score(
            window=_window(),
            miner_id="alice",
            rows=_rows(),
            model_revision=EPOCH["model_revision"],
            harness_digest="different",
        )


def test_an_incomplete_epoch_is_unverifiable_rather_than_matching():
    """Absence of a mismatch is not evidence of agreement, and every version of this mistake in
    this repository has looked like a clean result."""
    issues = epoch_issues({}, model_revision="x", harness_digest="y")
    assert issues and "incomplete" in issues[0]
    assert epoch_issues({"model_revision": "x"}, model_revision="x", harness_digest="y")


def test_a_matching_epoch_produces_no_issues():
    assert epoch_issues(EPOCH, model_revision=EPOCH["model_revision"], harness_digest=EPOCH["harness_digest"]) == []


# --- correctness counts the withheld check --------------------------------------------------------


def test_passing_the_published_check_and_failing_the_withheld_one_is_not_a_pass():
    """The whole reason the withheld half is withheld. A miner's local rehearsal counts this as a
    win because they cannot see the other verifier."""
    arm, overfit, _ = candidate_arm(_rows(hidden=False))
    assert arm.passes == 0
    assert overfit == 10


def test_a_task_with_no_withheld_check_still_counts_as_a_pass():
    """`hidden_passed is None` means the task declares no withheld check, which is different from
    having failed one. Treating None as a failure would make every such task unwinnable."""
    arm, overfit, _ = candidate_arm(_rows(hidden=None))
    assert arm.passes == 10 and overfit == 0


def test_overfit_is_reported_apart_from_the_pass_count():
    """Averaging them into one rate hides the only signal that distinguishes a capability gap from
    a surface fitted to the benchmark."""
    card = _score(_window(), _rows(hidden=False))
    assert card.candidate.passes == 0
    assert card.overfit_attempts == 10
    assert "OVERFIT" in render(card)
    assert card.to_record()["overfit_attempts"] == 10


def test_protocol_failures_are_counted_and_shown():
    card = _score(_window(), _rows(malformed=2))
    assert card.protocol_failures == 10
    assert "unparseable turn" in render(card)


# --- the baseline is the published bar -------------------------------------------------------------


def test_the_baseline_comes_from_the_challenges_own_attempts():
    window = _window()
    arm = baseline_arm(window.challenge)
    assert arm.attempts == 10
    assert arm.passes == window.challenge.baseline.passes == 4
    assert len(set(arm.tokens)) > 1, "individual token counts, not an aggregate"


def test_a_single_attempt_baseline_cannot_bound_a_margin():
    """One measurement carries no information about its own variability, so no margin computed
    against it can be told apart from noise.

    Built with `Challenge` directly rather than `open_challenge`, which refuses a one-attempt
    baseline earlier and for its own reasons -- so routing through it tested that refusal instead of
    this one.
    """
    from hermes.challenge import Challenge

    thin = Challenge(
        task_id="t",
        failure_class="public_verify_failed",
        baseline=Baseline(
            task_id="t",
            attempts=(
                Attempt(
                    public_passed=False, hidden_passed=None, tokens=78_000, tool_calls=11, wall_time_s=1.0, steps=3
                ),
            ),
        ),
        epoch=EPOCH,
        task_pins={},
    )
    with pytest.raises(ScoreError, match="no information about its own variability"):
        baseline_arm(thin)


def test_a_baseline_with_no_spread_gives_a_collapsed_interval():
    """Why the private snapshot matters. Rebuilt from the *published* packet a baseline has no
    individual attempts, so the bootstrap has nothing to resample, the interval collapses, and a
    collapsed interval clears any margin at all.

    Compared on WIDTH at a matched median. The first version compared lower bounds while the two
    fixtures also had different medians, so it measured the point estimate rather than the
    variability it named.
    """
    spread = _score(_window(spread=True), _rows(tokens=60_000)).interval
    flat = _score(_window(spread=False), _rows(tokens=60_000)).interval
    assert (flat[1] - flat[0]) < (spread[1] - spread[0]), "no spread must narrow the interval"
    assert flat[1] - flat[0] == 0.0, "with no variability on either side there is nothing to bound"


def test_the_record_says_the_baseline_was_not_freshly_run():
    """A scorecard read later without this line invites the assumption that both arms were measured
    together, which is exactly what the validator does not do."""
    card = _score(_window(), _rows())
    assert card.to_record()["baseline_is_the_published_bar_not_a_fresh_run"] is True


# --- refusals -------------------------------------------------------------------------------------


def test_no_episodes_is_refused():
    with pytest.raises(ScoreError, match="nothing to score"):
        candidate_arm([])


def test_a_zero_token_episode_is_refused_rather_than_averaged_in():
    """Not a cheap run -- a run that did not happen. This is the seventh place in this repository
    where absence had to be stopped from reading as a measured zero."""
    with pytest.raises(ScoreError, match="did not happen"):
        candidate_arm(_rows(tokens=0))


def test_a_surface_that_is_correct_and_cheaper_is_accepted():
    card = _score(_window(), _rows(tokens=30_000))
    assert card.accepted is True
    assert card.interval[0] > 0
    assert "ACCEPTED" in render(card)


def test_a_surface_that_is_correct_and_more_expensive_is_refused():
    card = _score(_window(), _rows(tokens=150_000))
    assert card.accepted is False
    assert card.interval[1] < 0


def test_the_live_miner_arm_scores_as_refused_against_the_published_bar():
    """The real numbers from the box: 3 attempts, 0 verified, median 135,302 tokens against a
    published bar of 4/10 at 78,417. Refused at the correctness floor before efficiency is
    computed, which is the gate's designed ordering."""
    rows = [
        {"public_passed": False, "hidden_passed": None, "tokens_used": t, "tool_calls": 11, "steps": 34}
        for t in (110_332, 144_773, 135_302)
    ]
    card = _score(_window(), rows)
    assert card.accepted is False
    assert card.candidate.passes == 0
    assert "before efficiency is considered" in card.decision.reasons[0]

"""Probability Reward: the arithmetic, and the failures it is shaped to avoid.

Every test here corresponds to a decision in `hermes/reward_pr.py` that could plausibly be
"simplified" by a later reader into something that still runs, still trains, and is wrong --
the mean quietly becoming a product, the clip being dropped, the threshold being frozen. Those
are the edits this file exists to fail.
"""

from __future__ import annotations

import math

import pytest

from hermes.reward_pr import (
    PRScore,
    RewardError,
    StdFilter,
    baseline_cache_key,
    debias,
    population_std,
    probability_reward,
    probability_reward_from_logprobs,
)

# --- the reward itself ----------------------------------------------------------------


def test_the_reward_is_the_mean_of_reference_token_probabilities():
    assert probability_reward([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert probability_reward([1.0, 0.0]) == pytest.approx(0.5)


def test_the_mean_survives_a_synonym_where_the_product_would_not():
    """The paper's own argument for the mean, as three numbers.

    `(0.01, 0.7, 0.9)` and `(0.05, 0.7, 0.9)` differ only on a first token that a paraphrase
    would move anyway. Under the normalised product they score ~5x apart; under the mean they
    are within 3%. A reward that swings that far on the first token of a synonym trains the
    model to guess the reference's wording rather than to reason toward its meaning.
    """
    a, b = [0.01, 0.7, 0.9], [0.05, 0.7, 0.9]
    mean_ratio = probability_reward(b) / probability_reward(a)

    product = lambda xs: math.prod(xs) ** (1 / len(xs))  # noqa: E731 - the rejected f_seq
    product_ratio = product(b) / product(a)

    assert mean_ratio < 1.05, "the mean barely moves"
    assert product_ratio > 1.5, "the product moves a lot -- which is why it was rejected"


def test_logprobs_are_exponentiated_per_token_then_averaged():
    """Not summed and exponentiated, which is the normalised product by another name.

    The two are one line apart in any implementation and differ by exactly the failure above,
    which is why the conversion lives in the module rather than at each call site.
    """
    logprobs = [math.log(0.5), math.log(0.25)]
    assert probability_reward_from_logprobs(logprobs) == pytest.approx(0.375)
    assert probability_reward_from_logprobs(logprobs) != pytest.approx(math.exp(sum(logprobs) / 2))


def test_an_empty_reference_is_refused_rather_than_scored_zero():
    """A reference with no tokens means the caller sliced the logprob stream wrongly.

    `hermes/teachers.py` records a gateway returning 58 logprobs for a nine-token answer. Under
    a misalignment like that the reference slice can come back empty, and 0.0 is a
    legitimate-looking reward -- indistinguishable from a rollout the model genuinely scored
    badly, and it would train against that rollout for a bookkeeping error.
    """
    with pytest.raises(RewardError, match="not aligned"):
        probability_reward([])


def test_logits_passed_as_probabilities_are_refused():
    """The likeliest wrong input. A logprob is negative and a logit is unbounded; either one
    silently produces a reward outside [0, 1] that every downstream comparison then trusts."""
    with pytest.raises(RewardError, match=r"outside \[0, 1\]"):
        probability_reward([-0.7, -1.2])
    with pytest.raises(RewardError, match=r"outside \[0, 1\]"):
        probability_reward([4.2])


# --- debiasing ------------------------------------------------------------------------


def test_the_reward_is_the_improvement_the_reasoning_bought():
    assert debias(0.8, 0.5) == pytest.approx(0.3)
    assert debias(0.5, 0.5) == pytest.approx(0.0)


def test_reasoning_that_hurts_contributes_nothing_rather_than_backwards():
    """The clip, and why it is not cosmetic.

    Reasoning that makes the reference LESS likely gives a negative difference, and a negative
    reward flips the gradient for that rollout -- training away from a response that may simply
    have been worded differently from the reference. Zero makes it contribute nothing.
    """
    assert debias(0.2, 0.9) == 0.0
    assert debias(0.0, 1.0) == 0.0


def test_the_reward_stays_in_range_for_a_downstream_comparison():
    assert debias(1.0, 0.0) == 1.0
    assert 0.0 <= debias(0.99, 0.01) <= 1.0


def test_a_high_baseline_question_cannot_earn_reward_for_free():
    """The bias the subtraction removes.

    A short, high-frequency reference scores well after ANY reasoning. Without the baseline the
    model is paid for which questions happen to have easy answers, which is a signal it cannot
    act on and will learn to seek.
    """
    easy_question_raw, easy_question_baseline = 0.95, 0.94
    hard_question_raw, hard_question_baseline = 0.40, 0.10
    assert debias(hard_question_raw, hard_question_baseline) > debias(easy_question_raw, easy_question_baseline)


def test_the_baseline_key_is_the_reference_not_the_task():
    """Keying too coarsely reuses one baseline across different references, which rescales the
    reward for every rollout under that key -- invisibly, because all of them move together."""
    assert baseline_cache_key("Q", "A") == baseline_cache_key("Q", "A")
    assert baseline_cache_key("Q", "A") != baseline_cache_key("Q", "B")


# --- auditability ---------------------------------------------------------------------


def test_a_score_keeps_both_halves_so_it_can_be_recomputed():
    """The argument for using this signal at all is that a third party serving the same pin can
    reproduce it. Storing only the difference discards what makes that possible."""
    score = PRScore.compute([0.8, 0.6], baseline=0.4)
    assert score.raw == pytest.approx(0.7)
    assert score.baseline == pytest.approx(0.4)
    assert score.value == pytest.approx(0.3)
    assert score.reference_tokens == 2


# --- the adaptive filter --------------------------------------------------------------


def test_population_std_not_sample_std():
    """The rollouts ARE the group the objective normalises over, not a sample from a larger set
    that was never generated. The sample form divides by n-1 and inflates small groups -- which
    are exactly the groups sitting near the threshold."""
    assert population_std([0.0, 1.0]) == pytest.approx(0.5)
    assert population_std([0.5]) == 0.0
    assert population_std([]) == 0.0


def test_everything_is_kept_before_the_first_update():
    """Undefined is not zero. Filtering against a threshold nobody has measured either discards
    the first step entirely or admits it against a guess."""
    assert StdFilter().keeps([0.1, 0.9]) is True


def test_a_prompt_whose_rollouts_agree_is_dropped():
    """The whole point: a prompt every rollout scores alike carries no gradient under a
    group-relative objective, and PR being bounded means that shows up as low spread rather
    than as all-correct."""
    f = StdFilter()
    f.update([[0.1, 0.9], [0.2, 0.8]])
    assert f.keeps([0.50, 0.50]) is False
    assert f.keeps([0.05, 0.95]) is True


def test_the_threshold_follows_the_run_rather_than_being_fixed():
    """The std distribution drifts during training. A constant cut is too strict early and too
    loose later, and in BOTH regimes the run still trains -- on the wrong subset."""
    f = StdFilter(decay=0.5)
    first = f.update([[0.0, 1.0]])
    second = f.update([[0.4, 0.6]])
    assert first is not None and second is not None
    assert second < first, "a step with tighter spread must lower the cut"
    assert f.steps == 2


def test_a_single_rollout_group_does_not_drag_the_threshold_down():
    """A group of one has no spread to measure. Counting it as 0.0 would pull the EMA toward
    zero and quietly widen the filter for every prompt after it."""
    f = StdFilter()
    f.update([[0.2, 0.8]])
    before = f.threshold
    f.update([[0.5]])
    assert f.threshold == before


def test_a_decay_that_never_moves_is_refused():
    with pytest.raises(RewardError, match="never moves"):
        StdFilter(decay=1.0)


def test_generator_groups_are_measured_not_silently_zeroed():
    """A rollout buffer is naturally a generator. Testing the length and then measuring the
    spread reads the iterable twice: correct for the declared Sequence, and a threshold of zero
    -- which disables the filter entirely -- for anything lazy."""
    f = StdFilter()
    assert f.update(iter([iter([0.0, 1.0]), iter([0.2, 0.8])])) == pytest.approx(0.4)

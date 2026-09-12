"""Turning a served echo response into reference-token probabilities.

Every failure this module guards against returns a NUMBER rather than an error when unguarded:
a server that ignored `echo`, a server that generated instead of scoring, a token straddling the
boundary, a logprob stream aligned to something else. `hermes/teachers.py` records two of those
happening in practice against real gateways, both answering HTTP 200. So the tests here are
mostly about what gets refused.
"""

from __future__ import annotations

import math

import pytest

from hermes.reward_pr import RewardError
from hermes.reward_scoring import (
    BaselineCache,
    ScoredSequence,
    ScoringError,
    reference_logprobs,
    score_rollout,
)


def sequence(pieces: list[tuple[str, float | None]]) -> ScoredSequence:
    """Build an echo response from (token_text, logprob) pairs, computing offsets as a server would."""
    tokens, logprobs, offsets, cursor = [], [], [], 0
    for text, logprob in pieces:
        tokens.append(text)
        logprobs.append(logprob)
        offsets.append(cursor)
        cursor += len(text)
    return ScoredSequence(tokens=tuple(tokens), token_logprobs=tuple(logprobs), text_offset=tuple(offsets))


def fake_scorer(per_token: float):
    """A scorer assigning one logprob to every token, tokenising on spaces.

    Deliberately not a mock with recorded calls: the thing worth testing is that the alignment
    reads real offsets, and a tokeniser that splits differently from the slice boundary is exactly
    what would expose a bug.
    """

    def score(text: str) -> ScoredSequence:
        parts, cursor, pieces = text.split(" "), 0, []
        for index, part in enumerate(parts):
            chunk = part if index == len(parts) - 1 else part + " "
            pieces.append((chunk, None if cursor == 0 else per_token))
            cursor += len(chunk)
        return sequence(pieces)

    return score


# --- alignment ------------------------------------------------------------------------


def test_the_reference_is_located_by_character_offset():
    scored = sequence([("Q: ", None), ("think ", -0.1), ("answer", -0.2)])
    assert reference_logprobs(scored, prefix="Q: think ") == [-0.2]


def test_a_token_straddling_the_boundary_is_refused_not_assigned():
    """No slice is correct: that token's probability is partly the prefix's.

    Including it credits the reference with the prefix's predictability; excluding it drops a real
    reference token. Both are small and both are SYSTEMATIC, and a systematic error in a reward is
    a direction the policy learns to move in.
    """
    scored = sequence([("Q:", None), (" think+ans", -0.3)])
    with pytest.raises(ScoringError, match="spans the prefix/reference boundary"):
        reference_logprobs(scored, prefix="Q: think")


def test_a_response_without_the_reference_is_refused():
    """The shape a server that ignored `echo` returns."""
    scored = sequence([("Q: ", None), ("think", -0.1)])
    with pytest.raises(ScoringError, match="did not contain the reference"):
        reference_logprobs(scored, prefix="Q: think")


def test_an_unconditioned_first_token_is_refused():
    """Only token 0 can lack a logprob, so a None inside the slice means the prefix was empty --
    and an unconditioned score is a different quantity from Eq 2."""
    scored = sequence([("answer", None)])
    with pytest.raises(ScoringError, match="no logprob"):
        reference_logprobs(scored, prefix="")


def test_ragged_arrays_are_refused_on_construction():
    """Aligning across arrays of different lengths silently scores the wrong tokens."""
    with pytest.raises(ScoringError, match="internally inconsistent"):
        ScoredSequence(tokens=("a", "b"), token_logprobs=(-0.1,), text_offset=(0, 1))


def test_a_multi_token_reference_keeps_every_token():
    scored = sequence([("P", None), ("a", -0.1), ("b", -0.2), ("c", -0.3)])
    assert reference_logprobs(scored, prefix="P") == [-0.1, -0.2, -0.3]


# --- scoring a rollout ----------------------------------------------------------------


def test_the_reward_is_the_difference_the_reasoning_makes():
    """Both passes end at the same reference; they differ only in whether reasoning precedes it.
    That difference IS the reward."""

    def scorer(text: str) -> ScoredSequence:
        # the reference is likelier when the reasoning is present
        confident = "because " in text
        lp = math.log(0.9 if confident else 0.4)
        return fake_scorer(lp)(text)

    score = score_rollout(scorer, prompt="Q ", reasoning="because ", reference="A")
    assert score.raw == pytest.approx(0.9, abs=1e-6)
    assert score.baseline == pytest.approx(0.4, abs=1e-6)
    assert score.value == pytest.approx(0.5, abs=1e-6)


def test_reasoning_that_does_not_help_earns_nothing():
    score = score_rollout(fake_scorer(math.log(0.5)), prompt="Q ", reasoning="hmm ", reference="A")
    assert score.value == pytest.approx(0.0, abs=1e-6)


def test_an_empty_reference_is_refused():
    with pytest.raises(ScoringError, match="nothing to score"):
        score_rollout(fake_scorer(math.log(0.5)), prompt="Q ", reasoning="r ", reference="")


def test_a_rollout_with_no_reasoning_scores_zero_rather_than_erroring():
    """The degenerate case is legitimate -- a model that answered with no deliberation -- and it
    should report "the reasoning bought nothing", not fail."""
    score = score_rollout(fake_scorer(math.log(0.7)), prompt="Q ", reasoning="", reference="A")
    assert score.value == pytest.approx(0.0, abs=1e-6)


def test_the_generated_answer_is_never_sent():
    """Eq 2 REPLACES the generated answer with the reference. A caller holding the generated text
    should not be able to leak it into the scored sequence, so it is not a parameter at all."""
    seen: list[str] = []

    def recording(text: str) -> ScoredSequence:
        seen.append(text)
        return fake_scorer(math.log(0.5))(text)

    score_rollout(recording, prompt="Q ", reasoning="think ", reference="REF")
    assert seen == ["Q REF", "Q think REF"]
    assert all("GENERATED" not in text for text in seen)


# --- the baseline cache ---------------------------------------------------------------


def test_one_prompt_group_costs_k_plus_one_passes_not_two_k():
    """The saving this cache exists for. Sixteen rollouts of one prompt is 17 scoring passes
    rather than 32, and the passes are the cost of the whole method."""
    calls: list[str] = []

    def counting(text: str) -> ScoredSequence:
        calls.append(text)
        return fake_scorer(math.log(0.5))(text)

    cache = BaselineCache(scorer=counting)
    for index in range(16):
        score_rollout(counting, prompt="Q ", reasoning=f"r{index} ", reference="A", baselines=cache)

    assert cache.misses == 1, "the baseline is computed once"
    assert cache.hits == 15
    assert cache.passes_saved == 15
    assert len(calls) == 17, "16 rollout passes + 1 baseline"


def test_the_cache_keys_on_the_reference_not_only_the_prompt():
    """Keying too coarsely reuses one baseline across different references, rescaling every reward
    under that key at once -- invisibly, because they all move together."""
    cache = BaselineCache(scorer=fake_scorer(math.log(0.5)))
    cache.get("Q ", "A")
    cache.get("Q ", "B")
    assert cache.misses == 2


def test_a_scoring_error_is_a_reward_error():
    """So a caller catching the reward module's error type does not miss a serving failure and
    treat the rollout as unscored-but-fine."""
    assert issubclass(ScoringError, RewardError)

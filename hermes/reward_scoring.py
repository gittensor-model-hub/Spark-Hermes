"""Getting the probabilities `hermes.reward_pr` needs out of a served model.

`hermes/reward_pr.py` is the arithmetic and deliberately does not run a model. This is the half
that does: turning a rollout and its reference answer into the per-token probabilities Eq 2
averages, without letting the model generate anything.

## Scoring is not generation, and asking for it wrongly looks like it worked

RLPR needs `P(reference_token_i | everything before it)` for a sequence the model did **not**
produce. That is a forced continuation: the reference answer is written into the sequence and the
server is asked what it *would* have assigned. Every failure mode here produces a number rather
than an error, which is why this module refuses more than it repairs.

The request is `/v1/completions` with `echo=True`, `max_tokens=0` and `logprobs` enabled --
supported by both engines this project serves on. `max_tokens=0` is load-bearing: with anything
higher the server appends its own continuation, the echoed prefix still comes back, the slice
below still finds reference tokens, and the reward is computed from a sequence with generated
text stapled to the end. Nothing raises.

The chat endpoint is the wrong surface for this and is not offered. It has no echo, so the only
probabilities it returns are for tokens the model chose -- which is the opposite of the quantity
wanted.

## Alignment is the whole difficulty, and it is not solved by counting tokens

`hermes/teachers.py` records the measurement this module is shaped by: one gateway returned a
58-token logprob stream for a nine-token answer because it included reasoning tokens, and another
dropped the `logprobs` key entirely across eight request shapes. Both were HTTP 200. A consumer
that trusted position or length there would average the wrong tokens and report a reward.

So the reference tokens are located by **character offset**, not by index or count. The
completions logprob payload carries `text_offset` per token; the reference begins at
`len(prefix)`, so a token belongs to the reference exactly when its offset is at or past that.
No local tokenizer is consulted, deliberately -- tokenizing here to count tokens would introduce
a second tokenizer that can disagree with the server's, and a disagreement would silently shift
the slice rather than fail.

**The boundary can straddle.** If the prefix's last character and the reference's first fall in
one token, no slice is correct: that token's probability belongs partly to text the reasoning
wrote. It is refused rather than assigned to either side, because including it credits the
reference with the prefix's predictability and excluding it drops a real reference token. Both
are small, both are systematic, and a systematic error in a reward is a direction the policy
learns to move in. Joining prefix and reference on a newline, which is what the chat templates
already do between a reasoning block and an answer, makes this rare -- but rare is not never and
the check costs nothing.

## Two passes, and only one of them varies

Eq 4 subtracts a baseline: the same score with the reference decoded from the prompt alone, no
reasoning. That depends only on `(prompt, reference)` and is therefore identical across every
rollout sampled for one prompt. `BaselineCache` exists so a group of k rollouts costs k+1 scoring
passes rather than 2k. On a 16-rollout group that is the difference between 17 and 32 forward
passes per prompt, and the passes are the cost of this method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from hermes.reward_pr import PRScore, RewardError, baseline_cache_key, probability_reward_from_logprobs

__all__ = [
    "BaselineCache",
    "EchoScorer",
    "ScoredSequence",
    "ScoringError",
    "openai_echo_scorer",
    "reference_logprobs",
    "score_rollout",
]


class ScoringError(RewardError):
    """A served response cannot be turned into reference-token probabilities."""


@dataclass(frozen=True)
class ScoredSequence:
    """What an echo request returns, reduced to the three fields alignment needs.

    `token_logprobs` may hold `None` at index 0 and only there: the first token of a sequence has
    nothing before it to be conditioned on, so the server reports no logprob for it. That is a
    property of the format and not a fault, which is why it is representable here rather than
    rejected on arrival -- it is only a problem if the reference starts at token 0, and that case
    is caught where it means something.
    """

    tokens: tuple[str, ...]
    token_logprobs: tuple[float | None, ...]
    text_offset: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.tokens) == len(self.token_logprobs) == len(self.text_offset)):
            raise ScoringError(
                f"echo response is internally inconsistent: {len(self.tokens)} tokens, "
                f"{len(self.token_logprobs)} logprobs, {len(self.text_offset)} offsets. Aligning "
                "across ragged arrays would silently score the wrong tokens."
            )


class EchoScorer(Protocol):
    """Scores a fixed string without generating. See `openai_echo_scorer` for the real one."""

    def __call__(self, text: str) -> ScoredSequence: ...


def reference_logprobs(scored: ScoredSequence, *, prefix: str) -> list[float]:
    """The logprobs of exactly the reference tokens -- those at or past `len(prefix)`.

    Refuses rather than guesses in the three cases where a slice would be wrong:

    * a token straddling the prefix/reference boundary, whose probability belongs to both
    * a reference slice that is empty, which means the echo did not contain the reference at all
    * a `None` logprob inside the slice, which can only be token 0 and means the "prefix" was
      empty -- a caller asking for an unconditioned score, which Eq 2 is not

    A reward is a direction the policy moves in, so a systematic slice error is worse than a
    refusal that names it.
    """
    boundary = len(prefix)
    for index, offset in enumerate(scored.text_offset):
        if offset < boundary < offset + len(scored.tokens[index]):
            raise ScoringError(
                f"token {index} ({scored.tokens[index]!r}) spans the prefix/reference boundary at "
                f"character {boundary}: its probability is partly the prefix's. Neither including "
                "nor dropping it is correct, and both errors are systematic. Join the prefix and "
                "the reference on a token boundary -- a newline, as the chat templates do."
            )

    picked = [
        (i, lp)
        for i, off, lp in zip(range(len(scored.tokens)), scored.text_offset, scored.token_logprobs)
        if off >= boundary
    ]
    if not picked:
        raise ScoringError(
            f"no echoed token begins at or after character {boundary}; the response did not "
            "contain the reference. A server that ignored `echo` returns exactly this shape."
        )
    for index, logprob in picked:
        if logprob is None:
            raise ScoringError(
                f"token {index} has no logprob. Only the first token of a sequence can, so the "
                "prefix was empty -- Eq 2 scores the reference CONDITIONED on something, and an "
                "unconditioned score is a different quantity."
            )
    return [logprob for _, logprob in picked if logprob is not None]


def _score_text(scorer: EchoScorer, prefix: str, reference: str) -> float:
    if not reference:
        raise ScoringError("the reference answer is empty; there is nothing to score against")
    scored = scorer(prefix + reference)
    return probability_reward_from_logprobs(reference_logprobs(scored, prefix=prefix))


@dataclass
class BaselineCache:
    """`r'` per `(prompt, reference)`, because it does not vary across a prompt's rollouts.

    Keyed through `hermes.reward_pr.baseline_cache_key` rather than on the prompt alone: two
    rollouts of one prompt share a baseline, two prompts sharing a task id do not, and keying too
    coarsely rescales every reward under that key at once -- invisibly, because they all move
    together.
    """

    scorer: EchoScorer
    hits: int = 0
    misses: int = 0
    _values: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def get(self, prompt: str, reference: str) -> float:
        key = baseline_cache_key(prompt, reference)
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        self._values[key] = _score_text(self.scorer, prompt, reference)
        return self._values[key]

    @property
    def passes_saved(self) -> int:
        """Scoring passes this cache avoided -- counted, not inferred from timing.

        Worth reporting because the saving is the difference between k+1 and 2k forward passes
        per prompt group, and a cache silently missing every time looks identical to one working
        except in the bill.
        """
        return self.hits


def score_rollout(
    scorer: EchoScorer,
    *,
    prompt: str,
    reasoning: str,
    reference: str,
    baselines: BaselineCache | None = None,
) -> PRScore:
    """One rollout's debiased probability reward.

    `prompt` is what preceded the model's turn; `reasoning` is what it produced before its answer;
    `reference` is `y*`. The generated answer is not passed and is not used -- Eq 2 replaces it,
    so a caller holding it should not be tempted to send it.

    Both scored sequences end at the same reference text, and differ only in whether the reasoning
    sits between the prompt and it. That difference IS the reward.
    """
    baseline = baselines.get(prompt, reference) if baselines is not None else _score_text(scorer, prompt, reference)

    joined = prompt if not reasoning else f"{prompt}{reasoning}"
    scored = scorer(joined + reference)
    logprobs = reference_logprobs(scored, prefix=joined)
    return PRScore.compute([math.exp(lp) for lp in logprobs], baseline=baseline)


def openai_echo_scorer(
    *,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    timeout_s: float = 120.0,
    extra_body: dict[str, Any] | None = None,
) -> EchoScorer:
    """An `EchoScorer` over an OpenAI-compatible `/v1/completions` endpoint.

    `max_tokens=0` and `echo=True` together mean "tell me about this text, do not extend it".
    Both are sent explicitly on every call rather than defaulted anywhere, because a server that
    silently generates still returns a well-formed payload whose reference slice parses.

    `logprobs=0` asks for the chosen token's own logprob and no alternatives. RLPR never needs the
    top-k distribution -- it needs the probability of one specific token -- which is also why this
    works where `top_logprobs` truncation would not.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def score(text: str) -> ScoredSequence:
        response = client.completions.create(
            model=model, prompt=text, echo=True, max_tokens=0, logprobs=0, temperature=0.0, **(extra_body or {})
        )
        if not response.choices:
            raise ScoringError("the scoring endpoint returned no choices")
        payload = response.choices[0].logprobs
        if payload is None or payload.text_offset is None or payload.token_logprobs is None or payload.tokens is None:
            raise ScoringError(
                "the scoring endpoint returned no logprobs. hermes/teachers.py records endpoints "
                "answering 200 with the key ABSENT across eight request shapes -- this reward "
                "cannot be computed against such a path, and guessing one would be inventing it."
            )
        return ScoredSequence(
            tokens=tuple(payload.tokens),
            token_logprobs=tuple(payload.token_logprobs),
            text_offset=tuple(payload.text_offset),
        )

    return score

"""Probability Reward: a reward for tasks no verifier was written for.

`hermes.acceptance` scores a rollout by running a withheld verifier over it. That is the
strongest signal this project has and it is why a crown means something. It is also the
ceiling on what can enter the corpus at all: a task with no verifier cannot be scored, so it
cannot be mined, so it never becomes training data. Every domain in
`hermes/router/domains.py` that nobody has written checks for is outside the loop for that
reason alone.

RLPR (arXiv:2506.18254, Yu et al.) is the method this implements. Its claim is that a model's
own probability of emitting the *reference* answer, conditioned on the reasoning it just
produced, tracks the quality of that reasoning well enough to train on -- so a reward exists
for free wherever a reference answer exists, which is a far weaker requirement than a verifier.

## Why this does not weaken the project's verifiability rule

The obvious objection is that "the model's own probability" sounds like a private number, and
this repository's whole premise is improvement a stranger can check. The objection does not
survive the details:

**The scorer is the policy itself, not a separate judge.** Eq 2 feeds the modified sequence
back through the *policy model*. There is no second model to pin, no teacher to host, and no
API whose weights move underneath the score. The validator already serves the pinned policy;
computing this reward needs nothing it does not already have.

**A pinned model at fixed sampling is deterministic.** `hermes/base_model.json` pins a
40-character revision and `eval.hf_pin` refuses movable refs. Two parties serving that pin and
forcing the same continuation get the same per-token probabilities. That is reproducible in
exactly the sense `hermes.promotion` already requires -- more reproducible, in fact, than a
sandboxed verifier whose filesystem and clock differ between hosts.

So PR is not a softer standard than verification. It is a *different* signal with a different
failure mode, and the honest framing is that it trades a hard correctness guarantee for
coverage. `acceptance` should stay authoritative wherever a verifier exists. This is for the
rest.

## What the reward actually is

Split a response `o` into the reasoning `z` and the generated answer `y`. Rebuild the sequence
with the *reference* answer in place of the generated one, feed it to the policy, and take the
per-token probabilities of the reference tokens. Then (Eq 2):

    r = mean({p_i : o'_i in y*})

**Mean, not the normalised product.** The paper tested the product -- sequence likelihood --
and rejected it: it has high variance and is overly sensitive to synonyms. Their own
counterexample is worth keeping because it is the whole argument in three numbers: the
probability sequences `(0.01, 0.7, 0.9)` and `(0.05, 0.7, 0.9)` differ only on a first token
that a synonym would move anyway, and the product scores them vastly differently while the mean
barely moves. A reward that swings on the first token of a paraphrase is a reward that trains
the model to guess the reference's wording rather than to reason.

Length normalisation is redundant under GRPO, which normalises within the group anyway, and is
load-bearing for REINFORCE++ and anything else that does not. Kept unconditionally, because a
reward whose correctness depends on which optimiser consumes it is a trap for the next reader.

## Why the raw probability is not the reward

`r` is contaminated. The paper decomposes its contributors as `U_r = U_z + U_others` (Eq 3):
the part that comes from the reasoning, and the part that comes from the question and the
reference answer being what they are. A short, high-frequency reference answer scores well
after *any* reasoning, and a long technical one scores badly after good reasoning. Train on raw
`r` and the model learns which questions have easy answers.

So a baseline is subtracted (Eq 4): score the reference answer decoded with **no reasoning at
all**, and keep only the improvement.

    r_hat = clip(0, 1, r - r_baseline)

That makes the reward "what the reasoning bought", which is the quantity this project actually
wants to pay for -- and it is the same shape as the efficiency scoring in `hermes.acceptance`,
which credits a surface for the delta it causes rather than for the absolute number.

**This costs a second forward pass per rollout.** Both are forced-continuation scoring passes,
not generation, so they are cheap relative to producing the rollout -- but the baseline depends
only on `(question, reference)`, not on the rollout, so it is identical across every rollout of
one prompt and should be computed once per prompt and reused. `baseline_cache_key` exists for
that.

## Why the filter is adaptive

RLVR stabilises training by dropping prompts that are all-pass or all-fail, because neither
teaches anything. PR is continuous, so there is no "all correct" to test for -- and the paper's
observation is that the same prompts show up as *low standard deviation* across sampled
rollouts, since PR is bounded in [0, 1] and a prompt everyone scores alike has nowhere to
spread.

The threshold cannot be a constant. The std distribution drifts over training, so a fixed cut
is too strict early and too loose later, and in both regimes it is silently wrong: the run
still trains, on the wrong subset. `StdFilter` tracks an exponential moving average of each
step's mean std and cuts below that, which makes the filter a curriculum rather than a
constant.

## What this module does not do

It does not run the model. Getting per-token probabilities for a forced continuation is the
serving layer's job, and `hermes/teachers.py` already records how badly that varies between
providers -- one gateway returned a 58-token logprob stream for a nine-token answer because it
included reasoning tokens, which is exactly the misalignment that would make `r` meaningless
here. Anything feeding this module must align the probabilities to the reference tokens and
nothing else; `LOGPROB_FULL_STREAM` is not usable without slicing it down first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_EMA_DECAY",
    "PRScore",
    "RewardError",
    "StdFilter",
    "baseline_cache_key",
    "debias",
    "probability_reward",
    "probability_reward_from_logprobs",
]

# How fast the filtering threshold follows the run. 0.9 keeps roughly the last ten steps in
# view, which is long enough that one unusual batch does not move the cut and short enough to
# track the drift the paper describes.
DEFAULT_EMA_DECAY = 0.9


class RewardError(ValueError):
    """A probability reward cannot be computed from what was supplied."""


def probability_reward(probabilities: Sequence[float]) -> float:
    """`r` from Eq 2: the mean per-token probability of the reference answer.

    The input is the probabilities of the REFERENCE tokens only. Passing the whole response's
    probabilities silently computes a different quantity -- one dominated by the reasoning the
    model just wrote and therefore trivially maximised by writing confident nonsense.

    Empty input is refused rather than returned as 0.0. A reference answer with no tokens means
    the caller sliced the stream wrongly, and 0.0 is a legitimate-looking reward that would
    train against a rollout for a bookkeeping error.
    """
    values = list(probabilities)
    if not values:
        raise RewardError(
            "no reference-answer token probabilities; the reference is empty or the logprob "
            "stream was not aligned to it. Returning 0.0 here would be indistinguishable from a "
            "rollout the model scored badly."
        )
    for p in values:
        if not (0.0 <= p <= 1.0) or math.isnan(p):
            raise RewardError(f"probability {p!r} is outside [0, 1]; pass probabilities, not logprobs or logits")
    return sum(values) / len(values)


def probability_reward_from_logprobs(logprobs: Iterable[float]) -> float:
    """The same reward from log-probabilities, which is what every serving API actually returns.

    Exponentiated per token and then averaged -- NOT summed and exponentiated, which would be
    the normalised product the paper rejected. The two differ by exactly the failure this
    module's docstring describes, and they are one line apart in an implementation, so the
    conversion lives here rather than at each call site.
    """
    return probability_reward([math.exp(lp) for lp in logprobs])


def debias(reward: float, baseline: float) -> float:
    """`r_hat` from Eq 4: what the reasoning bought, clipped to [0, 1].

    `baseline` is the same probability reward computed with the reference answer decoded
    directly, with no reasoning in front of it. It depends only on the question and the
    reference, so one prompt needs it computed once however many rollouts are sampled.

    The clip is not cosmetic. Reasoning that makes the reference LESS likely produces a negative
    difference, and a negative reward flips the sign of the gradient for that rollout -- which
    trains away from a response that may simply have been differently worded. Clipping at zero
    makes such a rollout contribute nothing instead of contributing backwards.
    """
    return min(1.0, max(0.0, reward - baseline))


def baseline_cache_key(question: str, reference: str) -> tuple[str, str]:
    """What the baseline actually depends on.

    Spelled as a function rather than left to each caller to build, because the tempting key is
    the prompt or the task id -- and both are wrong in the same direction. Two rollouts of one
    prompt share a baseline; two prompts sharing a task id do not. Keying too coarsely reuses a
    baseline across different references, which silently rescales the reward for every rollout
    under that key.
    """
    return (question, reference)


@dataclass
class PRScore:
    """One rollout's reward, with both halves kept.

    `raw` and `baseline` are recorded beside `value` rather than discarded, because a reward
    that cannot be recomputed cannot be audited -- and the whole argument for using this signal
    in a project built on checkable results is that a third party serving the same pin can
    reproduce it. Storing only the difference throws away the part that makes that possible.
    """

    raw: float
    baseline: float
    value: float
    reference_tokens: int

    @classmethod
    def compute(cls, probabilities: Sequence[float], *, baseline: float) -> PRScore:
        raw = probability_reward(probabilities)
        return cls(raw=raw, baseline=baseline, value=debias(raw, baseline), reference_tokens=len(probabilities))


@dataclass
class StdFilter:
    """The adaptive curriculum from §2.4: drop prompts whose rollouts agree with each other.

    A prompt every rollout scores alike carries no gradient under a group-relative objective,
    and PR being bounded in [0, 1] means such prompts show up as low standard deviation rather
    than as all-correct or all-wrong. The cut follows an exponential moving average of the
    per-step mean std, so it is a curriculum that tracks the run instead of a constant that is
    wrong at both ends of it.

    `decay` is the weight kept on history. The threshold is undefined until the first `update`,
    and `keeps` admits everything until then -- deliberately, because the alternative is
    discarding the first step's prompts against a threshold of zero that no prompt can fail,
    or against a guess nobody measured.
    """

    decay: float = DEFAULT_EMA_DECAY
    threshold: float | None = None
    steps: int = 0
    _history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise RewardError(f"decay {self.decay!r} must be in [0, 1); at 1.0 the threshold never moves")

    def update(self, rewards_per_prompt: Iterable[Iterable[float]]) -> float | None:
        """Fold one training step's rollout groups into the threshold. Returns the new cut.

        Groups of fewer than two rollouts are skipped rather than counted as zero std: a single
        sample has no spread to measure, and feeding 0.0 in would drag the threshold down and
        quietly widen the filter for every prompt that followed.

        Each group is materialised ONCE. Testing the length and then measuring the spread reads
        the iterable twice, which is correct for the declared `Sequence` and silently returns a
        threshold of zero for a caller that passes generators -- the shape a rollout buffer
        naturally has.
        """
        groups = [list(group) for group in rewards_per_prompt]
        stds = [population_std(g) for g in groups if len(g) > 1]
        if not stds:
            return self.threshold
        mean_std = sum(stds) / len(stds)
        self.threshold = (
            mean_std if self.threshold is None else self.decay * self.threshold + (1 - self.decay) * mean_std
        )
        self.steps += 1
        self._history.append(mean_std)
        return self.threshold

    def keeps(self, rewards: Sequence[float]) -> bool:
        """Whether this prompt's rollouts spread enough to be worth training on."""
        if self.threshold is None or len(rewards) < 2:
            return True
        return population_std(rewards) >= self.threshold


def population_std(values: Sequence[float]) -> float:
    """Population standard deviation.

    Population rather than sample: the rollouts ARE the group the objective is normalised over,
    not a sample drawn from some larger set of rollouts that were never generated. Using the
    sample form would divide by `n-1` and inflate the spread of small groups, which are exactly
    the groups near the threshold.
    """
    data = list(values)
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return math.sqrt(sum((v - mean) ** 2 for v in data) / len(data))

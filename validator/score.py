"""Score a submitted surface against the challenge it was opened over.

    python -m validator.score --round r-001 --miner alice \\
        --base-url http://127.0.0.1:8000/v1 --model qwen3.6-27b --repeats 10

This is the step that turns an accepted pull request into a decision. `eval.strategy_track` decides
whether a surface may merge; this runs it and says whether it beat the baseline.

## The validator does not re-run a control arm, and that is the point

A miner rehearsing locally has to measure both arms, because their box is not the validator's. The
validator already holds the authoritative control: the challenge was *opened* from a measured
baseline, and `RoundWindow.snapshot()` carries every one of its individual attempts. That baseline
is the bar the round published, so re-measuring it would replace a published bar with a fresh one
and quietly move the target between submissions.

The cost of using a stored baseline is that it was measured at a different moment, so the
comparison is only meaningful while the epoch holds. `Challenge.epoch` records `model_revision` and
`harness_digest` for exactly this, and `epoch_issues` refuses a cross-epoch comparison rather than
producing a number. A candidate measured under a newer harness against a baseline measured under an
older one is a measurement of the harness change.

## Correctness here means the withheld check too

A miner's local rehearsal can only count `public_passed`; they do not hold the withheld verifiers,
which is the entire reason `overfit_rate` means anything. The validator counts
`Attempt.verified` -- published check passed *and* the withheld one did not fail -- so a surface
that learned the published assertions scores as the failure it is.

`overfit_attempts` is reported separately rather than folded into the pass count. "Failed the task"
and "passed the task it was shown and failed the one it was not" call for different responses: the
first is a capability gap, the second is a surface fitted to the benchmark, and averaging them into
one rate would hide the only signal that distinguishes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes.acceptance import Arm, Decision, decide, reduction_interval, repeats_needed


class ScoreError(RuntimeError):
    """A submission cannot be scored."""


@dataclass(frozen=True)
class Scorecard:
    round_id: str
    miner_id: str
    task_id: str
    candidate: Arm
    baseline: Arm
    decision: Decision
    interval: tuple[float, float]
    repeats_to_settle: int | None
    overfit_attempts: int
    protocol_failures: int

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def to_record(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "task_id": self.task_id,
            "candidate": {
                "verified_passes": self.candidate.passes,
                "attempts": self.candidate.attempts,
                "tokens": list(self.candidate.tokens),
            },
            "baseline": {
                "verified_passes": self.baseline.passes,
                "attempts": self.baseline.attempts,
                "tokens": list(self.baseline.tokens),
            },
            "decision": self.decision.to_record(),
            "reduction_interval": list(self.interval),
            "repeats_to_settle": self.repeats_to_settle,
            # Kept out of the pass count on purpose. "Failed the task" and "passed the published
            # check and failed the withheld one" are different findings and only the second says a
            # surface was fitted to the benchmark.
            "overfit_attempts": self.overfit_attempts,
            "protocol_failures": self.protocol_failures,
            "baseline_is_the_published_bar_not_a_fresh_run": True,
        }


def epoch_issues(epoch: dict[str, Any], *, model_revision: str, harness_digest: str) -> list[str]:
    """Whether a candidate measured now is comparable to a baseline measured then.

    Both directions are named rather than summarised. A model change and a harness change produce
    the same wrong number and call for different repairs -- re-pin, or re-baseline -- and a caller
    told only "epoch mismatch" has to go and find out which.

    An empty field in the challenge is reported as unverifiable rather than treated as matching.
    Absence of a mismatch is not evidence of agreement, and every version of this mistake in this
    repository has looked like a clean result.
    """
    issues: list[str] = []
    want_model = str(epoch.get("model_revision") or "")
    want_harness = str(epoch.get("harness_digest") or "")

    if not want_model or not want_harness:
        issues.append(
            f"the challenge's epoch is incomplete (model_revision={want_model!r}, "
            f"harness_digest={want_harness!r}), so nothing can establish that this candidate is "
            "comparable to its baseline"
        )
        return issues
    if want_model != model_revision:
        issues.append(
            f"the baseline was measured on model {want_model} and this candidate on {model_revision}; "
            "comparing them measures the model change rather than the surface"
        )
    if want_harness != harness_digest:
        issues.append(
            f"the baseline was measured under harness {want_harness} and this candidate under "
            f"{harness_digest}; comparing them measures the harness change rather than the surface"
        )
    return issues


def candidate_arm(rows: list[dict[str, Any]]) -> tuple[Arm, int, int]:
    """Build the candidate arm from episode metrics. Returns (arm, overfit, protocol failures).

    A pass is `public_passed and hidden_passed is not False`, matching `Attempt.verified`, because
    the validator holds the withheld verifiers and the whole point of holding them is that they
    count.
    """
    if not rows:
        raise ScoreError("no episodes; there is nothing to score")

    tokens = tuple(int(r.get("tokens_used") or 0) for r in rows)
    if not all(tokens):
        raise ScoreError(
            "an episode reported zero tokens. That is not a cheap run, it is a run that did not "
            "happen, and averaging it in would make the surface look free."
        )
    verified = sum(1 for r in rows if r.get("public_passed") and r.get("hidden_passed") is not False)
    overfit = sum(1 for r in rows if r.get("public_passed") and r.get("hidden_passed") is False)
    malformed = sum(1 for r in rows if int(r.get("malformed_turns") or 0) > 0)
    return (
        Arm(
            passes=verified,
            attempts=len(rows),
            tokens=tokens,
            tool_calls=tuple(int(r.get("tool_calls") or 0) for r in rows),
        ),
        overfit,
        malformed,
    )


def baseline_arm(challenge: Any) -> Arm:
    """The published bar, from the challenge's own attempts.

    Read off `Challenge.baseline`, which the round's private snapshot preserves in full. The
    *published* packet carries only aggregate statistics, so a baseline rebuilt from it would have
    no individual token counts and `reduction_interval` would have nothing to bootstrap -- the
    interval would collapse and every margin would clear.
    """
    attempts = challenge.baseline.attempts
    if len(attempts) < 2:
        raise ScoreError(
            f"the challenge's baseline has {len(attempts)} attempt(s). A single measurement carries "
            "no information about its own variability, so no margin computed against it can be "
            "distinguished from noise."
        )
    return Arm(
        passes=challenge.baseline.passes,
        attempts=len(attempts),
        tokens=tuple(a.tokens for a in attempts),
        tool_calls=tuple(a.tool_calls for a in attempts),
    )


def score(
    *,
    window: Any,
    miner_id: str,
    rows: list[dict[str, Any]],
    model_revision: str,
    harness_digest: str,
) -> Scorecard:
    """Judge one submitted surface. Refuses rather than returning a misleading number."""
    challenge = window.challenge
    mismatch = epoch_issues(challenge.epoch, model_revision=model_revision, harness_digest=harness_digest)
    if mismatch:
        raise ScoreError("; ".join(mismatch))

    candidate, overfit, malformed = candidate_arm(rows)
    baseline = baseline_arm(challenge)
    return Scorecard(
        round_id=window.round_id,
        miner_id=miner_id,
        task_id=challenge.task_id,
        candidate=candidate,
        baseline=baseline,
        decision=decide(candidate=candidate, baseline=baseline),
        interval=reduction_interval(baseline.tokens, candidate.tokens),
        repeats_to_settle=repeats_needed(baseline.tokens, candidate.tokens),
        overfit_attempts=overfit,
        protocol_failures=malformed,
    )


def render(card: Scorecard) -> str:
    from statistics import median

    lines = [
        f"round {card.round_id}   task {card.task_id}   miner {card.miner_id}",
        f"  baseline   {card.baseline.passes}/{card.baseline.attempts} verified   "
        f"median {median(card.baseline.tokens):,.0f} tokens   (the published bar)",
        f"  candidate  {card.candidate.passes}/{card.candidate.attempts} verified   "
        f"median {median(card.candidate.tokens):,.0f} tokens",
    ]
    if card.overfit_attempts:
        lines.append(
            f"  OVERFIT    {card.overfit_attempts} attempt(s) passed the published check and failed the withheld one"
        )
    if card.protocol_failures:
        lines.append(f"  protocol   {card.protocol_failures} attempt(s) had at least one unparseable turn")
    low, high = card.interval
    lines.append(f"  95% interval on the reduction: [{low:.1%}, {high:.1%}]")
    lines.append("")
    lines.append("VERDICT: " + ("ACCEPTED" if card.accepted else "REFUSED"))
    for reason in card.decision.reasons:
        lines.append(f"  - {reason}")
    if card.repeats_to_settle is not None:
        lines.append(f"  about {card.repeats_to_settle} paired repeats would settle evidence like this")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", dest="round_id", required=True)
    parser.add_argument("--miner", dest="miner_id", required=True)
    parser.add_argument("--episodes", type=Path, default=None, help="score an existing log instead of running")
    parser.add_argument("--harness-digest", default="", help="the harness digest this candidate ran under")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the scorecard as JSON")
    args = parser.parse_args(argv)

    from hermes.base_model import load as load_pin
    from hermes.challenge import episode_metrics_of
    from hermesbench.sink import read_episodes
    from validator.store import RoundStore

    if args.episodes is None:
        print(
            "validator.score: --episodes is required. Running the candidate is a separate step so a "
            "scorecard can be recomputed from a log without spending the GPU time again.",
            file=sys.stderr,
        )
        return 2
    try:
        window = RoundStore(args.store).load(args.round_id)
        rows = [episode_metrics_of(r) for r in read_episodes(args.episodes)]
        card = score(
            window=window,
            miner_id=args.miner_id,
            rows=rows,
            model_revision=load_pin().revision,
            harness_digest=args.harness_digest or str(window.challenge.epoch.get("harness_digest") or ""),
        )
    except ScoreError as exc:
        print(f"validator.score: {exc}", file=sys.stderr)
        return 2
    print(render(card))
    if args.out:
        args.out.write_text(json.dumps(card.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if card.accepted else 1


__all__ = [
    "ScoreError",
    "Scorecard",
    "baseline_arm",
    "candidate_arm",
    "epoch_issues",
    "main",
    "render",
    "score",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""One crown an hour. Score every standing solution, keep the best, close the rest, open the next.

    python -m validator.crown select --out datasets/crowns.json
    python -m validator.crown actions --out datasets/crowns.json

Each hour is a tournament, not a standing record. Every evaluated submission is scored, exactly
one wins, every other pull request is closed, the crown label moves to the winner, and a new round
opens. One miner is rewarded per hour and the rest start again.

## Ranking across different tasks

A single winner means comparing a submission on `tc-log-rotation-order` -- baseline 4 of 10 at
78,417 median tokens -- against one on a task whose bar is nothing like it. Absolute tokens cannot
do that: it would rank miners by which task they drew.

So the score is the **lower bound of the reduction interval against that submission's own
baseline**. Each challenge publishes its own bar, so the quantity is already relative, and using
the lower bound rather than the point estimate makes it noise-aware: a 40% win measured on a wildly
variable task scores below a 25% win measured on a tight one, because the second is the one we know
about. `hermes.acceptance.reduction_interval` computes it and this reuses it rather than inventing
a second notion of "better".

## Eligibility comes before ranking, and cannot be traded against it

A submission must pass **every** attempt, over at least `MIN_ATTEMPTS` of them, before its tokens
are looked at. This mirrors `acceptance.decide` -- correctness is a gate, not a term -- so no
amount of token reduction promotes a submission that fails the withheld check. Ranking eligible
entries only is what stops the hourly cadence from turning into "cheapest wrong answer wins".

## An hour with no winner

If nothing is eligible, no crown is awarded. The alternative is crowning the best of a bad field,
and an hourly reward that always pays out stops carrying information within a week. Whether the
losing pull requests still close in that case is a policy choice rather than a fact, so it is a
flag: closing keeps the round boundary clean, and not closing avoids discarding a field that no one
could win from. Either way it is reported.

## The label is moved by this job and nothing else

Removals are emitted before additions. A crown only ever added is a crown several people hold at
once, and this design has exactly one at a time by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.acceptance import MIN_ATTEMPTS, Arm, reduction_interval

CROWNS = Path("datasets/crowns.json")
CROWN_LABEL = "crown"


class CrownError(RuntimeError):
    """A round cannot be settled."""


@dataclass(frozen=True)
class Contender:
    """One evaluated submission, with the bar it was measured against."""

    task_id: str
    miner_id: str
    round_id: str
    candidate: Arm
    baseline: Arm
    pr: int = 0
    received_at: float = 0.0

    @property
    def median_tokens(self) -> float:
        from statistics import median

        return median(self.candidate.tokens)

    @property
    def median_tool_calls(self) -> float:
        from statistics import median

        return median(self.candidate.tool_calls)


@dataclass(frozen=True)
class Score:
    contender: Contender
    lower_bound: float
    interval: tuple[float, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.contender.task_id,
            "miner_id": self.contender.miner_id,
            "round_id": self.contender.round_id,
            "pr": self.contender.pr,
            "score": round(self.lower_bound, 4),
            "reduction_interval": [round(v, 4) for v in self.interval],
            "median_tokens": self.contender.median_tokens,
            "median_tool_calls": self.contender.median_tool_calls,
            "attempts": self.contender.candidate.attempts,
        }


@dataclass
class Outcome:
    winner: Score | None
    ranked: list[Score] = field(default_factory=list)
    ineligible: list[tuple[Contender, str]] = field(default_factory=list)

    @property
    def losers(self) -> list[Contender]:
        """Everyone who does not hold the crown, eligible or not, once each. Their round is over.

        Deduplicated because a contender can appear on both lists: when the best of the field does
        not clear zero it is ranked *and* recorded as ineligible with the reason. Without this the
        same pull request is closed twice, which the API tolerates and the action list does not --
        a duplicate reads as two miners losing when it is one.
        """
        keep = self.winner.contender if self.winner else None
        seen: set[tuple[str, str]] = set()
        out: list[Contender] = []
        for contender in [s.contender for s in self.ranked] + [c for c, _ in self.ineligible]:
            key = (contender.round_id, contender.miner_id)
            if contender is keep or key in seen:
                continue
            seen.add(key)
            out.append(contender)
        return out

    def to_record(self) -> dict[str, Any]:
        return {
            "winner": self.winner.to_record() if self.winner else None,
            "ranked": [s.to_record() for s in self.ranked],
            "ineligible": [
                {"miner_id": c.miner_id, "task_id": c.task_id, "pr": c.pr, "reason": why} for c, why in self.ineligible
            ],
            # Said in the record: an hour with no eligible entry is not an hour whose field was
            # ranked and found wanting, and a reader has to be able to tell those apart.
            "crowned": self.winner is not None,
        }


def eligibility(contender: Contender) -> tuple[bool, str]:
    """Whether a submission may be ranked at all. Correctness is a gate, not a term."""
    arm = contender.candidate
    if not arm.all_passed:
        return False, (
            f"passed {arm.passes} of {arm.attempts}; the withheld check must pass on every attempt "
            "before tokens are looked at, or the hour is won by the cheapest wrong answer"
        )
    if arm.attempts < MIN_ATTEMPTS:
        from hermesbench.repeats import wilson

        bound = wilson(arm.passes, arm.attempts)
        return False, (
            f"{arm.passes}/{arm.attempts} is 100% of too few attempts: it bounds the true success "
            f"rate only to {bound.low:.1%} at 95%. {MIN_ATTEMPTS} attempts are the floor."
        )
    if len(contender.baseline.tokens) < 2:
        return False, "its baseline has fewer than two measurements, so no reduction can be bounded"
    return True, ""


def score_of(contender: Contender) -> Score:
    """Rank by the lower bound of the reduction against this submission's own baseline.

    Relative, so tasks with different bars are comparable. Lower bound rather than point estimate,
    so a large margin measured on a noisy task does not outrank a smaller one that is actually
    known -- which is the same reason `decide` gates on the bound instead of the estimate.
    """
    interval = reduction_interval(contender.baseline.tokens, contender.candidate.tokens)
    return Score(contender=contender, lower_bound=interval[0], interval=interval)


def select(contenders: list[Contender]) -> Outcome:
    """Score the field and pick one winner. Ties break on tool calls, then on arriving first."""
    ranked: list[Score] = []
    ineligible: list[tuple[Contender, str]] = []
    for contender in contenders:
        ok, why = eligibility(contender)
        if ok:
            ranked.append(score_of(contender))
        else:
            ineligible.append((contender, why))

    # Sorted, not max(): the full ordering is published so a miner can see where they placed.
    # Ties go to fewer tool calls and then to whoever submitted first -- deterministic, and it
    # rewards the earlier of two identical results rather than the later.
    ranked.sort(key=lambda s: (-s.lower_bound, s.contender.median_tool_calls, s.contender.received_at))
    ineligible.sort(key=lambda pair: (pair[0].task_id, pair[0].miner_id))

    # A winner still has to have beaten its baseline. A field where every entry is correct but no
    # cheaper has no winner: the crown is for an improvement, not for turning up.
    winner = ranked[0] if ranked and ranked[0].lower_bound > 0.0 else None
    if ranked and winner is None:
        ineligible.append(
            (
                ranked[0].contender,
                f"best of the field, and its reduction interval {ranked[0].interval} does not clear "
                "zero: correct but not an improvement anyone can distinguish from noise",
            )
        )
    return Outcome(winner=winner, ranked=ranked, ineligible=ineligible)


def label_actions(outcome: Outcome, previous: dict[str, Any]) -> list[tuple[str, int]]:
    """(action, pr) pairs for the crown label. Removals first."""
    actions: list[tuple[str, int]] = []
    old_pr = int(((previous or {}).get("winner") or {}).get("pr") or 0)
    new_pr = outcome.winner.contender.pr if outcome.winner else 0
    if old_pr and old_pr != new_pr:
        actions.append(("remove", old_pr))
    if new_pr and old_pr != new_pr:
        actions.append(("add", new_pr))
    return actions


def close_actions(outcome: Outcome, *, close_when_no_winner: bool = True) -> list[tuple[int, str]]:
    """(pr, reason) for every pull request the round is finished with.

    The winner's stays open and carries the label. With no winner the choice is a policy one --
    closing keeps the round boundary clean, not closing avoids discarding a field nobody could have
    won from -- so it is a flag rather than an assumption.
    """
    if outcome.winner is None and not close_when_no_winner:
        return []
    reasons = {c.miner_id: "the round is over" for c in outcome.losers}
    for contender, why in outcome.ineligible:
        reasons[contender.miner_id] = why
    return [
        (c.pr, reasons.get(c.miner_id, "the round is over"))
        for c in outcome.losers
        if c.pr and (outcome.winner is None or c.pr != outcome.winner.contender.pr)
    ]


def contenders_from(scorecard_dir: Path, *, store: Any = None, registry: Path | None = None) -> list[Contender]:
    """Every graded result in the current round.

    A verdict is not published until `RoundWindow.grade`, so an ungraded round yields nothing:
    crowning on one would rank miners by a result the round itself refuses to serve.
    """
    from hermes.round import GRADED, SETTLED
    from validator.store import RoundStore

    store = store or RoundStore()
    prs = pr_numbers(registry) if registry else {}
    out: list[Contender] = []
    for path in sorted(scorecard_dir.glob("*.json")) if scorecard_dir.is_dir() else []:
        card = json.loads(path.read_text(encoding="utf-8"))
        round_id, miner_id = str(card.get("round_id") or ""), str(card.get("miner_id") or "")
        try:
            window = store.load(round_id)
        except Exception:  # noqa: BLE001 - a missing round is a skip, not a failure of the tournament
            continue
        if window.state not in (GRADED, SETTLED):
            continue
        candidate, baseline = card.get("candidate") or {}, card.get("baseline") or {}
        tokens = tuple(int(t) for t in candidate.get("tokens") or ())
        base_tokens = tuple(int(t) for t in baseline.get("tokens") or ())
        if not tokens or not base_tokens:
            continue
        entry = prs.get((round_id, miner_id), {})
        out.append(
            Contender(
                task_id=str(card.get("task_id") or ""),
                miner_id=miner_id,
                round_id=round_id,
                candidate=Arm(
                    passes=int(candidate.get("verified_passes") or 0),
                    attempts=int(candidate.get("attempts") or len(tokens)),
                    tokens=tokens,
                    tool_calls=tuple(int(c) for c in candidate.get("tool_calls") or ([0] * len(tokens))),
                ),
                baseline=Arm(
                    passes=int(baseline.get("verified_passes") or 0),
                    attempts=int(baseline.get("attempts") or len(base_tokens)),
                    tokens=base_tokens,
                    tool_calls=tuple(int(c) for c in baseline.get("tool_calls") or ([0] * len(base_tokens))),
                ),
                pr=int(entry.get("pr") or 0),
                received_at=float(entry.get("received_at") or 0.0),
            )
        )
    return out


def pr_numbers(registry: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """(round_id, miner_id) -> {pr, received_at}, from the strategy registry.

    Optional on the record. A submission with no number is reported and not labelled or closed:
    guessing which pull request belongs to a miner is how the wrong one gets closed, and closing is
    not reversible by this job.
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}
    if not registry.is_file():
        return found
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("pr") or 0):
            found[(str(record.get("round_id") or ""), str(record.get("miner_id") or ""))] = {
                "pr": int(record["pr"]),
                "received_at": float(record.get("received_at") or 0.0),
            }
    return found


def render(outcome: Outcome) -> str:
    lines = [f"{len(outcome.ranked) + len(outcome.ineligible)} submission(s) evaluated this round", ""]
    if outcome.ranked:
        lines.append(f"  {'#':<3} {'miner':<14} {'task':<28} {'score':>8}  {'tokens':>9}  interval")
        for place, s in enumerate(outcome.ranked, start=1):
            c = s.contender
            lines.append(
                f"  {place:<3} {c.miner_id:<14} {c.task_id:<28} {s.lower_bound:>7.1%}  "
                f"{c.median_tokens:>9,.0f}  [{s.interval[0]:.1%}, {s.interval[1]:.1%}]"
            )
        lines.append("")
        lines.append("  score is the LOWER BOUND of the reduction against each submission's own baseline:")
        lines.append("  relative, so tasks with different bars compare, and noise-aware, so a large margin")
        lines.append("  on a variable task does not outrank a smaller one that is actually known.")
    for contender, why in outcome.ineligible:
        lines.append(f"  --  {contender.miner_id:<14} {contender.task_id:<28} {why[:70]}")

    lines.append("")
    if outcome.winner:
        w = outcome.winner
        lines.append(f"CROWN: {w.contender.miner_id} on {w.contender.task_id} (#{w.contender.pr or '?'})")
        lines.append(f"  {w.lower_bound:.1%} reduction, lower bound, over {w.contender.candidate.attempts} attempts")
    else:
        lines.append("NO CROWN this round. Nothing cleared the bar, and crowning the best of a bad")
        lines.append("field would make an hourly reward that always pays out and therefore says nothing.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["select", "actions"])
    parser.add_argument("--scorecards", type=Path, default=Path("var/scorecards"))
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=Path("datasets/strategies.jsonl"))
    parser.add_argument("--out", type=Path, default=CROWNS)
    parser.add_argument(
        "--keep-open-on-no-winner",
        action="store_true",
        help="leave losing pull requests open when nothing was eligible, instead of closing a field "
        "nobody could have won from",
    )
    args = parser.parse_args(argv)

    from validator.store import RoundStore

    try:
        found = contenders_from(args.scorecards, store=RoundStore(args.store), registry=args.registry)
    except Exception as exc:  # noqa: BLE001
        print(f"validator.crown: {exc}", file=sys.stderr)
        return 2

    outcome = select(found)
    previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}

    if args.action == "actions":
        for act, pr in label_actions(outcome, previous):
            print(f"label {act} {CROWN_LABEL} #{pr}")
        for pr, why in close_actions(outcome, close_when_no_winner=not args.keep_open_on_no_winner):
            print(f"close #{pr} {why[:110]}")
        unknown = [c.miner_id for c in outcome.losers if not c.pr]
        if outcome.winner and not outcome.winner.contender.pr:
            unknown.append(outcome.winner.contender.miner_id)
        for miner in unknown:
            print(f"  no pull request recorded for {miner}; not labelling or closing", file=sys.stderr)
        return 0

    print(render(outcome))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


__all__ = [
    "CROWNS",
    "CROWN_LABEL",
    "Contender",
    "CrownError",
    "Outcome",
    "Score",
    "close_actions",
    "contenders_from",
    "eligibility",
    "label_actions",
    "main",
    "pr_numbers",
    "render",
    "score_of",
    "select",
]


if __name__ == "__main__":
    raise SystemExit(main())

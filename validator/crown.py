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

## An hour with no winner keeps the incumbent, and two of them move the task

Nothing is crowned when nothing beats its own baseline: an hourly reward that always pays out
stops carrying information within a week. But the crown does not vacate. The previous holder keeps
it and the task runs again, because a barren hour says nothing about the miner who last cleared the
bar -- it says the field this hour did not.

Two barren hours in a row is different. That is evidence about the *task*: either nobody can beat
its baseline or nobody is trying, and running it a third time spends an hour of everyone's GPU
time to learn the same thing. So the task rotates and the counter resets.

The crown carries across a rotation. It was won and nothing has taken it, and stripping it because
the subject changed would punish the holder for other people's failure.

## The label is moved by this job and nothing else

Removals are emitted before additions. A crown only ever added is a crown several people hold at
once, and this design has exactly one at a time by construction.

The crowned pull request stays open while it holds the crown -- it is the standing result, and
closing it would make the label point at a closed page. Every challenger closes each hour.
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


BARREN_ROUNDS_BEFORE_ROTATION = 2


@dataclass(frozen=True)
class Standing:
    """What carries from one hour to the next."""

    winner: dict[str, Any] | None = None
    task_id: str = ""
    barren_rounds: int = 0

    @classmethod
    def from_record(cls, record: dict[str, Any] | None) -> Standing:
        record = record or {}
        return cls(
            winner=record.get("winner"),
            task_id=str(record.get("task_id") or ""),
            barren_rounds=int(record.get("barren_rounds") or 0),
        )

    @property
    def pr(self) -> int:
        return int((self.winner or {}).get("pr") or 0)


def next_task(current: str, available: list[str]) -> str:
    """The task after this one, cycling. Empty when there is nothing to rotate to.

    Round-robin over the sorted list rather than random: a rotation nobody can predict is a
    rotation nobody can prepare for, and the point of moving on is to give the field a task it
    might actually beat.
    """
    if not available:
        return current
    ordered = sorted(available)
    if current not in ordered:
        return ordered[0]
    return ordered[(ordered.index(current) + 1) % len(ordered)]


def settle(
    previous: dict[str, Any] | None,
    outcome: Outcome,
    *,
    available_tasks: list[str] | None = None,
) -> tuple[Standing, list[tuple[str, int]]]:
    """The next standing, and the label moves. Returns (standing, label actions).

    Three cases and they are genuinely different:

    A winner takes the crown from whoever held it, and the barren counter resets. The task stays --
    someone beat it, so it is a task worth running again.

    No winner leaves the crown where it is and increments the counter. Nothing is removed: a barren
    hour is a fact about this hour's field, not about the miner who last cleared the bar.

    Two barren hours rotate the task and reset the counter. That is evidence about the task rather
    than the field, and a third run would spend an hour of everyone's GPU time to learn the same
    thing. The crown still carries: it was won, nothing has taken it, and stripping it because the
    subject changed would punish the holder for other people's failure.
    """
    standing = Standing.from_record(previous)
    task = standing.task_id or (outcome.winner.contender.task_id if outcome.winner else "")
    if not task and outcome.ranked:
        task = outcome.ranked[0].contender.task_id

    if outcome.winner is not None:
        actions: list[tuple[str, int]] = []
        new_pr = outcome.winner.contender.pr
        if standing.pr and standing.pr != new_pr:
            actions.append(("remove", standing.pr))
        if new_pr and standing.pr != new_pr:
            actions.append(("add", new_pr))
        return Standing(winner=outcome.winner.to_record(), task_id=task, barren_rounds=0), actions

    barren = standing.barren_rounds + 1
    if barren >= BARREN_ROUNDS_BEFORE_ROTATION:
        return Standing(winner=standing.winner, task_id=next_task(task, available_tasks or []), barren_rounds=0), []
    # The incumbent keeps the label: no actions at all, so the hourly job does not re-notify them.
    return Standing(winner=standing.winner, task_id=task, barren_rounds=barren), []


def available_tasks(challenges: Path) -> list[str]:
    """Task ids with a published challenge packet, which is what a round can be opened over."""
    return sorted(p.stem for p in challenges.glob("*.json")) if challenges.is_dir() else []


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


def close_actions(outcome: Outcome, standing: Standing | None = None) -> list[tuple[int, str]]:
    """(pr, reason) for every pull request this round is finished with.

    Every challenger closes each hour. Two pull requests are spared: this hour's winner, and the
    standing crown holder if nobody took it from them -- theirs is the current result and closing
    it would leave the label pointing at a closed page.
    """
    spared = {outcome.winner.contender.pr} if outcome.winner else set()
    if standing is not None and standing.pr:
        spared.add(standing.pr)
    reasons = {c.miner_id: "the round is over" for c in outcome.losers}
    for contender, why in outcome.ineligible:
        reasons[contender.miner_id] = why
    return [(c.pr, reasons.get(c.miner_id, "the round is over")) for c in outcome.losers if c.pr and c.pr not in spared]


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
        "--challenges",
        type=Path,
        default=Path("datasets/challenges"),
        help="where the challenge packets live; the task rotates through these after two barren rounds",
    )
    args = parser.parse_args(argv)

    from validator.store import RoundStore

    try:
        found = contenders_from(args.scorecards, store=RoundStore(args.store), registry=args.registry)
    except Exception as exc:  # noqa: BLE001
        print(f"validator.crown: {exc}", file=sys.stderr)
        return 2

    outcome = select(found)
    standing_before = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    tasks = available_tasks(args.challenges)
    standing_after, labels = settle(standing_before, outcome, available_tasks=tasks)

    if args.action == "actions":
        for act, pr in labels:
            print(f"label {act} {CROWN_LABEL} #{pr}")
        for pr, why in close_actions(outcome, Standing.from_record(standing_before)):
            print(f"close #{pr} {why[:110]}")
        if not labels and standing_after.pr:
            print(f"  crown stays with #{standing_after.pr}; nothing beat it this round", file=sys.stderr)
        if standing_after.task_id != Standing.from_record(standing_before).task_id:
            print(
                f"  task rotates to {standing_after.task_id!r} after {BARREN_ROUNDS_BEFORE_ROTATION} barren round(s)",
                file=sys.stderr,
            )
        for miner in [c.miner_id for c in outcome.losers if not c.pr]:
            print(f"  no pull request recorded for {miner}; not closing", file=sys.stderr)
        return 0

    print(render(outcome))
    if outcome.winner is None and standing_after.pr:
        print(
            f"\nNo crown this round, so it stays with #{standing_after.pr}. Barren rounds: {standing_after.barren_rounds}."
        )
    if standing_after.task_id != Standing.from_record(standing_before).task_id:
        print(
            f"Task rotates to {standing_after.task_id!r}: two barren rounds is evidence about the task, not the field."
        )
    elif standing_after.task_id:
        print(f"Next round runs {standing_after.task_id!r} again.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **outcome.to_record(),
        "winner": standing_after.winner,
        "this_round_winner": outcome.winner.to_record() if outcome.winner else None,
        "task_id": standing_after.task_id,
        "barren_rounds": standing_after.barren_rounds,
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    "Standing",
    "available_tasks",
    "next_task",
    "settle",
    "main",
    "pr_numbers",
    "render",
    "score_of",
    "select",
]


if __name__ == "__main__":
    raise SystemExit(main())

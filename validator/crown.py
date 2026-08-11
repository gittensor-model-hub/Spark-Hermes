"""Who holds each task's crown, and which pull request should carry the label.

    python -m validator.crown standings --out datasets/crowns.json
    python -m validator.crown labels    --repo owner/name

One crown per task, not one overall. Tasks have different bars -- `tc-log-rotation-order`'s
baseline passes 4 of 10 at 78,417 median tokens and another task's is nothing like it -- so a
single leaderboard would rank miners by which task they drew.

## The crown is a ratchet, and that is why it is hard to take

`hermes.acceptance.dominates` decides. A challenger must pass **every** attempt, spend fewer
tokens with the reduction's interval clear of zero, and use no more tool calls. Wall time is
excluded deliberately: it is not recomputable from the trace, and a bar that only rises would lock
in whichever run got favourable scheduling, permanently, because no later run could legitimately
beat it.

An equal challenger does not take the crown. Ties keep the incumbent, so a crown changes hands
only on evidence.

## Only graded rounds are eligible

A verdict is not published until `RoundWindow.grade`, and a crown computed from ungraded rounds
would rank miners on results the round itself refuses to serve. The standings therefore read the
store and skip any round that has not been graded, saying so rather than silently omitting it --
"no crown yet" and "the round has not been graded" are different states and only the first is
about the miners.

## Why the label moves on a schedule rather than on every merge

Recomputing on every merge means the label churns during a burst of submissions, and each move is
a notification to two miners. Hourly is slow enough that the label means something and fast enough
that a new king is not waiting a day. The scheduled job is the only thing that writes the label,
so a merge never grants a crown by itself -- which also keeps the crown out of the merge path,
where a mistake would block submissions rather than mislabel one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes.acceptance import Arm, dominates

CROWNS = Path("datasets/crowns.json")
CROWN_LABEL = "crown"


class CrownError(RuntimeError):
    """Standings cannot be computed."""


@dataclass(frozen=True)
class Contender:
    """One graded result, in the shape the crown rule compares."""

    task_id: str
    miner_id: str
    round_id: str
    arm: Arm
    pr: int = 0

    @property
    def median_tokens(self) -> float:
        from statistics import median

        return median(self.arm.tokens)


@dataclass(frozen=True)
class Reign:
    task_id: str
    miner_id: str
    round_id: str
    median_tokens: float
    median_tool_calls: float
    attempts: int
    pr: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "miner_id": self.miner_id,
            "round_id": self.round_id,
            "median_tokens": self.median_tokens,
            "median_tool_calls": self.median_tool_calls,
            "attempts": self.attempts,
            "pr": self.pr,
        }


def _reign(contender: Contender) -> Reign:
    from statistics import median

    return Reign(
        task_id=contender.task_id,
        miner_id=contender.miner_id,
        round_id=contender.round_id,
        median_tokens=contender.median_tokens,
        median_tool_calls=median(contender.arm.tool_calls),
        attempts=contender.arm.attempts,
        pr=contender.pr,
    )


def contest(incumbent: Contender | None, challenger: Contender) -> tuple[bool, str]:
    """Whether the challenger takes the crown from the incumbent.

    With no incumbent the challenger still has to clear the bar on its own: `dominates` against
    itself is the wrong question, so an empty throne is contested against the challenger's own
    correctness and attempt floor. A crown handed to the first arrival regardless of quality is a
    crown that means "went first".
    """
    if incumbent is None:
        if not challenger.arm.all_passed:
            return False, "an empty throne is not a lower bar: a challenger must still pass every attempt"
        # The attempt floor, borrowed from the same rule that guards a contested crown. `dominates`
        # would apply it; with no incumbent to compare against it has to be applied here.
        from hermes.acceptance import MIN_ATTEMPTS
        from hermesbench.repeats import wilson

        if challenger.arm.attempts < MIN_ATTEMPTS:
            bound = wilson(challenger.arm.passes, challenger.arm.attempts)
            return False, (
                f"{challenger.arm.passes}/{challenger.arm.attempts} is 100% of too few attempts: it "
                f"bounds the true success rate only to {bound.low:.1%} at 95%. The crown persists and "
                f"every later challenger must beat it, so a bar set from one lucky sample does not decay."
            )
        return True, ""
    return dominates(challenger.arm, incumbent.arm)


def standings(contenders: list[Contender]) -> tuple[dict[str, Reign], list[tuple[Contender, str]]]:
    """The king of each task, and every challenger that failed with its reason.

    Contenders are applied in round order so the outcome does not depend on dict iteration: a
    ratchet whose result changes with the order it is fed is not a ratchet.
    """
    kings: dict[str, Contender] = {}
    refused: list[tuple[Contender, str]] = []
    for contender in sorted(contenders, key=lambda c: (c.task_id, c.round_id, c.miner_id)):
        took, why = contest(kings.get(contender.task_id), contender)
        if took:
            kings[contender.task_id] = contender
        else:
            refused.append((contender, why))
    return {task: _reign(c) for task, c in kings.items()}, refused


def contenders_from(scorecard_dir: Path, *, store: Any = None, registry: Path | None = None) -> list[Contender]:
    """Every graded result eligible for a crown.

    A scorecard is written whenever a submission is judged, including for rounds that were never
    graded because another submission could not be run. Those are skipped here: a verdict is not
    published until grading, and crowning on one would rank miners by a result the round itself
    refuses to serve.
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
        except Exception:  # noqa: BLE001 - a missing round is a skip, not a failure of the standings
            continue
        if window.state not in (GRADED, SETTLED):
            continue
        candidate = card.get("candidate") or {}
        tokens = tuple(int(t) for t in candidate.get("tokens") or ())
        if not tokens:
            continue
        out.append(
            Contender(
                task_id=str(card.get("task_id") or ""),
                miner_id=miner_id,
                round_id=round_id,
                arm=Arm(
                    passes=int(candidate.get("verified_passes") or 0),
                    attempts=int(candidate.get("attempts") or len(tokens)),
                    tokens=tokens,
                    tool_calls=tuple(int(c) for c in candidate.get("tool_calls") or ([0] * len(tokens))),
                ),
                pr=int(prs.get((round_id, miner_id), 0)),
            )
        )
    return out


def pr_numbers(registry: Path) -> dict[tuple[str, str], int]:
    """(round_id, miner_id) -> pull request number, from the strategy registry.

    Optional on the record: a submission opened before the field existed simply has no number, and
    the label step reports it rather than guessing. Guessing which pull request belongs to a miner
    is how the wrong one gets labelled.
    """
    found: dict[tuple[str, str], int] = {}
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
        pr = int(record.get("pr") or 0)
        if pr:
            found[(str(record.get("round_id") or ""), str(record.get("miner_id") or ""))] = pr
    return found


def label_actions(kings: dict[str, Reign], previous: dict[str, Any]) -> list[tuple[str, int, str]]:
    """What to do to which pull request. Returns (action, pr, task) with action add or remove.

    Removals are emitted before additions by the caller's ordering, because a crown that is only
    ever added is a crown several people hold at once -- and the label is supposed to mean one.
    """
    actions: list[tuple[str, int, str]] = []
    old = {str(t): r for t, r in (previous.get("crowns") or {}).items()}
    for task, reign in sorted(kings.items()):
        was = old.get(task) or {}
        old_pr, new_pr = int(was.get("pr") or 0), reign.pr
        if old_pr and old_pr != new_pr:
            actions.append(("remove", old_pr, task))
        if new_pr and old_pr != new_pr:
            actions.append(("add", new_pr, task))
    for task, was in sorted(old.items()):
        if task not in kings and int(was.get("pr") or 0):
            # A task whose crown vanished -- its round was rolled back, or the scorecard removed.
            actions.append(("remove", int(was["pr"]), task))
    return actions


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["standings", "labels"])
    parser.add_argument("--scorecards", type=Path, default=Path("var/scorecards"))
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=Path("datasets/strategies.jsonl"))
    parser.add_argument("--out", type=Path, default=CROWNS)
    args = parser.parse_args(argv)

    from validator.store import RoundStore

    try:
        found = contenders_from(args.scorecards, store=RoundStore(args.store), registry=args.registry)
    except Exception as exc:  # noqa: BLE001
        print(f"validator.crown: {exc}", file=sys.stderr)
        return 2

    kings, refused = standings(found)
    previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}

    if args.action == "labels":
        actions = label_actions(kings, previous)
        for act, pr, task in actions:
            print(f"{act} {CROWN_LABEL} #{pr}  ({task})")
        missing = [r for r in kings.values() if not r.pr]
        for reign in missing:
            print(f"  no pull request recorded for {reign.miner_id} on {reign.task_id}; not labelling", file=sys.stderr)
        return 0

    print(f"{len(found)} graded result(s) over {len({c.task_id for c in found})} task(s)")
    for task, reign in sorted(kings.items()):
        print(
            f"  CROWN  {task:<30} {reign.miner_id:<12} {reign.median_tokens:>9,.0f} tokens  {reign.attempts} attempts"
        )
    for contender, why in refused:
        print(f"  --     {contender.task_id:<30} {contender.miner_id:<12} {why[:80]}")
    if not kings:
        print("\nno crowns: nothing has cleared the bar. A crown handed out anyway would mean 'went first'.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"crowns": {t: r.to_record() for t, r in kings.items()}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


__all__ = [
    "CROWNS",
    "CROWN_LABEL",
    "Contender",
    "CrownError",
    "Reign",
    "contenders_from",
    "contest",
    "label_actions",
    "main",
    "pr_numbers",
    "standings",
]


if __name__ == "__main__":
    raise SystemExit(main())

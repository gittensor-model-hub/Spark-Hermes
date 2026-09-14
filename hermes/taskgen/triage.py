"""Which generated tasks are worth keeping, decided by running them rather than by the gate.

    python -m hermes.taskgen.triage --episodes var/probe/episodes.jsonl --repeats 8

`gate` proves a task is WELL-FORMED. It cannot prove the task is WORTH SOLVING, and the difference
is the whole yield of a generation run. Measured on a 320-episode probe of 157 generated tasks:

    easy   2/2 pass    117  (75%)

Three quarters were free to pass and therefore worth nothing to train on, and every one of them had
cleared the gate. Nothing in this package could tell them apart until the pinned model had run them,
which is what this module reads.

## Why the old probe could not answer this

The probe ran `--repeats 2` and the classification above is a count out of two. Put through this
repository's own `hermesbench.repeats.wilson`:

    2/2   true pass rate in [0.34, 1.00]      0/2   true pass rate in [0.00, 0.66]

"Easy, solved 2/2" is consistent with a task the model passes a third of the time, and "hard, 0/2"
with one it passes two thirds of the time. Neither the 75% nor the 27 "hard" tasks were established
by that run. `hermes.challenge` already refuses to open a challenge on a count for exactly this
reason -- "failed once" and "fails reliably" are the same integer and very different facts -- and
that reasoning was never applied to the probe that feeds it.

## The band

A task is useful for training when the model sometimes succeeds and sometimes does not: an all-pass
task supplies no gradient and an all-fail task supplies no reachable example. So the rule is the
standard one -- keep `1 <= passes < repeats` -- and the classification is reported with the interval
that produced it, so a reader can see how much the number rests on.

`MIN_REPEATS` is 8 rather than 2 because the band is only meaningful once a single lucky or unlucky
attempt cannot move a task across it. At two attempts every task is 0, 1 or 2 and the middle is a
coin flip; at eight, `TRIVIAL` means eight consecutive passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermesbench.repeats import wilson

# Below this, the classification is a statement about sampling rather than about the task.
MIN_REPEATS = 8

TRIVIAL = "trivial"
FRONTIER = "frontier"
IMPOSSIBLE = "impossible"
UNDERSAMPLED = "undersampled"


@dataclass(frozen=True)
class TaskTriage:
    """One task's verdict, with the evidence that produced it."""

    task_id: str
    passes: int
    attempts: int
    verdict: str
    low: float
    high: float

    @property
    def keep(self) -> bool:
        return self.verdict == FRONTIER

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passes": self.passes,
            "attempts": self.attempts,
            "verdict": self.verdict,
            "true_pass_rate_interval": [round(self.low, 4), round(self.high, 4)],
        }


def classify(task_id: str, passes: int, attempts: int, *, min_repeats: int = MIN_REPEATS) -> TaskTriage:
    """One task's counts into a verdict.

    `UNDERSAMPLED` is a verdict rather than a silent pass-through. A task measured over three
    attempts is not a trivial task and not a frontier task, it is an unmeasured one, and reporting
    it as either would put a guess where the interval belongs.
    """
    if attempts <= 0:
        raise ValueError(f"{task_id}: no attempts, so there is nothing to classify")
    interval = wilson(passes, attempts)
    if attempts < min_repeats:
        verdict = UNDERSAMPLED
    elif passes == 0:
        verdict = IMPOSSIBLE
    elif passes >= attempts:
        verdict = TRIVIAL
    else:
        verdict = FRONTIER
    return TaskTriage(
        task_id=task_id, passes=passes, attempts=attempts, verdict=verdict, low=interval.low, high=interval.high
    )


def _succeeded(row: dict[str, Any]) -> bool:
    """Whether one episode counts as a pass, read the way the rest of the repository reads it.

    Deliberately routed through `hermes.challenge.episode_metrics_of` rather than reading a key: an
    episode that passed the published check and failed the withheld one is NOT a success, and a
    triage that counted it as one would classify an overfit-only task as frontier -- the exact
    strategy `overfit_rate` exists to catch, promoted into the corpus by the filter meant to
    protect it.
    """
    from hermes.challenge import episode_metrics_of

    metrics = episode_metrics_of(row)
    if metrics.get("setup_failed") is True:
        raise ValueError("setup failed")
    public = bool(metrics.get("public_passed"))
    hidden = metrics.get("hidden_passed")
    return public and hidden is not False


def triage_episodes(path: Path, *, min_repeats: int = MIN_REPEATS) -> tuple[list[TaskTriage], dict[str, int]]:
    """Read an episode log into per-task verdicts, and count what it could not read.

    Unreadable rows are counted, never guessed at. An episode whose setup failed says nothing about
    the model and folding it in as a failure would push a sound task toward `IMPOSSIBLE` for a
    reason that is the harness's fault.
    """
    passes: dict[str, int] = defaultdict(int)
    attempts: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped["unparseable_row"] += 1
            continue
        task_id = str(row.get("task_id") or (row.get("task") or {}).get("task_id") or "")
        if not task_id:
            skipped["no_task_id"] += 1
            continue
        try:
            ok = _succeeded(row)
        except Exception as exc:  # noqa: BLE001 - a row this cannot read is counted, not guessed at
            skipped["setup_failed" if "setup failed" in str(exc) else "unreadable_metrics"] += 1
            continue
        attempts[task_id] += 1
        passes[task_id] += int(ok)
        _ = number

    verdicts = [classify(t, passes[t], attempts[t], min_repeats=min_repeats) for t in sorted(attempts)]
    return verdicts, dict(skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, required=True, help="probe episode log (JSONL)")
    parser.add_argument(
        "--repeats",
        type=int,
        default=MIN_REPEATS,
        help="attempts per task the probe was run with; below this a verdict is reported as undersampled",
    )
    parser.add_argument("--report", type=Path, default=None, help="write the verdicts as JSON")
    parser.add_argument("--keep-list", type=Path, default=None, help="write the frontier task ids, one per line")
    args = parser.parse_args(argv)

    if not args.episodes.is_file():
        print(f"hermes.taskgen.triage: no episode log at {args.episodes}", file=sys.stderr)
        return 2

    verdicts, skipped = triage_episodes(args.episodes, min_repeats=args.repeats)
    if not verdicts:
        print("hermes.taskgen.triage: the log held no readable episodes", file=sys.stderr)
        return 1

    counts: dict[str, int] = defaultdict(int)
    for verdict in verdicts:
        counts[verdict.verdict] += 1

    print(f"{len(verdicts)} task(s) over {sum(v.attempts for v in verdicts)} episode(s)")
    for name in (FRONTIER, TRIVIAL, IMPOSSIBLE, UNDERSAMPLED):
        if counts.get(name):
            print(f"  {counts[name]:4}  {name}")
    for reason, count in sorted(skipped.items()):
        print(f"  {count:4}  skipped: {reason}", file=sys.stderr)

    under = counts.get(UNDERSAMPLED, 0)
    if under:
        print(
            f"\n{under} task(s) were measured over fewer than {args.repeats} attempts, so their verdict is "
            "an interval too wide to act on. Re-probe those before discarding any of them: at two "
            "attempts, 2/2 leaves the true pass rate anywhere up to 100% and 0/2 anywhere up to 66%.",
            file=sys.stderr,
        )

    report = {
        "tasks": [v.to_record() for v in verdicts],
        "counts": dict(counts),
        "skipped": skipped,
        "min_repeats": args.repeats,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    if args.keep_list:
        args.keep_list.parent.mkdir(parents=True, exist_ok=True)
        args.keep_list.write_text("\n".join(v.task_id for v in verdicts if v.keep) + "\n", encoding="utf-8")
        print(f"wrote {args.keep_list}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

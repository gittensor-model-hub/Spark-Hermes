"""Turn settled rounds into the private SFT and preference sets.

    python -m validator.aggregate --rounds var/rounds --out var/datasets

The last step of the loop. A round produces verified episodes and failed ones on the same task
with the same pinned model; that pairing is exactly what preference training wants, and it is
thrown away if nobody collects it.

## Only verified episodes become SFT rows, and overfit ones become negatives

An episode counts when it passed the published check *and* the withheld one did not fail --
`Attempt.verified`, the same rule the scorecard uses. Training on episodes that merely passed the
published half would teach the model the assertions it was shown, which is precisely the failure
`overfit_rate` exists to measure.

They are not discarded, though. An episode that passed the visible check and failed the withheld
one is the sharpest negative in the corpus -- it is what fitting the benchmark actually looks like
-- so it is kept as a rejected example. Excluded from imitation, retained for contrast.

## Preference pairs come from the same task, same round, same model

A pair is (verified, unverified) on one task inside one round. That holds the task, the pinned
model, the harness and the surface fixed, so the only thing separating chosen from rejected is
what the model did -- which is the only difference a preference model can usefully learn.

Pairing across rounds would put two different surfaces on either side and teach a comparison
nobody asked for. Pairing across tasks would teach that one task is better than another.

Pairs are capped per task. One task with forty verified and forty failed episodes yields 1,600
pairs and would dominate a set built from a dozen tasks with two apiece, so the cap is stated,
applied, and reported -- a silent top-N reads as "everything" when it is not.

## It refuses a log with no trajectories rather than emitting nothing

`--episodes-out` writes counts unless `--keep-trajectories` was passed, and a log without them
produces zero rows. Zero rows and "this run was not recorded for training" look identical in an
empty output file, so the second is said out loud.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SFT_FILE = "sft.jsonl"
PREFERENCE_FILE = "preference.jsonl"
MAX_PAIRS_PER_TASK = 32


class AggregateError(RuntimeError):
    """A dataset cannot be built from what was recorded."""


@dataclass
class Episode:
    """One recorded episode, with enough to decide what it is worth."""

    task_id: str
    round_id: str
    miner_id: str
    verified: bool
    overfit: bool
    tokens: int
    trajectory: dict[str, Any]

    @property
    def usable(self) -> bool:
        return bool(self.trajectory)


@dataclass
class Summary:
    sft_rows: int = 0
    pairs: int = 0
    episodes_read: int = 0
    without_trajectory: int = 0
    overfit_skipped: int = 0
    capped: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "sft_rows": self.sft_rows,
            "preference_pairs": self.pairs,
            "episodes_read": self.episodes_read,
            # Reported rather than folded into a total. An episode with no trajectory was recorded
            # without `--keep-trajectories`; it is missing data, not a failed episode, and the two
            # would be indistinguishable in a single count.
            "episodes_without_trajectory": self.without_trajectory,
            # Excluded from SFT and *kept* as rejected examples. An episode that passed the
            # published check and failed the withheld one is the sharpest negative there is: it is
            # what fitting the visible assertions looks like. Training on it teaches that; training
            # against it teaches the opposite.
            "overfit_episodes_excluded_from_sft": self.overfit_skipped,
            "tasks_capped": sorted(self.capped),
        }


def read_episodes(path: Path, *, round_id: str, miner_id: str) -> list[Episode]:
    """Read one episode log into the shape aggregation needs."""
    from hermes.challenge import episode_metrics_of
    from hermesbench.sink import read_episodes as read_lines

    out: list[Episode] = []
    for row in read_lines(path):
        metrics = episode_metrics_of(row)
        public = bool(metrics.get("public_passed"))
        hidden = metrics.get("hidden_passed")
        out.append(
            Episode(
                task_id=str(metrics.get("task_id") or row.get("task_id") or ""),
                round_id=round_id,
                miner_id=miner_id,
                # `Attempt.verified`: passed what is published AND was not caught by what is
                # withheld. `hidden is None` means the task declares no withheld check, which is
                # not the same as having failed one.
                verified=public and hidden is not False,
                overfit=public and hidden is False,
                tokens=int(metrics.get("tokens_used") or 0),
                trajectory=row.get("trajectory") or {},
            )
        )
    return out


def sft_rows(episodes: list[Episode], *, system_policy: str = "keep") -> list[dict[str, Any]]:
    """One messages record per verified episode.

    Rendered by `hermes.format.to_messages_record`, which is what the training track already reads.
    Writing a second renderer here would give two definitions of what a trajectory looks like as
    training data, and they would drift on the first protocol change.
    """
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if not episode.verified or not episode.usable:
            continue
        trajectory = AgentTrajectory.from_record(episode.trajectory)
        record = to_messages_record(trajectory, system_policy=system_policy)
        rows.append(
            {
                **record,
                "task_id": episode.task_id,
                "round_id": episode.round_id,
                "miner_id": episode.miner_id,
            }
        )
    return rows


def preference_pairs(
    episodes: list[Episode],
    *,
    max_per_task: int = MAX_PAIRS_PER_TASK,
) -> tuple[list[dict[str, Any]], list[str]]:
    """(chosen, rejected) on the same task in the same round. Returns (pairs, tasks that hit the cap).

    Rejected episodes are ordered most-expensive first and chosen cheapest first, so a capped task
    keeps the pairs with the widest separation. A cap that kept an arbitrary slice would spend the
    budget on pairs the model can learn least from.
    """
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    by_task: dict[tuple[str, str], list[Episode]] = {}
    for episode in episodes:
        if episode.usable:
            by_task.setdefault((episode.round_id, episode.task_id), []).append(episode)

    pairs: list[dict[str, Any]] = []
    capped: list[str] = []
    for (round_id, task_id), group in sorted(by_task.items()):
        chosen = sorted([e for e in group if e.verified], key=lambda e: e.tokens)
        rejected = sorted([e for e in group if not e.verified], key=lambda e: -e.tokens)
        if not chosen or not rejected:
            continue

        made = 0
        for good in chosen:
            for bad in rejected:
                if made >= max_per_task:
                    break
                pairs.append(
                    {
                        "task_id": task_id,
                        "round_id": round_id,
                        "chosen": to_messages_record(AgentTrajectory.from_record(good.trajectory))["messages"],
                        "rejected": to_messages_record(AgentTrajectory.from_record(bad.trajectory))["messages"],
                        "chosen_tokens": good.tokens,
                        "rejected_tokens": bad.tokens,
                    }
                )
                made += 1
            if made >= max_per_task:
                break
        if made >= max_per_task and len(chosen) * len(rejected) > max_per_task:
            capped.append(task_id)
    return pairs, capped


def aggregate(
    episodes: list[Episode],
    out: Path,
    *,
    max_per_task: int = MAX_PAIRS_PER_TASK,
) -> Summary:
    """Write both datasets. Refuses a corpus with no trajectories rather than emitting nothing."""
    summary = Summary(episodes_read=len(episodes))
    summary.without_trajectory = sum(1 for e in episodes if not e.usable)
    summary.overfit_skipped = sum(1 for e in episodes if e.overfit)

    if episodes and summary.without_trajectory == len(episodes):
        raise AggregateError(
            "no episode carries a trajectory. The logs were written without --keep-trajectories, so "
            "they hold counts only and nothing can be built from them. An empty output file would "
            "look the same as a run that produced nothing worth training on."
        )

    rows = sft_rows(episodes)
    pairs, capped = preference_pairs(episodes, max_per_task=max_per_task)
    summary.sft_rows = len(rows)
    summary.pairs = len(pairs)
    summary.capped = capped

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / SFT_FILE, rows)
    _write_jsonl(out / PREFERENCE_FILE, pairs)
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def collect(
    *,
    rounds: Path | None = None,
    episode_root: Path | None = None,
    store: Any = None,
) -> list[Episode]:
    """Every episode from every settled round, with the round and miner it belongs to.

    Only SETTLED rounds. A round still open or frozen may yet change, and a dataset built from one
    would have to be rebuilt when it did -- silently, because nothing records which rounds a
    dataset was made from.
    """
    from hermes.round import SETTLED
    from validator.store import RoundStore

    store = store or RoundStore(rounds)
    root = episode_root or Path("var/judge")
    out: list[Episode] = []
    for round_id in store.round_ids():
        try:
            window = store.load(round_id)
        except Exception:  # noqa: BLE001 - one unreadable round must not stop the aggregation
            continue
        if window.state != SETTLED:
            continue
        for miner_id in sorted(window.submissions):
            log = root / round_id / f"{miner_id}.jsonl"
            if log.is_file():
                out.extend(read_episodes(log, round_id=round_id, miner_id=miner_id))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=Path("var/judge"))
    parser.add_argument("--out", type=Path, default=Path("var/datasets"))
    parser.add_argument("--max-pairs-per-task", type=int, default=MAX_PAIRS_PER_TASK)
    args = parser.parse_args(argv)

    try:
        episodes = collect(rounds=args.rounds, episode_root=args.episodes)
        summary = aggregate(episodes, args.out, max_per_task=args.max_pairs_per_task)
    except AggregateError as exc:
        print(f"validator.aggregate: {exc}", file=sys.stderr)
        return 2

    record = summary.to_record()
    print(f"read {record['episodes_read']} episode(s) from settled rounds")
    print(f"  {record['sft_rows']} SFT row(s)          -> {args.out / SFT_FILE}")
    print(f"  {record['preference_pairs']} preference pair(s) -> {args.out / PREFERENCE_FILE}")
    if record["episodes_without_trajectory"]:
        print(
            f"  {record['episodes_without_trajectory']} episode(s) carried no trajectory; those runs "
            "were recorded without --keep-trajectories"
        )
    if record["overfit_episodes_excluded_from_sft"]:
        print(
            f"  {record['overfit_episodes_excluded_from_sft']} episode(s) passed the published check "
            "and failed the withheld one: kept out of SFT, kept as rejected examples. Training on "
            "them teaches the visible assertions; training against them teaches the opposite."
        )
    for task in record["tasks_capped"]:
        print(f"  {task}: hit the per-task pair cap of {args.max_pairs_per_task}")
    return 0


__all__ = [
    "MAX_PAIRS_PER_TASK",
    "PREFERENCE_FILE",
    "SFT_FILE",
    "AggregateError",
    "Episode",
    "Summary",
    "aggregate",
    "collect",
    "main",
    "preference_pairs",
    "read_episodes",
    "sft_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())

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
from statistics import median
from typing import Any

SFT_FILE = "sft.jsonl"
PREFERENCE_FILE = "preference.jsonl"
MAX_PAIRS_PER_TASK = 32

# Verified episodes a task needs before an efficiency pair means anything. Below four, the "typical"
# cost is a couple of samples and any bar derived from it is arbitrary.
MIN_VERIFIED_FOR_EFFICIENCY = 4

# How much more than a typical correct solution the rejected side must cost. Measured on this suite,
# two verified solutions to one task differ by 1.02x to 2.01x, so this cut keeps roughly the upper
# half of the observed range and drops the rest as sampling noise.
#
# Compared against the MEDIAN, not against the cheapest episode. An earlier version of this compared
# the min-to-max range against the interquartile spread, which fires on almost any task: the extremes
# sit outside the quartiles by construction, and the range grows with the number of samples while the
# quartile band does not. A flat task whose attempts all cost within 2% of each other produced a pair.
EFFICIENCY_MARGIN = 0.5


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
    # Which wire dialect produced it. Carried because it decides the SHAPE of the training row, not
    # only its markup: reasoning belongs in `content` for one dialect and in `reasoning_content` for
    # another, and tool arguments are a JSON string for one and a mapping for another. A corpus
    # rendered in the wrong shape does not look wrong -- it raises in the trainer's chat template, or
    # silently drops every reasoning block. See `hermes.protocol.Dialect`.
    dialect: str = ""
    # Whether the harness cut this episode off at a step budget. Load-bearing for pair construction:
    # a truncated trajectory is censored mid-work, so it is neither a model to imitate nor an honest
    # measurement of what the attempt cost. See `preference_pairs`.
    truncated: bool = False

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
    reasoning_markup_stripped: int = 0
    truncated_skipped: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "sft_rows": self.sft_rows,
            "preference_pairs": self.pairs,
            "episodes_read": self.episodes_read,
            # Reported rather than folded into a total. An episode with no trajectory was recorded
            # without `--keep-trajectories`; it is missing data, not a failed episode, and the two
            # would be indistinguishable in a single count.
            "episodes_without_trajectory": self.without_trajectory,
            # Complete call blocks removed from reasoning that was about to be trained onto this
            # dialect's private-deliberation channel. Non-zero means the episodes predate the
            # harness fix that stopped recording them; the rows are clean, the logs are not.
            "reasoning_markup_stripped": self.reasoning_markup_stripped,
            # Verified episodes held back because the harness cut them off mid-work. Reported rather
            # than silently dropped: a corpus that is smaller than the verified count needs to say
            # why, and a rising number here means the action budgets are too tight.
            "truncated_skipped": self.truncated_skipped,
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
                dialect=str(metrics.get("dialect") or ""),
                truncated=bool(metrics.get("max_steps_hit")),
            )
        )
    return out


def _render_context(episodes: list[Episode]) -> dict[str, Any]:
    """The dialect every usable episode was run in, and the schemas its rows will need.

    Refused when the episodes disagree. One corpus file is trained with one chat template, so mixing
    dialects means half the rows are shaped for a template that will not render them -- and the half
    that fails is decided by which dialect the trainer's template happens to be. `check_graders`
    refuses a cross-dialect comparison for the same reason; this is that refusal on the corpus side.

    An empty dialect keeps the Hermes default, so a log recorded before the field existed still
    aggregates instead of failing on absence.
    """
    from hermes.protocol import DIALECTS, HERMES_4

    seen = {e.dialect for e in episodes if e.usable and e.dialect}
    if len(seen) > 1:
        raise AggregateError(
            f"the episodes were run in more than one wire dialect: {sorted(seen)}. The dialect decides "
            "the shape of a training row, so one corpus cannot hold both"
        )
    dialect = DIALECTS.get(next(iter(seen), ""), HERMES_4)
    schemas: dict[str, dict[str, Any]] | None = None
    if not dialect.tools_in_prompt:
        # This dialect's template renders tool definitions from the row, so bare names are not
        # enough. Read from the committed set, which is the one the model was actually shown.
        from hermes.pin import load_tool_schemas
        from hermesbench.runner import HARNESS_DIR

        schemas = load_tool_schemas(HARNESS_DIR / "tools.json")
    return {"dialect": dialect, "tool_schemas": schemas}


def sft_rows(episodes: list[Episode], *, system_policy: str = "keep") -> list[dict[str, Any]]:
    """One messages record per verified episode.

    Rendered by `hermes.format.to_messages_record`, which is what the training track already reads.
    Writing a second renderer here would give two definitions of what a trajectory looks like as
    training data, and they would drift on the first protocol change.
    """
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    context = _render_context(episodes)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if not episode.verified or not episode.usable:
            continue
        if episode.truncated:
            # SFT is pure imitation, so this matters more here than in `preference_pairs` -- where a
            # truncated episode is already barred from the chosen side. A trajectory the harness cut
            # off at the step budget ends mid-work, and its last recorded step is the harness saying
            # so, not the model finishing. Training on it teaches stopping short.
            #
            # Measured on a 152-episode run: 18 of 143 verified episodes were truncated, concentrated
            # on four tasks whose action budgets are still tight. Emitting them from here while
            # excluding them there was an inconsistency, and the imitation signal is the stronger one.
            continue
        trajectory = AgentTrajectory.from_record(episode.trajectory)
        record = to_messages_record(trajectory, system_policy=system_policy, **context)
        rows.append(
            {
                **record,
                "task_id": episode.task_id,
                "round_id": episode.round_id,
                "miner_id": episode.miner_id,
            }
        )
    return rows


def _efficiency_pairs(
    verified: list[Episode], *, round_id: str, task_id: str, context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Cheapest correct solution against the most expensive, when the gap is bigger than the noise.

    At a high pass rate there is no correctness signal left -- every attempt passes -- and the whole
    point of this project is that the same model can solve the same task for fewer tokens. That is
    what the promotion gate scores, so it is worth training toward.

    The trap is pairing on sampling noise. Measured on this suite, two verified solutions to the same
    task differ by anywhere from 1.02x to 2.01x, so a fixed ratio would either fire on noise or never
    fire. The threshold comes from the task's own distribution instead: the pair must be separated by
    more than the interquartile spread of that task's verified episodes. A task whose attempts all
    cost about the same produces nothing, which is correct -- there is no lesson in it.
    """
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    if len(verified) < MIN_VERIFIED_FOR_EFFICIENCY:
        return []
    costs = sorted(e.tokens for e in verified)
    if costs[0] <= 0:
        # Unpriced episodes: every token count is 0, so every gap is 0 and every pair would look
        # infinitely good. Refused rather than ranked on a field that was never populated -- and
        # `mean_tokens` was a column of zeros in this repo once, so that is a real failure mode.
        return []
    typical = median(costs)
    cheapest, dearest = verified[0], verified[-1]
    # The rejected side has to be clearly worse than typical, and the chosen side at least as good as
    # typical. Requiring both to be atypical would reject the common shape, where one attempt wandered
    # and the rest were ordinary -- and that wandering attempt is exactly the lesson.
    if dearest.tokens < typical * (1 + EFFICIENCY_MARGIN) or cheapest.tokens > typical:
        return []
    return [
        {
            "task_id": task_id,
            "round_id": round_id,
            "chosen": to_messages_record(AgentTrajectory.from_record(cheapest.trajectory), **context)["messages"],
            "rejected": to_messages_record(AgentTrajectory.from_record(dearest.trajectory), **context)["messages"],
            "chosen_tokens": cheapest.tokens,
            "rejected_tokens": dearest.tokens,
            "kind": "efficiency",
            # So a reader can see the bar this pair cleared rather than trusting that one existed.
            "typical_tokens": typical,
            "efficiency_margin": EFFICIENCY_MARGIN,
            "verified_episodes": len(verified),
        }
    ]


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

    context = _render_context(episodes)
    by_task: dict[tuple[str, str], list[Episode]] = {}
    for episode in episodes:
        if episode.usable:
            by_task.setdefault((episode.round_id, episode.task_id), []).append(episode)

    pairs: list[dict[str, Any]] = []
    capped: list[str] = []
    for (round_id, task_id), group in sorted(by_task.items()):
        # A truncated episode is never `chosen`: its trajectory stops mid-work at the step budget, so
        # imitating it teaches an agent to stop before finishing. Measured reason this matters -- two
        # runs of this suite had 53% and 56% of episodes truncated, and 5 of 9 capped episodes had
        # already passed, so "verified" and "complete" are not the same thing.
        chosen = sorted([e for e in group if e.verified and not e.truncated], key=lambda e: e.tokens)
        rejected = sorted([e for e in group if not e.verified], key=lambda e: -e.tokens)
        made = 0
        if not chosen:
            continue
        if not rejected:
            # Every attempt passed, so there is no correctness signal here -- and at a 94.7% suite
            # pass rate that is most tasks, which is why this branch exists at all. What is left is
            # the efficiency signal the promotion gate actually scores: two correct solutions to one
            # task, one of them far cheaper.
            for pair in _efficiency_pairs(chosen, round_id=round_id, task_id=task_id, context=context):
                if made >= max_per_task:
                    break
                pairs.append(pair)
                made += 1
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
                        "chosen": to_messages_record(AgentTrajectory.from_record(good.trajectory), **context)[
                            "messages"
                        ],
                        "rejected": to_messages_record(AgentTrajectory.from_record(bad.trajectory), **context)[
                            "messages"
                        ],
                        "chosen_tokens": good.tokens,
                        "rejected_tokens": bad.tokens,
                        # Which signal produced this pair. A correctness pair and an efficiency pair
                        # teach different things and a trainer may want to weight them differently;
                        # a corpus that does not say which is which cannot be reweighted afterwards.
                        "kind": "correctness",
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
    # Counted off the rows themselves rather than tracked through the renderer, so the number always
    # describes what was actually written.
    summary.truncated_skipped = sum(1 for e in episodes if e.verified and e.usable and e.truncated)
    summary.reasoning_markup_stripped = sum(
        int(message.get("reasoning_markup_stripped") or 0) for row in rows for message in row["messages"]
    ) + sum(
        int(message.get("reasoning_markup_stripped") or 0)
        for pair in pairs
        for side in ("chosen", "rejected")
        for message in pair[side]
    )

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

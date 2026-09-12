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
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def task_key(self) -> str:
        from admin.artifacts import content_digest

        return (
            content_digest({k: self.provenance.get(k) for k in ("task_version", "epoch", "origin", "model_revision")})
            + self.task_id
        )

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
    not_best_skipped: int = 0

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
            "not_best_skipped": self.not_best_skipped,
            # Excluded from SFT and *kept* as rejected examples. An episode that passed the
            # published check and failed the withheld one is the sharpest negative there is: it is
            # what fitting the visible assertions looks like. Training on it teaches that; training
            # against it teaches the opposite.
            "overfit_episodes_excluded_from_sft": self.overfit_skipped,
            "tasks_capped": sorted(self.capped),
        }


def read_episodes(path: Path, *, round_id: str, miner_id: str) -> list[Episode]:
    """Legacy rendering/inspection helper; does not grant corpus admission authority."""
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


def admit_episode(
    row: dict[str, Any], *, round_id: str, miner_id: str, private_required: bool, provenance: dict[str, Any]
) -> Episode:
    """Shared operator/competition ingress: strict evidence and complete executed steps."""
    from hermes.trajectory import AgentTrajectory, validate
    from validator.score import normalize_episode, validate_execution

    metrics = normalize_episode(row)
    validate_execution(row, private_required=private_required)
    raw = row.get("trajectory")
    if not isinstance(raw, dict) or raw.get("metadata", {}).get("executed") is not True:
        raise AggregateError("episode has no executed trajectory")
    if raw.get("task_id") != metrics.get("task_id") or type(raw.get("success")) is not bool:
        raise AggregateError("trajectory task/success identity is missing or contradictory")
    if not isinstance(raw.get("tools_available"), list) or not raw["tools_available"]:
        raise AggregateError("trajectory must retain the tools offered during execution")
    if raw.get("schema_version") != 1 or type(raw["schema_version"]) is not int:
        raise AggregateError("unsupported trajectory schema")
    steps = raw.get("steps")
    if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
        raise AggregateError("trajectory must retain complete executed steps")
    for step in steps:
        if "content" in step and not isinstance(step["content"], str):
            raise AggregateError("trajectory content must be text")
        if step.get("kind") == "tool_result" and type(step.get("ok")) is not bool:
            raise AggregateError("trajectory tool result needs an observed boolean outcome")
    trajectory = AgentTrajectory.from_record(raw)
    validate(trajectory)
    if len(trajectory.tool_calls) != metrics["tool_calls"] or len(trajectory.steps) != metrics["steps"]:
        raise AggregateError("trajectory length/tool calls disagree with execution measurements")
    if not isinstance(metrics.get("dialect"), str) or not metrics["dialect"]:
        raise AggregateError("missing executed dialect")
    return Episode(
        task_id=metrics["task_id"],
        round_id=round_id,
        miner_id=miner_id,
        verified=metrics["success"],
        overfit=metrics["public_passed"] and metrics.get("hidden_passed") is False,
        tokens=metrics["tokens_used"],
        trajectory=raw,
        dialect=metrics["dialect"],
        provenance=provenance,
    )


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


def sft_rows(episodes: list[Episode], *, system_policy: str = "keep", best_only: bool = True) -> list[dict[str, Any]]:
    """The BEST verified episode per task, rendered as a messages record.

    Rendered by `hermes.format.to_messages_record`, which is what the training track already reads.
    Writing a second renderer here would give two definitions of what a trajectory looks like as
    training data, and they would drift on the first protocol change.

    `best_only` is rejection sampling and is on by default. Keeping every verified attempt sounds
    like more data and is not: measured on a real 8-repeat run, one task contributed 8 rows spanning
    17,779 to 38,507 tokens -- the same task, solved the same way, eight times. SFT is imitation, so
    that corpus teaches the model that the 38k path is as good as the 17k one, and pays for the
    lesson in rows.

    Cheapest verified attempt wins, which is the same quantity the promotion gate scores. The losers
    are not wasted: `preference_pairs` puts them on the rejected side, which is where a worse-but-
    correct trajectory is actually worth something.
    """
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    context = _render_context(episodes)
    if best_only:
        best: dict[str, Episode] = {}
        for episode in episodes:
            if not episode.verified or not episode.usable or episode.truncated or episode.tokens <= 0:
                continue
            current = best.get(episode.task_key)
            if current is None or episode.tokens < current.tokens:
                best[episode.task_key] = episode
        chosen = {id(episode) for episode in best.values()}
        episodes = [episode for episode in episodes if id(episode) in chosen]

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if not episode.verified or not episode.usable or episode.tokens <= 0:
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
                "metadata": {"executed": trajectory.metadata.get("executed") is True},
                **({"provenance": [episode.provenance]} if episode.provenance else {}),
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
            **({"provenance": [cheapest.provenance, dearest.provenance]} if cheapest.provenance else {}),
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

    if max_per_task < 1:
        raise AggregateError("max_per_task must be positive")
    context = _render_context(episodes)
    by_task: dict[tuple[str, str, str, str], list[Episode]] = {}
    for episode in episodes:
        if episode.usable:
            key = (episode.round_id, episode.task_id, episode.task_key, episode.provenance.get("bundle_sha256", ""))
            by_task.setdefault(key, []).append(episode)

    pairs: list[dict[str, Any]] = []
    capped: list[str] = []
    for (round_id, task_id, _, _bundle), group in sorted(by_task.items()):
        # A truncated episode is never `chosen`: its trajectory stops mid-work at the step budget, so
        # imitating it teaches an agent to stop before finishing. Measured reason this matters -- two
        # runs of this suite had 53% and 56% of episodes truncated, and 5 of 9 capped episodes had
        # already passed, so "verified" and "complete" are not the same thing.
        usable = [e for e in group if not e.truncated and e.tokens > 0]
        chosen = sorted([e for e in usable if e.verified], key=lambda e: e.tokens)
        rejected = sorted([e for e in usable if not e.verified], key=lambda e: -e.tokens)
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
                        **({"provenance": [good.provenance, bad.provenance]} if good.provenance else {}),
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
    best_only: bool = True,
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

    rows = sft_rows(episodes, best_only=best_only)
    pairs, capped = preference_pairs(episodes, max_per_task=max_per_task)
    # Preference messages need the same schema block as SFT. Retain the exact
    # task/round tool set instead of silently dropping it when taking ["messages"].
    from hermes.format import to_messages_record
    from hermes.trajectory import AgentTrajectory

    context = _render_context(episodes)
    tools_by_task = {}
    paired_tasks = {(p["round_id"], p["task_id"]) for p in pairs}
    for episode in episodes:
        key = (episode.round_id, episode.task_id)
        if episode.usable and not episode.truncated and episode.tokens > 0 and key in paired_tasks:
            record = to_messages_record(AgentTrajectory.from_record(episode.trajectory), **context)
            tools = record.get("tools")
            if key in tools_by_task and tools_by_task[key] != tools:
                raise AggregateError(f"{episode.task_id}: preference episodes used different tool schemas")
            tools_by_task[key] = tools
    for pair in pairs:
        tools = tools_by_task.get((pair["round_id"], pair["task_id"]))
        if tools:
            pair["tools"] = tools
    summary.sft_rows = len(rows)
    summary.pairs = len(pairs)
    summary.capped = capped
    # Counted off the rows themselves rather than tracked through the renderer, so the number always
    # describes what was actually written.
    summary.truncated_skipped = sum(1 for e in episodes if e.verified and e.usable and e.truncated)
    # Verified attempts that lost to a cheaper one on the same task. Reported because a corpus of 150
    # rows built from 1,200 episodes has to say so -- otherwise the row count reads as the data being
    # thin rather than as rejection sampling having done its job.
    summary.not_best_skipped = (
        sum(1 for e in episodes if e.verified and e.usable and not e.truncated) - summary.sft_rows
    )
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
    replay: Any = None,
    source: str | None = None,
    round_ids: list[str] | None = None,
) -> list[Episode]:
    """Collect through configured committed authority, never a SETTLED JSON snapshot."""
    from hermes.round import SETTLED
    from validator.store import RoundStore

    if replay is None:
        store = store or RoundStore(rounds)
        if any(store.load(r).state == SETTLED for r in store.round_ids()):
            raise AggregateError("settled snapshots are not authority; configure admin.replay with committed sources")
        return []
    if not source or not round_ids:
        raise AggregateError("collection requires an explicit configured source and round IDs")
    for round_id in round_ids:
        replay.import_round(source, round_id)
    return replay.experiences()[0]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=Path("var/judge"))
    parser.add_argument("--out", type=Path, default=Path("var/datasets"))
    parser.add_argument("--max-pairs-per-task", type=int, default=MAX_PAIRS_PER_TASK)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--round", action="append", dest="round_ids")
    parser.add_argument("--mode", choices=("production", "fixture"))
    parser.add_argument("--namespace")
    args = parser.parse_args(argv)

    try:
        from admin.pipeline import Workspace
        from admin.replay import ReplayStore
        from validator.persistence import state_identity

        if args.replay_root is None:
            raise AggregateError("configure --replay-root; raw episode paths/SETTLED snapshots cannot authorize corpus")
        replay = ReplayStore(args.replay_root, mode=args.mode, namespace=args.namespace)
        collect(replay=replay, source=args.source, round_ids=args.round_ids)
        state_identity(args.out, mode=args.mode, namespace=args.namespace)
        record = replay.freeze(Workspace(args.out))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"validator.aggregate: {exc}", file=sys.stderr)
        return 2

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

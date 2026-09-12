"""Whether the model you just trained replaces the one you are serving.

    python -m hermes.promotion --incumbent m0.json --candidate m1.json
    -> promote:YES   success improved beyond the paired test, and nothing else regressed
    -> promote:NO    every reason, not the first

The decision this module exists to refuse is the cheap one: M1 scored higher than M0, ship it.
Almost every way that number moves has nothing to do with the model.

**The comparison has to be the same comparison.** `hermes.harness.comparable` already refuses
two runs whose suite or harness digest disagree, and that digest covers the system prompt, the
tool schemas, the executor, the observation limit and the container. It does not cover how the
model was *served* -- precision, device, engine, sampling parameters -- because those are not
properties of the harness. They decide outcomes anyway. Serving M1 at BF16 against an M0 run
recorded at NVFP4, or at a different temperature, produces a promotion that is a serving-config
change wearing a model's name. `Serving` carries them, and every field must be stated: two runs
that both left precision blank must not compare equal, which is what a defaulted field would do.

**A higher rate is not an improvement.** Nineteen tasks at ten repeats is 190 episodes, and the
run-to-run spread on that is wide enough to move the headline several points with identical
weights. So success is decided by a paired test over tasks -- which tasks changed direction,
not by how much the average moved -- and a result that cannot reach significance says so. Six
discordant tasks is the floor: with five, the two-sided sign test's smallest attainable p is
0.0625, so no split of them can be significant and reporting p = 0.0625 as "not significant"
hides that the run never could have said yes.

**Efficiency has to be able to veto.** The user of this model pays per successful task, not per
episode, so tokens and tool calls are amortised over successes rather than averaged over
episodes: a model that gives up early on the tasks it would fail looks cheap per episode and is
not cheaper to use. A regression beyond `MAX_EFFICIENCY_REGRESSION` refuses the promotion even
when success improved, which is the whole point of a guardrail.

**Conformance is not an efficiency term.** Hermes is upstream and fixed; the product is a
better model for it. But the wire format lives in prose inside the system prompt, so anything
appended after it can degrade conformance without touching code -- and degrading it is
*cheaper*, because a well-formed call and a planning block both cost tokens. An efficiency
score with no conformance term therefore pays for drifting off-protocol. `malformed_turns` gets
a hard bound rather than the efficiency tolerance.

## What this module will not do

It will not compare two runs whose per-task attempt counts differ, and it will not silently
drop a task that only one side ran. Both are the same failure in different clothes: a headline
whose denominator changed between the two numbers being compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from hermes.harness import RunManifest, comparable

YES = "promote:YES"
NO = "promote:NO"

# Two-sided significance for the paired task-level test.
ALPHA = 0.05

# Below this many tasks that changed direction, no split can reach ALPHA: the smallest attainable
# two-sided p at k discordant tasks is 2 * 0.5**k, which first clears 0.05 at k = 6. Reported as
# "underpowered" rather than as "not significant", because the two mean different things to
# whoever has to decide what to run next.
MIN_DISCORDANT_TASKS = 6

# How much worse an efficiency metric may get while success improves. A starting point, like
# `acceptance.MIN_TOKEN_REDUCTION`, and for the same reason: the run-to-run spread that would
# calibrate it is not known until enough paired runs exist. Erring loose here is deliberate --
# the gate that decides promotion is the success test, and an efficiency bound tight enough to
# fire on noise turns into the number people delete.
MAX_EFFICIENCY_REGRESSION = 0.15

# Conformance gets an absolute bound instead of a ratio: at a base rate near zero any ratio is
# either unbounded or meaningless, and one malformed turn in fifty is not 100% worse than one in
# a hundred in any sense that matters.
MAX_MALFORMED_INCREASE = 0.02

SERVING_FIELDS = ("precision", "device", "engine", "temperature", "top_p", "max_model_len")


class PromotionError(ValueError):
    """A promotion cannot be decided from what was supplied."""


@dataclass(frozen=True)
class Serving:
    """How a model was served. Not in `harness_digest`, and it decides outcomes.

    Every field is required. A defaulted one would make two runs that recorded nothing about
    precision compare as identically served, which is the exact shape of the bug this module is
    built to avoid: absence reading as agreement.
    """

    precision: str
    device: str
    engine: str
    temperature: float
    top_p: float
    max_model_len: int
    # Encrypts host-device traffic and inflates wall time. Recorded because latency is one of the
    # metrics being compared and this is a real, measured effect on the box this project runs on
    # -- not a hypothetical. `None` is unstated, which is refused rather than assumed False.
    confidential_computing: bool | None = None

    def unstated(self) -> list[str]:
        blank = [f for f in ("precision", "device", "engine") if not getattr(self, f).strip()]
        if not math.isfinite(self.temperature) or self.temperature < 0:
            blank.append("temperature")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            blank.append("top_p")
        if self.max_model_len <= 0:
            blank.append("max_model_len")
        if type(self.confidential_computing) is not bool:
            blank.append("confidential_computing")
        return blank

    def differences(self, other: Serving) -> list[str]:
        out = []
        for name in (*SERVING_FIELDS, "confidential_computing"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                out.append(f"{name}: {mine!r} vs {theirs!r}")
        return out

    def to_record(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (*SERVING_FIELDS, "confidential_computing")}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Serving:
        try:
            return cls(
                precision=str(record.get("precision") or ""),
                device=str(record.get("device") or ""),
                engine=str(record.get("engine") or ""),
                temperature=float(record.get("temperature", float("nan"))),
                top_p=float(record.get("top_p") or 0.0),
                max_model_len=int(record.get("max_model_len") or 0),
                confidential_computing=record.get("confidential_computing"),
            )
        except (TypeError, ValueError) as exc:
            raise PromotionError(f"serving record is malformed: {exc}") from exc


@dataclass(frozen=True)
class Episode:
    """One attempt at one task, at the grain the gate needs.

    `malformed_turns` and `max_steps_hit` are here and not in `harness.TaskResult` because they
    are the two failure modes that do not show up in any of the other numbers: a run can be
    faster, cheaper and more successful while emitting more calls Hermes cannot parse.
    """

    task_id: str
    success: bool
    tokens: int
    tool_calls: int
    wall_time_s: float
    malformed_turns: int = 0
    max_steps_hit: bool = False
    setup_failed: bool = False
    # Which grader scored this episode, and which wire dialect drove it. Both are stamped by the
    # runner and both are empty in logs written before those fields existed, which is treated as
    # unverifiable rather than as agreement -- the same reading `hermes.challenge` gives them.
    verify_digest: str = ""
    dialect: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Episode:
        # An episode log line nests its numbers under `metrics`; a hand-written run file may not.
        nested = record.get("metrics")
        metrics: dict[str, Any] = nested if isinstance(nested, dict) else record
        task_id = str(metrics.get("task_id") or "")
        if not task_id:
            raise PromotionError("an episode without a task_id cannot be paired against anything")
        return cls(
            task_id=task_id,
            success=bool(metrics.get("success")),
            tokens=int(metrics.get("tokens_used") or 0),
            tool_calls=int(metrics.get("tool_calls") or 0),
            wall_time_s=float(metrics.get("wall_time_s") or 0.0),
            malformed_turns=int(metrics.get("malformed_turns") or 0),
            max_steps_hit=bool(metrics.get("max_steps_hit")),
            setup_failed=bool(metrics.get("setup_failed") or record.get("setup_failed")),
            verify_digest=str(metrics.get("verify_digest") or ""),
            dialect=str(metrics.get("dialect") or ""),
        )


@dataclass(frozen=True)
class Run:
    """One model's benchmark run: what it scored, and how it was served."""

    model: str
    serving: Serving
    episodes: tuple[Episode, ...]
    manifest: RunManifest | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise PromotionError("a run without a model names no subject")
        if not self.episodes:
            raise PromotionError(f"{self.model} has no episodes; there is nothing to compare")

    @property
    def ran(self) -> tuple[Episode, ...]:
        """Episodes that reached the agent.

        Setup failures are infrastructure breakage, not model behaviour, and counting them as
        failures makes a broken container read as a worse model. Excluded from every rate here
        and reported separately, the same way `hermesbench.metrics.suite_metrics` does it.
        """
        return tuple(e for e in self.episodes if not e.setup_failed)

    @property
    def setup_failures(self) -> int:
        return len(self.episodes) - len(self.ran)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.ran if e.success)

    @property
    def success_rate(self) -> float:
        return self.successes / len(self.ran) if self.ran else 0.0

    @property
    def tasks(self) -> dict[str, tuple[Episode, ...]]:
        out: dict[str, list[Episode]] = {}
        for episode in self.ran:
            out.setdefault(episode.task_id, []).append(episode)
        return {task: tuple(episodes) for task, episodes in out.items()}

    def rate_on(self, task_id: str) -> float:
        episodes = self.tasks.get(task_id, ())
        return sum(1 for e in episodes if e.success) / len(episodes) if episodes else 0.0

    @property
    def tokens_per_success(self) -> float:
        """Everything spent, over what was achieved.

        Amortised rather than averaged over episodes on purpose. A model that gives up early on
        the tasks it would have failed shows a lower mean token count per episode and costs more
        per task actually completed, which is the number anyone using it pays.
        """
        return sum(e.tokens for e in self.ran) / self.successes if self.successes else math.inf

    @property
    def tool_calls_per_success(self) -> float:
        return sum(e.tool_calls for e in self.ran) / self.successes if self.successes else math.inf

    @property
    def median_wall_time_s(self) -> float:
        """Median, not mean: one 900-second timeout should not decide a latency comparison."""
        return median(e.wall_time_s for e in self.ran) if self.ran else 0.0

    @property
    def malformed_rate(self) -> float:
        """Episodes containing at least one turn Hermes could not parse."""
        return sum(1 for e in self.ran if e.malformed_turns > 0) / len(self.ran) if self.ran else 0.0

    @property
    def catastrophic_rate(self) -> float:
        """Episodes that ran out of budget without succeeding: no answer, full cost."""
        return sum(1 for e in self.ran if e.max_steps_hit and not e.success) / len(self.ran) if self.ran else 0.0


def binomial_sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test on discordant pairs.

    Exact rather than normal-approximated: nineteen tasks is not enough for the approximation,
    and this is the number the decision turns on. Ties carry no information about direction and
    are excluded, which is what makes the denominator `wins + losses`.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class PairedSuccess:
    """The task-level comparison the promotion turns on."""

    wins: int
    losses: int
    ties: int
    p_value: float
    incumbent_rate: float
    candidate_rate: float

    @property
    def discordant(self) -> int:
        return self.wins + self.losses

    @property
    def underpowered(self) -> bool:
        return self.discordant < MIN_DISCORDANT_TASKS

    @property
    def significant(self) -> bool:
        return not self.underpowered and self.p_value <= ALPHA and self.wins > self.losses

    def to_record(self) -> dict[str, Any]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "p_value": round(self.p_value, 5),
            "incumbent_rate": round(self.incumbent_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "underpowered": self.underpowered,
            "significant": self.significant,
        }


def paired_success(incumbent: Run, candidate: Run) -> PairedSuccess:
    """Compare per task, not in aggregate.

    An aggregate rate moves with which tasks happened to be in the run. Pairing by task removes
    that: every task is its own control, and what is counted is how many changed direction.
    """
    left, right = incumbent.tasks, candidate.tasks
    wins = losses = ties = 0
    for task in sorted(set(left) & set(right)):
        before = sum(1 for e in left[task] if e.success) / len(left[task])
        after = sum(1 for e in right[task] if e.success) / len(right[task])
        if after > before:
            wins += 1
        elif after < before:
            losses += 1
        else:
            ties += 1
    return PairedSuccess(
        wins=wins,
        losses=losses,
        ties=ties,
        p_value=binomial_sign_test(wins, losses),
        incumbent_rate=incumbent.success_rate,
        candidate_rate=candidate.success_rate,
    )


def check_serving(incumbent: Run, candidate: Run) -> list[str]:
    """The half of "identical conditions" that no digest in this repo covers."""
    issues = []
    for run in (incumbent, candidate):
        blank = run.serving.unstated()
        if blank:
            issues.append(
                f"{run.model} does not record {', '.join(blank)}. Unstated is refused rather than "
                "assumed: two runs that both left a field blank would otherwise compare as "
                "identically served."
            )
    if issues:
        return issues
    differences = incumbent.serving.differences(candidate.serving)
    if differences:
        return [
            "the two runs were not served the same way (" + "; ".join(differences) + "). A promotion "
            "decided across a serving change is a serving change wearing the model's name."
        ]
    return []


def check_evidence(incumbent: Run, candidate: Run) -> list[str]:
    """Same tasks, same number of attempts at each.

    A task only one side ran, or ran a different number of times, changes the denominator between
    the two numbers being compared. Dropping it quietly is how a partial rerun becomes a result.
    """
    issues = []
    before, after = incumbent.tasks, candidate.tasks
    only_incumbent = sorted(set(before) - set(after))
    only_candidate = sorted(set(after) - set(before))
    if only_incumbent or only_candidate:
        issues.append(
            f"the runs do not cover the same tasks: {incumbent.model} alone ran {only_incumbent[:5]}, "
            f"{candidate.model} alone ran {only_candidate[:5]}. Comparing what is left over changes "
            "the denominator between the two numbers."
        )
    uneven = [
        f"{task} ({len(before[task])} vs {len(after[task])})"
        for task in sorted(set(before) & set(after))
        if len(before[task]) != len(after[task])
    ]
    if uneven:
        issues.append(
            "the two runs made different numbers of attempts at " + ", ".join(uneven[:5]) + ". Per-task "
            "rates over different denominators are not paired observations."
        )
    for run in (incumbent, candidate):
        if run.setup_failures:
            issues.append(
                f"{run.model} had {run.setup_failures} setup failures. They are excluded from every rate "
                "here as infrastructure rather than behaviour, but they mean the two sides did not "
                "gather the same amount of evidence."
            )
    return issues


def check_graders(incumbent: Run, candidate: Run) -> list[str]:
    """Whether the two runs were scored by the same verifier, in the same dialect.

    This is not hypothetical. Two tasks in this repo -- `fix-failing-test` and
    `verify-speedup-claim` -- failed ten of ten attempts because their published verify scripts
    invoked a bare `python`, and were fixed in a later commit. A model benchmarked before the fix
    and one benchmarked after differ by 2 of 19 tasks with nothing to do with either model, which
    is exactly the size of margin this gate is asked to rule on.

    An unstamped digest is refused rather than matched. Empty on both sides would otherwise
    compare equal, which is the reading that makes the check unable to fire on old logs -- the
    ones most likely to predate a grader fix.
    """
    issues = []
    for run in (incumbent, candidate):
        unstamped = sorted({e.task_id for e in run.ran if not e.verify_digest})
        if unstamped:
            issues.append(
                f"{run.model} has episodes with no verify_digest ({unstamped[:5]}), so which grader "
                "scored them is unknown. Two runs that both recorded nothing would compare as "
                "having used the same one."
            )
    if issues:
        return issues

    before, after = incumbent.tasks, candidate.tasks
    changed = [
        task
        for task in sorted(set(before) & set(after))
        if {e.verify_digest for e in before[task]} != {e.verify_digest for e in after[task]}
    ]
    if changed:
        issues.append(
            f"the verify script changed between the runs for {changed[:5]}. A grader fix scores as a "
            "model improvement, and the two tasks this actually happened to were worth 2 of 19."
        )
    dialects = {e.dialect for e in (*incumbent.ran, *candidate.ran)}
    if len(dialects) > 1:
        issues.append(
            f"the runs did not use one wire dialect: {sorted(dialects)}. The dialect decides how "
            "tool calls are spelled, so it decides how many of them parse."
        )
    return issues


def _regression(before: float, after: float) -> float:
    """How much worse `after` is, as a fraction of `before`. Negative is an improvement."""
    if before <= 0 or math.isinf(before):
        return 0.0 if after <= before else math.inf
    return (after - before) / before


@dataclass(frozen=True)
class Efficiency:
    """The secondary metrics, each able to veto."""

    name: str
    before: float
    after: float
    limit: float

    @property
    def regression(self) -> float:
        return _regression(self.before, self.after)

    @property
    def regressed(self) -> bool:
        return self.regression > self.limit

    def to_record(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "incumbent": None if math.isinf(self.before) else round(self.before, 4),
            "candidate": None if math.isinf(self.after) else round(self.after, 4),
            "regression": None if math.isinf(self.regression) else round(self.regression, 4),
            "limit": self.limit,
            "regressed": self.regressed,
        }


def efficiency_terms(incumbent: Run, candidate: Run) -> list[Efficiency]:
    return [
        Efficiency(
            "tokens_per_success", incumbent.tokens_per_success, candidate.tokens_per_success, MAX_EFFICIENCY_REGRESSION
        ),
        Efficiency(
            "tool_calls_per_success",
            incumbent.tool_calls_per_success,
            candidate.tool_calls_per_success,
            MAX_EFFICIENCY_REGRESSION,
        ),
        Efficiency(
            "median_wall_time_s", incumbent.median_wall_time_s, candidate.median_wall_time_s, MAX_EFFICIENCY_REGRESSION
        ),
        Efficiency(
            "catastrophic_rate", incumbent.catastrophic_rate, candidate.catastrophic_rate, MAX_EFFICIENCY_REGRESSION
        ),
    ]


def check_conformance(incumbent: Run, candidate: Run) -> list[str]:
    """The one term an efficiency score would otherwise pay to lose.

    A well-formed `<tool_call>` and a reasoning block both cost tokens, so a model that drifts
    off-protocol scores *better* on every other metric here. Bounded absolutely rather than
    proportionally: at a base rate near zero a ratio is either unbounded or meaningless.
    """
    increase = candidate.malformed_rate - incumbent.malformed_rate
    if increase > MAX_MALFORMED_INCREASE:
        return [
            f"malformed turns rose from {incumbent.malformed_rate:.1%} to {candidate.malformed_rate:.1%} of "
            f"episodes, over the {MAX_MALFORMED_INCREASE:.0%} bound. Hermes is upstream and fixed; a model "
            "that emits calls it cannot parse is cheaper on every other metric here, which is why this one "
            "is not folded into the efficiency tolerance."
        ]
    return []


@dataclass(frozen=True)
class Decision:
    """Promote or not, with every reason."""

    incumbent: str
    candidate: str
    success: PairedSuccess
    efficiency: tuple[Efficiency, ...]
    issues: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def promote(self) -> bool:
        return not self.issues

    def to_record(self) -> dict[str, Any]:
        return {
            "verdict": YES if self.promote else NO,
            "comparison_kind": "model-only",
            "authorizes_activation": False,
            "incumbent": self.incumbent,
            "candidate": self.candidate,
            "success": self.success.to_record(),
            "efficiency": [e.to_record() for e in self.efficiency],
            "issues": list(self.issues),
            "notes": list(self.notes),
        }


def decide(incumbent: Run, candidate: Run) -> Decision:
    """Whether the candidate replaces the incumbent. Every failing reason, not the first.

    Order matters only in that comparability is established before anything is measured: a
    number computed across two different harnesses is not a wrong number, it is a number about
    a different question, and reporting it beside the others invites reading it as evidence.
    """
    issues: list[str] = []
    notes: list[str] = []

    if incumbent.manifest is not None and candidate.manifest is not None:
        ok, reason = comparable(incumbent.manifest, candidate.manifest)
        if not ok:
            issues.append(reason)
    else:
        notes.append(
            "no run manifests supplied, so the suite and harness digests were not checked. "
            "`hermes.harness.comparable` is what refuses a harness change read as a model change."
        )

    issues.extend(check_serving(incumbent, candidate))
    issues.extend(check_evidence(incumbent, candidate))
    issues.extend(check_graders(incumbent, candidate))

    success = paired_success(incumbent, candidate)
    efficiency = tuple(efficiency_terms(incumbent, candidate))

    if success.underpowered:
        issues.append(
            f"only {success.discordant} tasks changed direction, and at fewer than "
            f"{MIN_DISCORDANT_TASKS} no split can reach p <= {ALPHA}. This is underpowered rather than "
            "negative: more repeats, or more tasks, would let the run answer."
        )
    elif not success.significant:
        issues.append(
            f"success did not improve beyond the paired test: {success.wins} tasks better, "
            f"{success.losses} worse, {success.ties} unchanged, p = {success.p_value:.3f}."
        )

    if success.candidate_rate <= success.incumbent_rate:
        issues.append(f"overall success did not improve: {success.incumbent_rate:.1%} -> {success.candidate_rate:.1%}.")

    for term in efficiency:
        if term.regressed:
            issues.append(
                f"{term.name} regressed by {term.regression:.1%} ({term.before:.4g} -> {term.after:.4g}), "
                f"over the {term.limit:.0%} guardrail."
            )

    issues.extend(check_conformance(incumbent, candidate))

    return Decision(
        incumbent=incumbent.model,
        candidate=candidate.model,
        success=success,
        efficiency=efficiency,
        issues=tuple(issues),
        notes=tuple(notes),
    )


def load_run(path: Path) -> Run:
    """A run file: the model, how it was served, and its episodes.

    Accepts the episode log `hermesbench.sink` writes, wrapped with the serving record the log
    cannot know about -- nothing in the runner is told what device it is on.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{path} is not a readable run file: {exc}") from exc
    if not isinstance(record, dict):
        raise PromotionError(f"{path} does not hold a run record")
    episodes = record.get("episodes")
    if episodes is None and record.get("episodes_path"):
        # The usual case: the runner wrote a JSONL log and the run file describes it. The path is
        # resolved against the run file rather than the working directory, so a run file stays
        # valid wherever it is read from.
        episodes_path = path.parent / str(record["episodes_path"])
        if record.get("episodes_sha256"):
            try:
                with episodes_path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != record["episodes_sha256"]:
                        raise PromotionError("episode log changed after evaluation")
            except OSError as exc:
                raise PromotionError(f"cannot read episode log: {exc}") from exc
        episodes = _read_episode_log(episodes_path)
    if not isinstance(episodes, list):
        raise PromotionError(f"{path} carries neither an episodes list nor an episodes_path naming a run log")
    manifest = None
    if record.get("manifest_path"):
        manifest_path = path.parent / str(record["manifest_path"])
        try:
            with manifest_path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record.get("manifest_sha256"):
                    raise PromotionError("run manifest changed after evaluation")
            manifest = RunManifest.from_record(json.loads(manifest_path.read_text()))
            if manifest.model != record.get("model"):
                raise PromotionError("run record and manifest name different models")
        except (OSError, ValueError) as exc:
            raise PromotionError(f"cannot load run manifest: {exc}") from exc
    return Run(
        model=str(record.get("model") or ""),
        serving=Serving.from_record(record.get("serving") or {}),
        episodes=tuple(Episode.from_record(e) for e in episodes),
        manifest=manifest,
    )


def _read_episode_log(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL `hermesbench.sink` writes.

    A truncated final line is the episode the run died inside, and the sink documents it as such.
    It is dropped with the rest of the log kept -- but a log that ends mid-episode means the run
    did not finish, and a comparison against a partial run is a comparison against a smaller
    denominator, so `check_evidence` still has to see the shortfall. It does: the missing
    attempts show up as uneven per-task counts.
    """
    if not path.is_file():
        raise PromotionError(f"episode log not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise PromotionError(f"{path} holds no readable episodes")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incumbent", required=True, type=Path, help="the run of the model being served (M0)")
    parser.add_argument("--candidate", required=True, type=Path, help="the run of the model being proposed (M1)")
    parser.add_argument("--json", action="store_true", help="emit the decision as a record")
    args = parser.parse_args(argv)

    try:
        decision = decide(load_run(args.incumbent), load_run(args.candidate))
    except PromotionError as exc:
        print(f"{NO} {exc}")
        return 1

    if args.json:
        print(json.dumps(decision.to_record(), indent=2))
        return 0 if decision.promote else 1

    success = decision.success
    print(f"{YES if decision.promote else NO} {decision.candidate} vs {decision.incumbent}")
    print(
        f"  success {success.incumbent_rate:.1%} -> {success.candidate_rate:.1%}  "
        f"({success.wins} tasks better, {success.losses} worse, {success.ties} unchanged, p = {success.p_value:.3f})"
    )
    for term in decision.efficiency:
        print(f"  {term.name}: {term.before:.4g} -> {term.after:.4g}")
    for note in decision.notes:
        print(f"  note: {note}")
    for issue in decision.issues:
        print(f"  - {issue}")
    return 0 if decision.promote else 1


__all__ = [
    "ALPHA",
    "MAX_EFFICIENCY_REGRESSION",
    "MAX_MALFORMED_INCREASE",
    "MIN_DISCORDANT_TASKS",
    "NO",
    "SERVING_FIELDS",
    "YES",
    "Decision",
    "Efficiency",
    "Episode",
    "PairedSuccess",
    "PromotionError",
    "Run",
    "Serving",
    "binomial_sign_test",
    "check_conformance",
    "check_evidence",
    "check_graders",
    "check_serving",
    "decide",
    "efficiency_terms",
    "load_run",
    "main",
    "paired_success",
]


if __name__ == "__main__":
    raise SystemExit(main())

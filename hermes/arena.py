"""The arena: one task, every teacher, a verified winner, and rows you can trace back.

This is the loop the whole repository was assembled around. A seeded round names the tasks
a miner owes; each task is run once per teacher under one pinned harness; the verifier
decides; the tournament picks a winner; and the artifacts come out as SFT rows, DPO pairs
and router evidence with the provenance still attached.

Four things it refuses to blur, each because blurring it silently corrupts the corpus:

**A teacher that errored is not a teacher that failed.** An API timeout, a rate limit, a
502 -- none of those are evidence about the model, and scoring them as losses would build
a capability matrix that measures uptime. Errored runs are recorded and excluded, the way
`setup_failed` is excluded from every behavioural rate in `hermesbench.metrics`.

**A field of one is not a tournament.** If only one teacher produced a candidate, there is
a trajectory but no comparison: no DPO pair means anything, and "won" is not a fact about
it. Those tasks are kept as `incomparable` rather than reported as unanimous wins.

**One harness for the whole batch.** `Tournament` already refuses candidates that disagree
on the harness digest; the arena computes it once and stamps every candidate, so the
invariant is satisfied by construction rather than by everyone remembering.

**The manifest is the thing that gets attested, and it is built from digests only.** The
attested step is *verification*, never generation -- a TDX worker has no egress and could
not call a teacher if it wanted to, and attesting generation would prove a model emitted
tokens, which is not the claim the dataset makes. What the dataset claims is that these
trajectories pass these checks, and that is exactly what a sealed verifier can establish.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.harness import digest_mapping, digest_text
from hermes.selection import SelectionPolicy, select
from hermes.teachers import Teacher
from hermes.tournament import CandidateRun, Tournament, TournamentError, Verdict, build_artifacts
from hermesbench.tasks import Task

SCHEMA_VERSION = 1

# Why a teacher produced no candidate for a task.
ERROR_PROVIDER = "provider_error"
ERROR_PERMANENT = "permanent_error"
ERROR_SETUP = "setup_failed"
ERROR_BUDGET = "budget_exhausted"


class ArenaError(ValueError):
    """A batch cannot be run or its artifacts cannot be trusted."""


# Failures that will not succeed on retry. Retrying them wastes calls and, worse, dresses a
# configuration mistake as flakiness: a wrong key retried three times with backoff looks
# exactly like a busy gateway, and the operator tunes the backoff instead of fixing the key.
# This was learned the hard way -- a 401 from a mis-pointed base_url was diagnosed as rate
# limiting twice before anyone printed the error.
_PERMANENT = ("authenticationerror", "permissiondenieddeniederror", "permissiondeniederror", "notfounderror")
_PERMANENT_TEXT = ("invalid_api_key", "incorrect api key", "invalid_request_error", "model_not_found")


def is_permanent(exc: BaseException) -> bool:
    """Whether retrying this failure could ever help."""
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _PERMANENT):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_TEXT)


def trajectory_digest(trajectory: Any) -> str:
    """Content address for one trajectory.

    Over the record rather than the object, so the digest a miner publishes and the digest
    a reviewer recomputes come from the same bytes.
    """
    return digest_mapping(trajectory.to_record())


@dataclass(frozen=True)
class TeacherRun:
    """One teacher's attempt at one task, or the reason there is not one."""

    teacher_id: str
    task_id: str
    candidate: CandidateRun | None = None
    trajectory: Any = None
    error: str = ""
    detail: str = ""

    @property
    def ran(self) -> bool:
        return self.candidate is not None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"teacher_id": self.teacher_id, "task_id": self.task_id, "ran": self.ran}
        if self.candidate is not None:
            record["candidate"] = self.candidate.to_record()
            record["trajectory_sha256"] = self.candidate.trajectory_sha256
        if self.error:
            record["error"] = self.error
            record["detail"] = self.detail[:400]
        return record


@dataclass(frozen=True)
class TaskOutcome:
    """Everything one task produced across the field."""

    task_id: str
    runs: tuple[TeacherRun, ...]
    tournament: Tournament | None = None
    artifacts: Any = None
    decided_by_evidence: bool = False

    @property
    def comparable(self) -> bool:
        """Whether there were at least two candidates to compare.

        A field of one produces a trajectory and no comparison. Calling that a win would
        put an unearned row in the capability matrix and a meaningless one in DPO.
        """
        return sum(1 for r in self.runs if r.ran) >= 2

    @property
    def errored(self) -> tuple[TeacherRun, ...]:
        return tuple(r for r in self.runs if not r.ran)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "task_id": self.task_id,
            "comparable": self.comparable,
            "runs": [r.to_record() for r in self.runs],
            "errored": [r.teacher_id for r in self.errored],
        }
        if self.artifacts is not None:
            record["artifacts"] = self.artifacts.to_record()
            record["decided_by_evidence"] = self.decided_by_evidence
        return record


@dataclass
class ArenaBatch:
    """One miner's work for one round: what ran, what won, and what it can be trained on."""

    round_id: str
    miner_id: str
    harness_digest: str
    teachers: tuple[Teacher, ...]
    outcomes: list[TaskOutcome] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def comparable(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if o.comparable]

    @property
    def incomparable(self) -> list[TaskOutcome]:
        """Tasks where fewer than two teachers produced a candidate.

        Kept and reported rather than dropped: a batch where half the tasks are here is
        telling you a provider was down, and silently shrinking the batch would present
        that as a smaller round instead.
        """
        return [o for o in self.outcomes if not o.comparable]

    def sft_rows(self) -> list[dict[str, Any]]:
        """Winning trajectories whose teacher may be trained on.

        Rights are checked at export rather than at generation on purpose. A rights-denied
        teacher's run is still evidence about the task, and the capability matrix wants it;
        it simply never becomes a trained token. `build_artifacts` already enforces this,
        and the winner's sha being absent is how it says so.
        """
        rows = []
        for outcome in self.comparable:
            artifacts = outcome.artifacts
            if artifacts is None or artifacts.sft_trajectory_sha256 is None or artifacts.winner is None:
                continue
            run = next((r for r in outcome.runs if r.teacher_id == artifacts.winner.model and r.ran), None)
            if run is None or run.trajectory is None:
                continue
            rows.append(
                {
                    "task_id": outcome.task_id,
                    "teacher": artifacts.winner.model,
                    "trajectory_sha256": artifacts.sft_trajectory_sha256,
                    "trajectory": run.trajectory.to_record(),
                    "win_reasons": list(artifacts.win_reasons),
                    "decided_by_evidence": outcome.decided_by_evidence,
                }
            )
        return rows

    def dpo_rows(self) -> list[dict[str, Any]]:
        rows = []
        for outcome in self.comparable:
            if outcome.artifacts is None:
                continue
            for pair in outcome.artifacts.dpo_pairs:
                rows.append({"task_id": outcome.task_id, **pair.to_record()})
        return rows

    def router_rows(self) -> list[dict[str, Any]]:
        """Per-teacher outcome vectors, including from teachers we may not train on.

        Knowing that a model solved a task is a fact about the task. Withholding it from
        the router because the trajectory is unusable would throw away the measurement to
        protect a rule about the text.
        """
        return [o.artifacts.router_example for o in self.comparable if o.artifacts is not None]

    def withheld_for_rights(self) -> list[str]:
        seen: list[str] = []
        for outcome in self.comparable:
            if outcome.artifacts is not None:
                seen.extend(outcome.artifacts.withheld_for_rights)
        return sorted(set(seen))

    def check_task_ids(self) -> None:
        """Refuse duplicate task ids before digesting.

        `manifest()` sorts on task_id alone, and Python's sort is stable, so two outcomes
        sharing an id keep insertion order -- which makes the digest depend on the order
        outcomes were appended rather than on their content.
        """
        ids = [o.task_id for o in self.outcomes]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ArenaError(f"duplicate task ids {duplicates}; the manifest digest would depend on run order")

    def manifest(self, *, export_digests: dict[str, str] | None = None) -> dict[str, Any]:
        """What a receipt binds: digests, never content.

        Deliberately small and content-free. The attested step runs sealed with no egress,
        so whatever it is handed has to fit in the request -- and it does not need the
        trajectories themselves, only their addresses, because the claim being attested is
        "these digests were checked and these verdicts came out".
        """
        entries = []
        for outcome in self.outcomes:
            entries.append(
                {
                    "task_id": outcome.task_id,
                    "comparable": outcome.comparable,
                    "runs": sorted(
                        (
                            {
                                "teacher_id": r.teacher_id,
                                "trajectory_sha256": r.candidate.trajectory_sha256 if r.candidate else "",
                                "passed": bool(r.candidate and r.candidate.verdict.passed),
                                "disqualified": bool(r.candidate and r.candidate.disqualified),
                            }
                            for r in outcome.runs
                            if r.ran
                        ),
                        key=lambda e: e["teacher_id"],
                    ),
                    "winner": (
                        outcome.artifacts.winner.model
                        if outcome.artifacts is not None and outcome.artifacts.winner is not None
                        else None
                    ),
                }
            )
        self.check_task_ids()
        body = {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "harness_digest": self.harness_digest,
            # Pinned identity, not just a name. Two batches from the same teacher id served
            # from different weights are different batches, and a name-only list calls them
            # the same one.
            "teachers": sorted(
                (
                    {
                        "teacher_id": t.teacher_id,
                        "pin": t.pin,
                        "weights_revision": t.weights_revision,
                        "training_rights": t.training_rights,
                    }
                    for t in self.teachers
                ),
                key=lambda e: e["teacher_id"],
            ),
            # Folded in so the attested manifest covers the published rows. Outside it, the
            # digests are a claim the sealed check never saw.
            "exports": dict(sorted((export_digests or {}).items())),
            "tasks": sorted(entries, key=lambda e: e["task_id"]),
        }
        return {**body, "manifest_digest": digest_mapping(body)}

    def to_record(self) -> dict[str, Any]:
        manifest = self.manifest()
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "harness_digest": self.harness_digest,
            "teachers": [t.to_record() for t in self.teachers],
            "manifest_digest": manifest["manifest_digest"],
            "tasks": len(self.outcomes),
            "comparable": len(self.comparable),
            "incomparable": [o.task_id for o in self.incomparable],
            "withheld_for_rights": self.withheld_for_rights(),
            "sft_rows": len(self.sft_rows()),
            "dpo_rows": len(self.dpo_rows()),
            "outcomes": [o.to_record() for o in self.outcomes],
        }


# Runs one task against one teacher and returns (candidate, trajectory) or raises.
RunOne = Callable[[Task, Teacher], tuple[CandidateRun, Any]]


def run_task(
    task: Task,
    teachers: tuple[Teacher, ...],
    *,
    run_one: RunOne,
    harness_digest: str,
    policy: SelectionPolicy,
    primary_evidence: str = "",
    prefer_on_efficiency: bool = False,
    attempts: int = 3,
    backoff_s: float = 5.0,
    sleep: Callable[[float], None] | None = None,
) -> TaskOutcome:
    """Run one task across the field and build its artifacts.

    A teacher that raises is recorded as an errored run rather than a failing candidate.
    The distinction is the whole reason this function catches at all: an exception here is
    almost always the provider, and a provider outage that reads as a model failure poisons
    the capability matrix for as long as the record survives.
    """
    runs: list[TeacherRun] = []
    for teacher in teachers:
        # Retried with backoff, because a transient gateway error costs the whole task's
        # comparison and not just one teacher's run: with one side missing there are fewer
        # than two candidates, so the task is incomparable and produces no rows at all.
        #
        # The backoff is the part that matters. Observed live: a teacher failed three
        # consecutive tasks and then ran each of them cleanly in isolation, which is rate
        # limiting rather than breakage -- and three immediate retries against a rate
        # limiter are one retry that took slightly longer.
        wait = sleep if sleep is not None else time.sleep
        candidate = trajectory = None
        failures: list[str] = []
        permanent = False
        for attempt in range(max(1, attempts)):
            if attempt:
                wait(backoff_s * (2 ** (attempt - 1)))
            try:
                candidate, trajectory = run_one(task, teacher)
                break
            except Exception as exc:  # noqa: BLE001 -- any provider failure, deliberately
                failures.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                candidate = trajectory = None
                if is_permanent(exc):
                    permanent = True
                    break
        if candidate is None:
            runs.append(
                TeacherRun(
                    teacher_id=teacher.teacher_id,
                    task_id=task.task_id,
                    error=ERROR_PERMANENT if permanent else ERROR_PROVIDER,
                    detail=" | ".join(failures),
                )
            )
            continue
        # The candidate must address the trajectory that came back with it. A runner that
        # returns a digest of some other trajectory would put a row in the corpus under an
        # address that resolves to different work, and every downstream check would agree
        # with itself.
        expected = trajectory_digest(trajectory)
        if candidate.trajectory_sha256 and candidate.trajectory_sha256 != expected:
            runs.append(
                TeacherRun(
                    teacher_id=teacher.teacher_id,
                    task_id=task.task_id,
                    error=ERROR_PROVIDER,
                    detail=f"candidate cites {candidate.trajectory_sha256} but the trajectory digests to {expected}",
                )
            )
            continue
        from dataclasses import replace as _replace

        stamped = _stamp(
            _replace(candidate, trajectory_sha256=expected), harness_digest=harness_digest, teacher=teacher
        )
        runs.append(
            TeacherRun(teacher_id=teacher.teacher_id, task_id=task.task_id, candidate=stamped, trajectory=trajectory)
        )

    outcome = TaskOutcome(task_id=task.task_id, runs=tuple(runs))
    if not outcome.comparable:
        return outcome

    candidates = tuple(r.candidate for r in runs if r.candidate is not None)
    tournament = Tournament(task=_spec_for(task), candidates=candidates, primary_evidence=primary_evidence)
    selection = select(list(tournament.passing), policy=policy)
    artifacts = build_artifacts(
        tournament,
        prefer_on_efficiency=prefer_on_efficiency,
        winner=selection.winner,
        win_reasons=selection.policy_dimensions,
    )
    return TaskOutcome(
        task_id=task.task_id,
        runs=tuple(runs),
        tournament=tournament,
        artifacts=artifacts,
        decided_by_evidence=selection.decided_by_evidence,
    )


def _stamp(candidate: CandidateRun, *, harness_digest: str, teacher: Teacher) -> CandidateRun:
    """Bind the batch's harness and the teacher's rights onto the candidate.

    Set here rather than trusted from the caller so the fair-fight invariant holds by
    construction: `Tournament` refuses a field whose members disagree on the harness, and
    the only way to satisfy that reliably is for one place to stamp them all.
    """
    from dataclasses import replace

    return replace(
        candidate,
        model=teacher.teacher_id,
        harness_digest=harness_digest,
        training_rights=teacher.training_rights,
    )


def _spec_for(task: Task) -> Any:
    """A minimal TaskSpec for the tournament, derived from the bench task."""
    from hermes.router.spec import TaskSpec

    return TaskSpec(
        task_id=task.task_id,
        prompt=task.prompt,
        domain=("swe",),
        action=("implement",),
        # Asserted from the task rather than defaulted. `verification` in particular was
        # claiming `judge_only` on a suite whose verdicts come from a verifier's exit
        # status, which is the opposite of what happens.
        horizon="long" if task.is_long_horizon or task.max_steps > 20 else "medium",
        verification="unit_tests",
        metadata={"category": task.category},
    )


def verdict_from(result: Any) -> Verdict:
    """Turn an `EpisodeResult` into a tournament verdict.

    Deterministic, because it comes from the task's own `verify` exit status rather than
    from a judge. `evidence` stays empty unless the task published a measured quantity --
    inventing one would give the evidence tie-break something to prefer that nobody
    measured.
    """
    return Verdict(
        passed=result.metrics.success,
        verifier="hermesbench",
        deterministic=True,
        detail="" if result.verification.passed else (result.verification.stderr or "")[:200],
    )


def candidate_from(result: Any, *, teacher: Teacher, harness_digest: str) -> CandidateRun:
    """Build a candidate from a finished episode."""
    metrics = result.metrics
    return CandidateRun(
        model=teacher.teacher_id,
        trajectory_sha256=trajectory_digest(result.trajectory),
        verdict=verdict_from(result),
        harness_digest=harness_digest,
        tool_calls=metrics.tool_calls,
        invalid_tool_calls=metrics.failed_calls,
        recovered=metrics.recovered,
        hit_failure=metrics.hit_failure,
        self_checked=metrics.self_checked,
        tokens=metrics.tokens_used,
        wall_time_s=metrics.wall_time_s,
        cost=metrics.cost,
        training_rights=teacher.training_rights,
        disqualified=result.integrity.disqualified,
        # Only disqualifying signals. A warning listed here reads as a disqualification
        # reason on a run that was not disqualified, which is the opposite of what the
        # severity split is for.
        disqualification_reason="; ".join(s.code for s in result.integrity.signals if s.disqualifying),
        metadata={"integrity_warnings": [s.code for s in result.integrity.signals if not s.disqualifying]},
    )


def write_exports(batch: ArenaBatch, out_dir: Path) -> dict[str, int]:
    """Write the three views a batch produces, and report the row counts.

    Three files rather than one, because they are consumed by different trainers and a
    single mixed file would need a discriminator column that every reader has to remember
    to filter on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in (
        ("sft", batch.sft_rows()),
        ("dpo", batch.dpo_rows()),
        ("router", batch.router_rows()),
    ):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        written[name] = len(rows)
    manifest = batch.manifest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["manifest_digest"] = manifest["manifest_digest"]  # type: ignore[assignment]
    return written


def export_digests(out_dir: Path) -> dict[str, str]:
    """Digest each exported file, for the submission record."""
    digests = {}
    for name in ("sft", "dpo", "router"):
        path = out_dir / f"{name}.jsonl"
        if path.is_file():
            digests[name] = digest_text(path.read_text(encoding="utf-8"))
    return digests


def check_batch(batch: ArenaBatch) -> list[str]:
    """Problems with a finished batch, worst first. Empty means it is fit to submit."""
    problems: list[str] = []
    if not batch.outcomes:
        problems.append("the batch ran no tasks")
        return problems
    if not batch.harness_digest:
        problems.append("no harness digest; the runs cannot be compared to any other batch")
    incomparable = batch.incomparable
    if len(incomparable) == len(batch.outcomes):
        problems.append(
            "no task had two teachers produce a candidate; the batch contains trajectories but no "
            "comparisons, so nothing in it is a tournament result"
        )
    elif len(incomparable) > len(batch.outcomes) // 2:
        problems.append(
            f"{len(incomparable)} of {len(batch.outcomes)} tasks were incomparable; a provider was "
            "probably down, and the surviving rows are a biased sample of the round"
        )
    if not batch.sft_rows():
        problems.append("no winning trajectory may be trained on; the batch yields router evidence only")
    return problems


def tournament_or_none(outcome: TaskOutcome) -> Tournament | None:
    """The tournament, if one could be built. Never raises."""
    try:
        return outcome.tournament
    except TournamentError:
        return None

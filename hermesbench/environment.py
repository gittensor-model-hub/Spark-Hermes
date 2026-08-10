"""Environments: the world an agent acts in, as a first-class thing.

    Environment = task + agent harness + verifier + state

Until now the runner hard-coded all four against a filesystem workspace: setup wrote
files, tools read and wrote files, the verifier ran a shell command in that directory.
That works for SWE and CUDA and cannot express a browser session, an emulator, or
hardware in the loop -- none of which have a `Path` to hand the executor. Adding one of
those meant rewriting the loop rather than implementing an interface.

Two separations are load-bearing:

**Environment is not dataset.** An environment is a capability test; the dataset is the
tasks inside it. `WorkspaceEnvironment` is one environment that a thousand generated
tasks can run in, which is what lets tasks be procedurally generated without the harness
knowing anything about how.

**`verify()` is not `reward()`.** Verification is the binary the pipeline trusts: did the
task get done, per something other than the agent. Reward is a shaped scalar for training,
and shaping is exactly where gaming enters -- an agent optimising a reward will find the
cheapest component to move. So reward is computed *from* the verified outcome and reports
its components, and a run that failed verification cannot earn a positive one however good
its efficiency looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hermes.trajectory import AgentTrajectory, Step
from hermesbench.integrity import (
    INTEGRITY_PARTIAL,
    WARNING,
    IntegrityReport,
    IntegritySignal,
    digest_paths,
)
from hermesbench.tasks import Task
from hermesbench.verify import VerificationResult, resolve_env, run_checkpoints, setup_task, verify_hidden, verify_task


@dataclass(frozen=True)
class Observation:
    """What the environment reports back after an action.

    `ok=False` marks a failure the agent has to react to; it is not an exception, because
    a tool that failed is information the episode should continue from.
    """

    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvState:
    """Per-task execution context.

    Explicit rather than implicit in a directory, so it can be snapshotted, carried
    across a handoff, and included in a proof. A state that only exists as "whatever is
    on disk right now" cannot be compared against what a miner claimed it was.
    """

    task_id: str
    step_count: int = 0
    workspace: Path | None = None
    protected_digests: dict[str, str | None] = field(default_factory=dict)
    checkpoint_timeline: list[dict[str, bool]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "workspace": str(self.workspace) if self.workspace else None,
            "checkpoint_samples": len(self.checkpoint_timeline),
        }


@dataclass(frozen=True)
class Reward:
    """A shaped training signal, derived from a verified outcome.

    `total` is clamped to 0 when the task was not verifiably completed. Efficiency and
    cleanliness are worth rewarding *given* success; on their own they reward an agent
    that did nothing quickly, which is the cheapest possible component to move.
    """

    verified: bool
    total: float
    components: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "total": round(self.total, 4),
            "components": {k: round(v, 4) for k, v in sorted(self.components.items())},
        }


@dataclass(frozen=True)
class EnvOutcome:
    """The composed verdict: published checks, withheld checks, and integrity together."""

    public: VerificationResult
    hidden: VerificationResult | None
    integrity_disqualified: bool
    verified: bool
    # The full report, so a caller can tell "checked and clean" from "not checked".
    # `integrity_disqualified` alone collapses those two into the same False.
    integrity: IntegrityReport | None = None

    @property
    def overfit(self) -> bool:
        """Passed what is published, failed what is withheld."""
        return self.public.passed and self.hidden is not None and not self.hidden.passed

    @property
    def fully_checked(self) -> bool:
        """Whether every integrity detector actually ran."""
        return self.integrity is not None and not any(s.code == INTEGRITY_PARTIAL for s in self.integrity.signals)

    def to_record(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "public_passed": self.public.passed,
            "hidden_passed": None if self.hidden is None else self.hidden.passed,
            "integrity_disqualified": self.integrity_disqualified,
            "fully_checked": self.fully_checked,
            "overfit": self.overfit,
        }


class Environment(Protocol):
    """The world an agent acts in.

    Deliberately close to the `reset / step / verify` shape agent-RL frameworks use, so
    an environment written here is portable and a rollout collected elsewhere is
    interpretable. `reward` is separate from `verify` for the reason above.
    """

    def reset(self, task: Task) -> EnvState:
        """Prepare the world for this task and return its starting state."""
        ...

    def step(self, action: Step) -> Observation:
        """Perform one agent action and report what actually happened."""
        ...

    def verify(self) -> VerificationResult:
        """Did the task get done? Decided by the environment, not the agent."""
        ...

    def outcome(self, trajectory: AgentTrajectory | None = None) -> EnvOutcome:
        """Published AND withheld AND not-cheated -- the verdict the pipeline trusts.

        Pass the episode's trajectory: without it, two of the three integrity detectors
        have nothing to read and return clean without looking.
        """
        ...

    def reward(self, *, verified: bool, tool_calls: int, failed_calls: int) -> Reward:
        """Shaped signal for training, derived from the verified outcome."""
        ...


class WorkspaceEnvironment:
    """A filesystem workspace: the environment SWE, CUDA and terminal tasks live in.

    Wraps what the runner used to do inline. The behaviour is unchanged; what changes is
    that it is now one implementation of an interface rather than the only thing possible.
    """

    name = "workspace"

    def __init__(self, executor: Any, root: Path) -> None:
        self.executor = executor
        self.root = root
        self.task: Task | None = None
        self.state: EnvState | None = None
        self._env: dict[str, str] | None = None
        self._setup_result: VerificationResult | None = None
        self._final_sampled = False
        self._protected_after: dict[str, str | None] | None = None

    def reset(self, task: Task) -> EnvState:
        """Build a clean world for this task.

        The workspace is removed first. Reusing one environment across a suite -- the
        stated use case -- otherwise inherits the previous episode's files, and the second
        run of a task starts already solved: verified success at maximum reward for an
        agent that did nothing.

        Nothing is adopted onto `self` until the new state exists. Assigning the task
        first meant that a reset which raised part-way left the object holding the new
        task with the old workspace, so `verify()` ran one task's checks against another
        task's files and could report a pass for work never attempted.
        """
        import shutil

        workspace = self.root / task.task_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        env = resolve_env(task.env)
        setup_result = setup_task(task, workspace)

        state = EnvState(task_id=task.task_id, workspace=workspace)
        # Snapshot protected paths *after* setup and *before* the agent runs: taken any
        # earlier they would record an empty workspace, any later they would record the
        # agent's own edits as the baseline.
        state.protected_digests = digest_paths(workspace, task.protected_paths)
        if task.checkpoints:
            state.checkpoint_timeline.append(run_checkpoints(task, workspace))

        self.task = task
        self._env = env
        self._setup_result = setup_result
        self._final_sampled = False
        self.state = state
        return state

    @property
    def setup_failed(self) -> bool:
        return self._setup_result is not None and not self._setup_result.passed

    @property
    def setup_result(self) -> VerificationResult | None:
        return self._setup_result

    def step(self, action: Step) -> Observation:
        """Perform one action. Every call advances the clock, including a refused one.

        Counting only calls that reached the executor let a policy spamming a disallowed
        tool run forever: a caller bounding the episode on `step_count` never saw it move.
        A refused call is still an action the agent chose to take.
        """
        if self.task is None or self.state is None or self.state.workspace is None:
            raise RuntimeError("environment.step called before reset")

        self.state.step_count += 1

        if self.setup_failed:
            # The world was never built, so nothing done in it means anything. Refusing
            # here rather than trusting callers to check `setup_failed` first: the runner
            # returns early on setup failure, and an environment that quietly proceeds
            # scores infrastructure breakage as a real attempt.
            return Observation(ok=False, content="task setup failed; the environment was never built")

        if action.tool not in set(self.task.tools):
            # The task decides which tools exist. Refusing here keeps a read-only task
            # from handing out a shell just because a policy named one.
            self._maybe_sample()
            return Observation(
                ok=False,
                content=f"tool {action.tool!r} is not available for this task; allowed: {sorted(self.task.tools)}",
            )
        ok, content = self.executor.execute(
            action.tool or "", action.args, workspace=self.state.workspace, env=self._env
        )
        if self.task.checkpoints and self.state.step_count % self.task.checkpoint_every == 0:
            self.state.checkpoint_timeline.append(run_checkpoints(self.task, self.state.workspace))
        return Observation(ok=ok, content=content)

    def _maybe_sample(self) -> None:
        assert self.task is not None and self.state is not None and self.state.workspace is not None
        if self.task.checkpoints and self.state.step_count % self.task.checkpoint_every == 0:
            self.state.checkpoint_timeline.append(run_checkpoints(self.task, self.state.workspace))

    def _freeze_protected(self) -> dict[str, str | None]:
        """Digest the protected paths once, the first time any grader is about to run.

        Recomputing it per call made the verdict depend on call order: `verify()` runs a
        shell grader in this workspace, so a second `outcome()` -- or the very sequence the
        docstring above recommends, `verify()` then `outcome()` -- would digest a tree the
        grader had already written to and charge its file to the agent. Frozen once, the
        answer is the same however many times it is asked.
        """
        if self.state is None or self.state.workspace is None or self.task is None:
            raise RuntimeError("environment used before reset")
        if self._protected_after is None:
            self._protected_after = digest_paths(self.state.workspace, self.task.protected_paths)
        return self._protected_after

    def verify(self) -> VerificationResult:
        """The published checks only. See `outcome()` for the verdict to actually use."""
        if self.task is None or self.state is None or self.state.workspace is None:
            raise RuntimeError("environment.verify called before reset")
        self._freeze_protected()
        if self.setup_failed:
            return self._setup_result  # type: ignore[return-value]
        if self.task.checkpoints and not self._final_sampled:
            # Exactly one final sample, so the last state of every objective is recorded
            # regardless of where the episode stopped. Guarded because verify() is a
            # question a caller may ask more than once, and each unguarded call both
            # lengthened the timeline and re-ran every checkpoint subprocess.
            self.state.checkpoint_timeline.append(run_checkpoints(self.task, self.state.workspace))
            self._final_sampled = True
        return verify_task(self.task, self.state.workspace)

    def verify_withheld(self) -> VerificationResult | None:
        if self.task is None or self.state is None or self.state.workspace is None:
            raise RuntimeError("environment.verify_withheld called before reset")
        self._freeze_protected()
        return verify_hidden(self.task, self.state.workspace)

    def outcome(self, trajectory: AgentTrajectory | None = None) -> EnvOutcome:
        """The verdict the pipeline should trust: published AND withheld AND not cheated.

        `verify()` alone is not it. An agent that overwrote a protected file, or that
        passed the published checks and failed the withheld ones, gets a green
        `verify()` -- and a caller wiring `reward(verified=env.verify().passed, ...)`
        pays out for a run the runner disqualifies. Since that verdict becomes training
        data, the composed answer is the one that gets a name.

        **The trajectory is what two of the three integrity detectors read.** Without it
        there are no steps and no final answer, so `check_verification_ran` and
        `check_unmeasured_claims` return clean without having looked at anything -- leaving
        the protected-path diff and reporting a third of the anti-cheat as all of it.
        Omitting it therefore raises a warning rather than passing quietly: a report that
        cannot say what it did not check is worse than no report.
        """
        from hermesbench.integrity import check_integrity, enforce

        if self.task is None or self.state is None or self.state.workspace is None:
            raise RuntimeError("environment.outcome called before reset")

        protected_after = self._freeze_protected()

        public = self.verify()
        hidden = self.verify_withheld()

        observed = trajectory or AgentTrajectory(task=self.task.prompt, steps=(), success=public.passed)
        integrity = check_integrity(
            observed,
            protected_before=self.state.protected_digests,
            protected_after=protected_after,
            verification_tools=self.task.verification_tools,
        )
        if trajectory is None:
            integrity = IntegrityReport(
                signals=(
                    *integrity.signals,
                    IntegritySignal(
                        code=INTEGRITY_PARTIAL,
                        severity=WARNING,
                        detail=(
                            "outcome() was called without a trajectory, so only the protected-path "
                            "check ran; a clean report here does not mean the episode was clean"
                        ),
                    ),
                )
            )
        both = public.passed and (hidden.passed if hidden is not None else True)
        return EnvOutcome(
            public=public,
            hidden=hidden,
            integrity_disqualified=integrity.disqualified,
            verified=enforce(both, integrity),
            integrity=integrity,
        )

    def reward(self, *, verified: bool, tool_calls: int, failed_calls: int) -> Reward:
        """Success dominates; efficiency and cleanliness only modulate a success.

        A failed run scores 0 rather than a small positive, because a reward that pays
        anything for a tidy failure is one an agent can farm by failing tidily.

        Every input is clamped. Unclamped, `failed_calls > tool_calls` drove cleanliness
        negative and a *verified* run scored below the 0.0 floor of an unverified one --
        inverting the ordering this function exists to guarantee. Those inputs are
        reachable: `load_jsonl` does not call `validate()`, so a trajectory with more
        results than calls can be loaded and measured.
        """
        if not verified:
            return Reward(verified=False, total=0.0, components={"success": 0.0})

        calls = max(0, int(tool_calls))
        failures = min(max(0, int(failed_calls)), calls)

        if calls == 0:
            # A verified success that took no action is anomalous, not excellent. Scoring
            # it the global maximum would make "do nothing" the optimal policy the moment
            # any upstream bug lets a no-op verify.
            efficiency = 0.5
            cleanliness = 0.5
        else:
            # Fewer calls is better, with diminishing effect -- a 12-call solution beating
            # a 40-call one matters; 12 against 11 does not.
            efficiency = 1.0 / (1.0 + calls / 25.0)
            cleanliness = 1.0 - failures / calls

        components = {"success": 1.0, "efficiency": efficiency, "cleanliness": cleanliness}
        total = min(1.0, max(0.0, 0.7 + 0.2 * efficiency + 0.1 * cleanliness))
        return Reward(verified=True, total=total, components=components)

    def snapshot(self) -> dict[str, Any]:
        return {
            "environment": self.name,
            "state": self.state.to_record() if self.state else None,
            "setup_failed": self.setup_failed,
        }

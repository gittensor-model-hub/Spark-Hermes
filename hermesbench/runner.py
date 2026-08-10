"""HermesBench episode runner.

Wires the pieces together: an `AgentPolicy` proposes steps, a `ToolExecutor` performs
them and reports what actually happened, and `hermesbench.verify` decides afterwards
whether the task was done. The trajectory that falls out is the same
`hermes.trajectory.AgentTrajectory` shape used for training -- an executed episode is
training data, with `metadata.executed = true` to distinguish it from the simulated rows
`hermes.generate` produces.

**Running a policy's commands executes untrusted, model-authored code.** `LocalToolExecutor`
therefore refuses to start unless the caller passes `allow_unsandboxed=True`, which is an
assertion by the operator that the process is already inside a container, VM, or
throwaway machine. The flag provides no isolation of its own -- it exists so that
"benchmark run wiped my home directory" requires someone to have typed the words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hermes.pin import load_tool_schemas
from hermes.protocol import DIALECTS
from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench import BENCH_VERSION
from hermesbench.integrity import IntegrityReport, check_integrity, digest_paths, enforce
from hermesbench.metrics import EpisodeMetrics, SuiteMetrics, episode_metrics, suite_metrics
from hermesbench.repeats import RepeatedSuite, TaskRepeats
from hermesbench.sink import EpisodeSink, JsonlEpisodeSink
from hermesbench.tasks import Task, load_suite
from hermesbench.verify import (
    OBSERVATION_LIMIT,
    VerificationResult,
    resolve_env,
    run_checkpoints,
    setup_task,
    verify_hidden,
    verify_task,
)

# The committed harness: the tool schemas the model is shown and the system prompt it is
# given. Committed rather than assembled at runtime so both sit in the history alongside
# the results they produced, and so both can be digested into a pin.
HARNESS_DIR = Path(__file__).resolve().parent / "harness"


class SandboxError(RuntimeError):
    """Refused to execute model-authored commands without an explicit sandbox assertion."""


class ToolExecutor(Protocol):
    """Performs a tool call and reports the observed outcome."""

    def execute(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        workspace: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Return `(ok, output)`. `ok=False` marks a failure the agent must react to."""
        ...


class AgentPolicy(Protocol):
    """Chooses the next steps given the task and the trajectory so far.

    Returns a batch of steps -- any number of `thinking` steps followed by zero or more
    `tool_call` steps, or a terminal `final` step. The runner executes each tool call
    and appends the real result before asking again, so a policy never sees a tool
    result it invented.
    """

    def next_steps(self, task: Task, history: list[Step]) -> list[Step]: ...

    @property
    def tokens_used(self) -> int: ...


@dataclass
class ReplayPolicy:
    """Replays a recorded trajectory's agent-side steps.

    For exercising the harness itself without a served model: tool results are dropped
    from the recording and re-observed for real, so a replay against a changed workspace
    genuinely fails rather than reproducing the recorded outcome.
    """

    recorded: AgentTrajectory
    tokens: int = 0

    def __post_init__(self) -> None:
        self._batches = _agent_batches(self.recorded.steps)
        self._index = 0

    def next_steps(self, task: Task, history: list[Step]) -> list[Step]:
        if self._index >= len(self._batches):
            return [Step(kind=FINAL, content="replay exhausted")]
        batch = self._batches[self._index]
        self._index += 1
        return batch

    @property
    def tokens_used(self) -> int:
        return self.tokens


def _agent_batches(steps: tuple[Step, ...]) -> list[list[Step]]:
    """Split a recorded trajectory into the batches an agent would have emitted."""
    batches: list[list[Step]] = []
    current: list[Step] = []
    for step in steps:
        if step.kind == TOOL_RESULT:
            if current:
                batches.append(current)
                current = []
            continue
        current.append(step)
        if step.kind == FINAL:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _truncate(text: str) -> str:
    if len(text) <= OBSERVATION_LIMIT:
        return text
    return text[:OBSERVATION_LIMIT] + f"\n... [truncated {len(text) - OBSERVATION_LIMIT} chars]"


class LocalToolExecutor:
    """Runs tool calls directly on the host, inside a task workspace.

    Path arguments are resolved and confined to the workspace: a `file_read` of
    `../../.ssh/id_rsa` fails rather than succeeding quietly. This is containment
    against an agent that wanders, not a security boundary against one that attacks --
    the `terminal` tool runs a shell, and a shell can go anywhere the process can. Real
    isolation is the operator's job.
    """

    def __init__(self, *, allow_unsandboxed: bool = False, timeout_s: int = 120) -> None:
        if not allow_unsandboxed:
            raise SandboxError(
                "LocalToolExecutor runs model-authored shell commands on this host. "
                "Pass allow_unsandboxed=True only from inside a container, VM, or "
                "disposable machine."
            )
        self.timeout_s = timeout_s

    def _resolve(self, workspace: Path, raw: str) -> Path:
        candidate = (workspace / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        root = workspace.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path {raw!r} escapes the task workspace")
        return candidate

    def execute(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        workspace: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        try:
            if tool in ("terminal", "bash", "shell"):
                return self._terminal(str(args.get("command", "")), workspace, env)
            if tool in ("file_read", "read"):
                return True, _truncate(self._resolve(workspace, str(args["path"])).read_text(encoding="utf-8"))
            if tool in ("file_write", "write", "edit"):
                path = self._resolve(workspace, str(args["path"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(args.get("content", "")), encoding="utf-8")
                return True, f"wrote {path.relative_to(workspace.resolve())}"
            if tool == "python":
                return self._python(str(args.get("code", "")), workspace, env)
        except (OSError, KeyError, ValueError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return False, f"unknown tool {tool!r}"

    def _terminal(self, command: str, workspace: Path, env: dict[str, str] | None) -> tuple[bool, str]:
        if not command.strip():
            return False, "empty command"
        return self._run(command, workspace, env, shell=True)

    def _python(self, code: str, workspace: Path, env: dict[str, str] | None) -> tuple[bool, str]:
        """Run a code snippet from a temp file rather than piping it through a shell.

        A heredoc would let any snippet containing the delimiter line escape into the
        surrounding shell, and would mangle snippets whose own quoting collides with the
        shell's. The file lives outside the workspace so it cannot show up in a task's
        verification, but cwd stays the workspace so relative paths in the snippet still
        resolve the way the agent expects.
        """
        if not code.strip():
            return False, "empty code"
        with tempfile.TemporaryDirectory() as scratch:
            snippet = Path(scratch) / "snippet.py"
            snippet.write_text(code, encoding="utf-8")
            # `sys.executable`, not `"python"`. Resolving through PATH makes the score a
            # function of the operator's shell: on a stock container there may be no
            # `python` at all, and on a developer box it can resolve into an unrelated
            # vendored virtualenv sitting earlier in PATH. Either way two people run the
            # same suite and measure two different interpreters.
            return self._run([sys.executable, str(snippet)], workspace, env, shell=False)

    def _run(
        self,
        command: str | list[str],
        workspace: Path,
        env: dict[str, str] | None,
        *,
        shell: bool,
    ) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                command,
                shell=shell,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"command timed out after {self.timeout_s}s"
        except OSError as exc:
            return False, f"{type(exc).__name__}: {exc}"
        output = _truncate((completed.stdout or "") + (completed.stderr or ""))
        return completed.returncode == 0, output or f"(exit {completed.returncode}, no output)"


@dataclass
class EpisodeResult:
    task_id: str
    trajectory: AgentTrajectory
    verification: VerificationResult
    metrics: EpisodeMetrics
    setup_failed: bool = False
    integrity: IntegrityReport = field(default_factory=IntegrityReport)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trajectory": self.trajectory.to_record(),
            "verification": self.verification.to_record(),
            "metrics": self.metrics.to_record(),
            "setup_failed": self.setup_failed,
            "integrity": self.integrity.to_record(),
        }


def _sample_checkpoints(
    task: Task,
    workspace: Path,
    *,
    baselines: tuple[dict[str, str | None], ...],
) -> dict[str, bool]:
    """Run the checkpoint probes, absorbing anything they change on a protected path.

    Checkpoints are maintainer-authored graders, but unlike `verify` they run *interleaved*
    with the episode, so they cannot simply be hoisted outside the digest bracket. If one
    writes to a protected path -- a `.pytest_cache`, a coverage file, a build artifact --
    the next post-step digest sees it and the agent is disqualified for a file it never
    touched. So any path the checkpoint itself changed is folded into every baseline,
    leaving the diff describing only what the agent did.
    """
    if not task.protected_paths:
        return run_checkpoints(task, workspace)
    before = digest_paths(workspace, task.protected_paths)
    sample = run_checkpoints(task, workspace)
    for path, digest in digest_paths(workspace, task.protected_paths).items():
        if before.get(path) != digest:
            for baseline in baselines:
                baseline[path] = digest
    return sample


def _pinned_dialect(default: str = "hermes-3") -> str:
    """The dialect `hermes/base_model.json` pins, or `default` if it pins none.

    Falls back rather than raising: `--help` must work in a checkout whose pin is missing or
    malformed, and a broken pin is better reported by the run than by argument parsing.
    """
    try:
        from hermes.base_model import load as load_pin

        pinned = load_pin().hermes_dialect
    except Exception:
        return default
    return pinned if pinned in DIALECTS else default


def verify_digest(task: Task) -> str:
    """sha256 of the task's published verify script, stamped onto every episode it produces.

    So a log can say which grader produced it. Two commits after the first live baseline, two
    verifiers were changed because they invoked a bare `python` the harness does not guarantee;
    the episodes they had already written still read as a 0/10 capability gap against graders
    that now pass 10/10, and nothing in the record could tell the difference.

    Published script only -- the withheld one would need the salt, and the observed failure was
    in the public half.
    """
    return "sha256:" + hashlib.sha256((task.verify or "").encode("utf-8")).hexdigest()


def run_episode(
    task: Task,
    policy: AgentPolicy,
    executor: ToolExecutor,
    workspace: Path,
    *,
    price_book: Any = None,
    model: str = "",
) -> EpisodeResult:
    """Run one task to completion and score it against the task's own verification.

    With a `price_book` and a policy that reports normalised `usage`, the episode is also
    priced. Without one the cost stays `None` rather than becoming 0.0: an unpriced run
    reported as free is the failure `PriceBook.price_of` exists to refuse, and it would
    arrive here through the back door.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Off the policy, not passed in. The policy is what renders the system prompt in a dialect,
    # so its own field is the truth about what this episode actually ran under -- a parameter
    # could disagree with it, and the point of stamping is that the log cannot.
    dialect_name = str(getattr(getattr(policy, "dialect", None), "name", "") or "")

    # Read the same way `usage` is below. The prompt is what makes an executed trajectory
    # exportable: it is the context that caused these tokens, and `_system_content` refuses
    # a row whose assistant turns are paired with a prompt that did not produce them.
    # `""` from a policy that carries no prompt stays falsy and is recorded as None.
    system_prompt = getattr(policy, "system", None) or None

    setup_result = setup_task(task, workspace)
    if setup_result is not None and not setup_result.passed:
        trajectory = AgentTrajectory(
            task=task.prompt,
            steps=(Step(kind=FINAL, content="setup failed; episode not run"),),
            success=False,
            task_id=task.task_id,
            tools_available=task.tools,
            system=system_prompt,
            source=f"hermesbench:{BENCH_VERSION}",
            metadata={"executed": True, "setup_error": setup_result.stderr, "harness_final": True},
        )
        return EpisodeResult(
            task_id=task.task_id,
            trajectory=trajectory,
            verification=setup_result,
            metrics=episode_metrics(
                trajectory,
                task_id=task.task_id,
                verified_success=False,
                wall_time_s=time.monotonic() - started,
                mutating_tools=task.mutating_tools,
                setup_failed=True,
                verify_digest=verify_digest(task),
                dialect=dialect_name,
            ),
            setup_failed=True,
        )

    protected_before = digest_paths(workspace, task.protected_paths)
    protected_after = dict(protected_before)
    baselines = (protected_before, protected_after)

    steps: list[Step] = []
    allowed_tools = set(task.tools)
    verification_tools = set(task.verification_tools)
    finished = False
    max_steps_hit = False
    stalled = False
    agent_steps = 0
    verification_steps = 0
    task_env = resolve_env(task.env)

    # Pass/fail of every sub-objective, sampled as the episode runs. This is what turns
    # "the agent drifted" into something checkable: an objective that passes at one
    # sample and fails at a later one was undone by the agent's own later work.
    checkpoint_timeline: list[dict[str, bool]] = []
    if task.checkpoints:
        checkpoint_timeline.append(_sample_checkpoints(task, workspace, baselines=baselines))

    while not finished and not max_steps_hit:
        before = agent_steps + verification_steps
        for step in policy.next_steps(task, list(steps)):
            # A policy does not get to author observations. If it emits a tool_result,
            # that is a fabricated one -- drop it and keep only what the executor saw.
            if step.kind == TOOL_RESULT:
                continue

            # Checking your own work draws on a separate allowance. Sharing one budget
            # would mean the harness penalises exactly the behavior `self_check_rate`
            # rewards, and could cut an episode off mid-verification.
            is_verification = step.kind == TOOL_CALL and step.tool in verification_tools
            if is_verification:
                if verification_steps >= task.max_verification_steps:
                    max_steps_hit = True
                    break
            # Checked per step, not per turn: a policy that returns fifty calls in one
            # batch would otherwise run all fifty against the host regardless of budget.
            elif agent_steps >= task.max_steps:
                max_steps_hit = True
                break

            steps.append(step)
            if is_verification:
                verification_steps += 1
            else:
                agent_steps += 1
            if step.kind == TOOL_CALL:
                if step.tool not in allowed_tools:
                    # The task decides which tools exist. Refusing here (rather than
                    # executing anything a policy names) is what keeps a file_read-only
                    # task from handing out a shell, and keeps the trajectory's
                    # tools_available claim true.
                    steps.append(
                        Step(
                            kind=TOOL_RESULT,
                            call_id=step.call_id,
                            content=f"tool {step.tool!r} is not available for this task; allowed: {sorted(allowed_tools)}",
                            ok=False,
                        )
                    )
                    continue
                ok, output = executor.execute(step.tool or "", step.args, workspace=workspace, env=task_env)
                steps.append(Step(kind=TOOL_RESULT, call_id=step.call_id, content=output, ok=ok))
                # Re-digest after every agent action, never after a grader. Checkpoints
                # run interleaved with the episode and `verify`/`hidden_verify` run at the
                # end, all of them shell commands in this same directory -- so a snapshot
                # taken at any later point can contain a file the maintainer's own grader
                # wrote, and `protected_path_created` is disqualifying. This is the only
                # sampling point that means "the state the agent left behind".
                protected_after = digest_paths(workspace, task.protected_paths)
            elif step.kind == FINAL:
                finished = True
                break

            if task.checkpoints and (agent_steps + verification_steps) % task.checkpoint_every == 0:
                checkpoint_timeline.append(_sample_checkpoints(task, workspace, baselines=baselines))

        if not finished and not max_steps_hit and (agent_steps + verification_steps) == before:
            # The turn produced nothing -- an empty response, or a batch of only
            # fabricated tool_results. The step budget can never be reached from here,
            # so without this the loop spins forever on a single stuck episode.
            stalled = True
            break

    # These closing steps are written by the harness, not the agent. They are flagged in
    # metadata so a training filter can drop them -- rendered naively they would teach a
    # student to answer "step budget exhausted".
    if max_steps_hit:
        steps.append(Step(kind=FINAL, content=f"step budget exhausted ({task.max_steps})"))
    elif stalled:
        steps.append(Step(kind=FINAL, content="policy returned no actionable steps"))

    # Final sample, so the last state of every objective is always on the timeline
    # regardless of where the episode happened to stop relative to checkpoint_every.
    if task.checkpoints:
        checkpoint_timeline.append(_sample_checkpoints(task, workspace, baselines=baselines))

    verification = verify_task(task, workspace)
    # Withheld checks decide the real outcome; the published ones can be trained against.
    hidden = verify_hidden(task, workspace)
    public_passed = verification.passed
    hidden_passed = hidden.passed if hidden is not None else None
    both_passed = public_passed and (hidden_passed is not False)
    integrity = check_integrity(
        AgentTrajectory(task=task.prompt, steps=tuple(steps), success=verification.passed),
        protected_before=protected_before,
        protected_after=protected_after,
        verification_tools=task.verification_tools,
    )
    # A cheated pass is not a weak pass; it is not a pass.
    verified_success = enforce(both_passed, integrity)
    elapsed = time.monotonic() - started

    usage = getattr(policy, "usage", None)
    usage_record = usage.to_record() if usage is not None else None
    cost = None
    if price_book is not None and usage is not None:
        if not model:
            raise ValueError("pricing an episode needs the model name; a price book cannot price an anonymous run")
        cost = price_book.cost(model, usage).total

    trajectory = AgentTrajectory(
        task=task.prompt,
        steps=tuple(steps),
        success=verified_success,
        task_id=task.task_id,
        tools_available=task.tools,
        system=system_prompt,
        source=f"hermesbench:{BENCH_VERSION}",
        metadata={
            "executed": True,
            "verify_exit_code": verification.exit_code,
            "integrity_disqualified": integrity.disqualified,
            "public_passed": public_passed,
            "hidden_passed": hidden_passed,
            "harness_final": max_steps_hit or stalled,
        },
    )
    return EpisodeResult(
        task_id=task.task_id,
        trajectory=trajectory,
        verification=verification,
        metrics=episode_metrics(
            trajectory,
            # Read off the policy rather than recounted from the trajectory: by the time a
            # malformed turn reaches `steps`, `steps_from_turn` has already flattened it
            # into ordinary reasoning text and the evidence is gone.
            malformed_turns=int(getattr(policy, "parse_failures", 0)),
            task_id=task.task_id,
            verified_success=verified_success,
            tokens_used=policy.tokens_used,
            cost=cost,
            usage=usage_record,
            wall_time_s=elapsed,
            mutating_tools=task.mutating_tools,
            max_steps_hit=max_steps_hit,
            verify_digest=verify_digest(task),
            dialect=dialect_name,
            category=task.category,
            public_passed=public_passed,
            hidden_passed=hidden_passed,
            checkpoint_timeline=tuple(checkpoint_timeline),
        ),
        integrity=integrity,
    )


def run_suite(
    tasks: list[Task],
    policy_factory: Any,
    executor: ToolExecutor,
    workspace_root: Path,
    *,
    keep_workspaces: bool = False,
    repeats: int = 1,
    sink: EpisodeSink | None = None,
) -> tuple[SuiteMetrics, list[EpisodeResult]]:
    """Run every task in its own workspace and aggregate the scores.

    With `repeats > 1` each task runs that many times, every attempt in its **own**
    workspace. Sharing one across attempts would let the second attempt start from the
    edits the first left behind, and an agent that inherits a finished repository looks
    both more capable and more efficient than it is.

    Every attempt is returned. Keeping only the best would turn the pass rate into a
    best-of-k order statistic -- `1-(1-p)^k`, which converges to 1.0 for any p above zero --
    and the point of repeating a run is to measure that spread, not to hide it.

    `sink` receives each episode as it finishes. Without one this function is all-or-nothing:
    results accumulate in memory and the caller writes after the last episode returns, so a run
    that dies at episode 189 of 190 produces nothing -- not a partial score, not even a list of
    which tasks got as far as running. Measured on a real baseline: 190 episodes across 8
    shards, every shard log at 0 bytes until its shard finished. Hours of paid inference would
    have bought a traceback. See `hermesbench.sink`.

    Recorded after the episode joins `results`, and never inside a `try` that could swallow a
    scoring error. A sink is observation: it must not be able to change which episodes count,
    and a failed write must surface rather than quietly leave a short log that still looks like
    a complete run.
    """
    if repeats < 1:
        raise ValueError("a suite must be run at least once")
    results: list[EpisodeResult] = []
    for task in tasks:
        for attempt in range(repeats):
            # The suffix is dropped on the first attempt so single-shot runs keep the
            # workspace layout every existing caller and manifest already expects.
            workspace = workspace_root / (task.task_id if attempt == 0 else f"{task.task_id}#{attempt}")
            if workspace.exists() and not keep_workspaces:
                shutil.rmtree(workspace)
            episode = run_episode(task, policy_factory(task), executor, workspace)
            results.append(episode)
            if sink is not None:
                sink.append(episode)
    return suite_metrics([r.metrics for r in results]), results


def repeated_from(results: list[EpisodeResult], *, repeats: int) -> RepeatedSuite:
    """Fold per-attempt results into the flakiness view `hermesbench.repeats` reports.

    Built from results already in hand rather than by calling `repeat_suite` with a
    closure that reruns episodes: the suite has run once by the time anyone wants this,
    and running it again to measure it would double the bill for the same information.
    """
    passes: dict[str, int] = {}
    attempts: dict[str, int] = {}
    for result in results:
        passes[result.task_id] = passes.get(result.task_id, 0) + (1 if result.metrics.success else 0)
        attempts[result.task_id] = attempts.get(result.task_id, 0) + 1
    return RepeatedSuite(
        tasks=tuple(TaskRepeats(task_id=t, passes=passes[t], attempts=attempts[t]) for t in passes),
        repeats=repeats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=BENCH_VERSION, help="bench versions: 'v1', 'v0,v1', or 'all'")
    parser.add_argument("--tags", default="", help="comma-separated tag filter")
    parser.add_argument("--workspace-root", type=Path, required=True, help="scratch directory for task workspaces")
    parser.add_argument("--out", type=Path, default=None, help="write the run manifest here")
    parser.add_argument("--list", action="store_true", help="list the suite's tasks and exit")
    parser.add_argument("--model", default="", help="model id to drive the suite with")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the endpoint's key")
    # Default from the pin, not from a literal. `hermes/base_model.json` records
    # `hermes_dialect: hermes-4` with its evidence -- the model's own chat template emits
    # `<think>` and never `<scratch_pad>` -- and this flag defaulted to `hermes-3`, which
    # instructs the model to use a block it does not speak.
    #
    # The failure is silent, which is why it survived. A run that omits this flag gets a model
    # answering in prose: zero tool calls, zero malformed turns, `protocol_clean: true`. Measured
    # on the rollout host, the same task and model gave 2 steps and 0 tool calls under hermes-3
    # where the pinned dialect drives a working agent. Nothing in the metrics said the dialect was
    # wrong, because the model complied -- with the wrong contract.
    parser.add_argument(
        "--dialect",
        default=_pinned_dialect(),
        choices=sorted(DIALECTS),
        help="Hermes wire dialect; defaults to hermes_dialect from hermes/base_model.json",
    )
    parser.add_argument("--repeats", type=int, default=1, help="run each task N times and report flakiness")
    parser.add_argument(
        "--task-ids",
        default="",
        help="comma-separated task ids, so one baseline can be sharded across processes",
    )
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=None,
        help=(
            "append each episode's metrics here as JSONL as it finishes. Without it a run that "
            "dies before the last episode produces nothing at all -- measured on a 190-episode "
            "baseline whose shard logs sat at 0 bytes until each shard completed."
        ),
    )
    parser.add_argument(
        "--miner-dir",
        type=Path,
        default=None,
        help=(
            "a miner submission to run: validated against hermes/miner_contract.json, then "
            "SOUL.md and SKILL.md are composed into the system prompt. This is both the "
            "validator's execution step and the miner's local evaluation -- the same code path, "
            "so a miner sees what the validator will see."
        ),
    )
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help=(
            "run model-authored shell on this host. Provides no isolation of its own -- it asserts "
            "that this process is already inside a container or a throwaway machine. See the Dockerfile."
        ),
    )
    args = parser.parse_args(argv)

    tags = tuple(t.strip() for t in args.tags.split(",") if t.strip())
    tasks = load_suite(args.suite, tags=tags)

    # Shard a baseline across processes. A repeated baseline is the only thing that measures
    # the run-to-run spread `hermes.acceptance` refuses to judge a margin without, and it is
    # the most expensive thing anyone runs here: 19 tasks x 10 repeats, sequentially, was
    # measured at roughly three minutes per episode against a served 27B on one card, so about
    # nine hours. Episodes are already independent by construction -- every attempt gets its
    # own workspace -- so the only thing preventing them running side by side was that a task
    # could not be named on the command line. `--tags` cannot substitute: tags are shared
    # across tasks, so no tag selection partitions the suite.
    #
    # Running shards concurrently changes wall time and nothing else that is scored. Tokens
    # and tool calls are per-episode counts, unaffected by what else the server is doing --
    # which is the same reason `dominates()` refuses to gate on latency at all.
    if args.task_ids:
        wanted = tuple(t.strip() for t in args.task_ids.split(",") if t.strip())
        known = {t.task_id for t in tasks}
        missing = [w for w in wanted if w not in known]
        if missing:
            # Refused rather than skipped. A typo would otherwise shrink one shard silently,
            # and the merged baseline would be short some attempts with nothing to show it --
            # which is precisely the kind of missing evidence the acceptance gates exist to
            # refuse, arriving in a form they cannot see.
            print(
                f"hermesbench: no such task(s) in suite {args.suite!r}: {', '.join(missing)}.\n"
                "A misspelled id would otherwise drop that task from this shard, and the merged "
                "baseline would be quietly missing attempts.",
                file=sys.stderr,
            )
            return 2
        order = {task_id: i for i, task_id in enumerate(wanted)}
        tasks = sorted((t for t in tasks if t.task_id in order), key=lambda t: order[t.task_id])

    # Checked before anything is paid for. `build_manifest` refuses to fingerprint a task
    # with a withheld check and no salt, and discovering that after a suite has run means
    # the run happened and the manifest cannot be written.
    salt = os.environ.get("HERMESBENCH_WITHHELD_SALT", "")
    if args.out and any(t.has_hidden_tests for t in tasks) and len(salt) < 16:
        print(
            "hermesbench: this suite has withheld checks, so writing a manifest needs "
            "HERMESBENCH_WITHHELD_SALT set to at least 16 characters.\n"
            "Without it the suite digest would either omit the withheld tests -- so two suites "
            "with different ones would digest the same -- or publish a guessable digest of them.",
            file=sys.stderr,
        )
        return 2

    if args.list:
        for task in tasks:
            print(f"{task.task_id}\t{','.join(task.tags)}\t{task.prompt[:70]}")
        return 0

    if not args.model or not args.base_url:
        # Still a refusal, but a narrower one than before: the harness can run now, it
        # just will not invent a model to run against. Reporting success_rate 0.0 for a
        # suite nothing attempted would be a measurement of nothing.
        print(
            "hermesbench: --model and --base-url are required to run a suite.\n"
            "Point them at any OpenAI-compatible endpoint (vLLM, SGLang, a hosted gateway).\n"
            "Use --list to inspect the suite without running it.",
            file=sys.stderr,
        )
        return 2

    from hermesbench.policy import ServedModelPolicy, openai_completion

    schemas = load_tool_schemas(HARNESS_DIR / "tools.json")
    system_prompt = (HARNESS_DIR / "system_prompt.txt").read_text(encoding="utf-8")

    # The step that made the miner-editable surface reachable. `hermes.miner_contract` and
    # `hermes.profile` were both built and tested and called from nowhere but tests, so a
    # submission could be validated and then had no way to affect a run: this file read one
    # system prompt from one path. A miner could write a perfect SKILL.md and change nothing.
    #
    # Validated BEFORE the model is contacted. Discovering a contract violation after a suite
    # has run means the run was paid for and cannot be scored.
    if args.miner_dir is not None:
        from hermes.base_model import load as load_pin
        from hermes.profile import PINNED_CONFIG_KEYS, ProfileError, assemble, compose_system_prompt

        pin = load_pin()
        try:
            profile = assemble(
                # The agent is pinned by commit elsewhere; a run driven from this CLI is
                # validator-executed by construction, which is what makes the config pin hold.
                agent_repository="NousResearch/hermes-agent",
                agent_commit="0" * 40,
                model_repository=pin.repository,
                model_revision=pin.revision,
                config=dict.fromkeys(PINNED_CONFIG_KEYS, "pinned"),
                miner_dir=args.miner_dir,
                validator_executed=True,
            )
            system_prompt = compose_system_prompt(system_prompt, args.miner_dir)
        except ProfileError as exc:
            print(f"hermesbench: miner submission refused: {exc}", file=sys.stderr)
            return 2
        print(
            f"hermesbench: running submission {args.miner_dir} "
            f"({len(profile.miner_files)} file(s): {', '.join(profile.miner_files)}); "
            f"system prompt {len(system_prompt)} chars",
            file=sys.stderr,
        )
    dialect = DIALECTS[args.dialect]
    pinned = _pinned_dialect()
    if args.dialect != pinned:
        # Not refused. Comparing dialects is legitimate work, and the epoch pins the model rather
        # than the harness's wire format. But it must be loud: the measured cost of getting this
        # wrong is an entire run of prose answers that reports a clean protocol.
        print(
            f"hermesbench: WARNING dialect {args.dialect!r} is not the pinned {pinned!r}. The pinned "
            "model's chat template decides which blocks it emits, and instructing another dialect "
            "produces a model that answers in prose -- zero tool calls, zero malformed turns, and "
            "protocol_clean true, because it complied with the wrong contract.",
            file=sys.stderr,
        )
    complete = openai_completion(base_url=args.base_url, model=args.model, api_key=os.environ.get(args.api_key_env, ""))

    def policy_factory(task: Task) -> ServedModelPolicy:
        return ServedModelPolicy(
            complete=complete,
            dialect=dialect,
            tool_schemas=schemas,
            system=system_prompt,
            scratch_pad=dialect.supports_scratch_pad,
        )

    executor = LocalToolExecutor(allow_unsandboxed=args.allow_unsandboxed)
    sink = JsonlEpisodeSink(args.episodes_out) if args.episodes_out else None
    try:
        metrics, results = run_suite(
            tasks, policy_factory, executor, args.workspace_root, repeats=args.repeats, sink=sink
        )
    finally:
        # Closed in `finally` so a crashed run still flushes the episode in flight. The whole
        # point is that a killed run leaves something readable behind.
        if sink is not None:
            sink.close()

    record = metrics.to_record()
    # The interval is reported on every run, not only repeated ones. A fifteen-task suite
    # locates its success rate to about a thirty-point span whatever the model does, and
    # that is a property of the suite size which no number of reruns shrinks -- printing
    # the point estimate alone invites a comparison the data cannot support.
    repeated = repeated_from(results, repeats=args.repeats)
    record["repeats"] = repeated.to_record()
    print(json.dumps(record, indent=2))
    if repeated.flaky_tasks:
        print(
            f"hermesbench: {len(repeated.flaky_tasks)} task(s) did not agree with themselves across "
            f"{args.repeats} attempts: {', '.join(t.task_id for t in repeated.flaky_tasks)}",
            file=sys.stderr,
        )

    if args.out:
        manifest = build_manifest(
            tasks,
            results,
            model=args.model,
            suite_name=args.suite,
            executor="local",
            metrics=metrics,
            tool_timeout_s=executor.timeout_s,
            salt=salt,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest.to_record(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote run manifest to {args.out}", file=sys.stderr)
    return 0


def build_manifest(
    tasks: list[Task],
    results: list[EpisodeResult],
    *,
    model: str,
    suite_name: str,
    executor: str,
    metrics: Any,
    tool_timeout_s: int = 0,
    salt: str = "",
) -> Any:
    """Assemble the published record of a run.

    Per-task results rather than an aggregate: a reader who doubts the headline needs
    something to examine, and `RunManifest` refuses a manifest without them. The harness
    pin is read from the repository, so a dirty tree fails here rather than producing a
    manifest that describes a state nobody can reproduce.
    """
    from hermes.harness import RunManifest, TaskResult, digest_suite, fingerprint_task, harness_digest
    from hermes.pin import build_pin
    from hermes.pin import load_tool_schemas as _load

    schemas = _load(HARNESS_DIR / "tools.json")
    pin = build_pin(
        system_prompt=(HARNESS_DIR / "system_prompt.txt").read_text(encoding="utf-8"),
        tool_schemas=schemas,
        container_image_digest=os.environ.get("HERMESBENCH_IMAGE_DIGEST", ""),
    )
    suite = digest_suite(suite_name, [fingerprint_task(t, salt=salt) for t in tasks])
    return RunManifest(
        model=model,
        suite=suite,
        harness=harness_digest(
            pin,
            suite=suite,
            executor=executor,
            observation_limit=OBSERVATION_LIMIT,
            # Threaded from the executor that actually ran. Introspecting the class gave 0
            # unconditionally -- its parameters are keyword-only, so __defaults__ is None --
            # and a harness digest that always records 0 does not describe the run.
            tool_timeout_s=tool_timeout_s,
        ),
        results=tuple(
            TaskResult(
                task_id=r.task_id,
                passed=r.metrics.success,
                hidden_passed=r.metrics.hidden_passed,
                disqualified=r.integrity.disqualified,
                steps=r.metrics.steps,
                # Carried so the published manifest can support the claim the competition is
                # decided on. Without these the token margin and the run-to-run spread that
                # `hermes.acceptance` gates on existed only in this process's stdout.
                tokens_used=r.metrics.tokens_used,
                tool_calls=r.metrics.tool_calls,
                wall_time_s=r.metrics.wall_time_s,
            )
            for r in results
        ),
        metrics={
            "success_rate": metrics.success_rate,
            "tool_efficiency": metrics.tool_efficiency,
            "mean_tokens": metrics.mean_tokens,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())

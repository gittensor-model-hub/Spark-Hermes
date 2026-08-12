"""The validity gate: what a generated task has to prove before anyone trains on it.

A generator that emits tasks nobody checks is a machine for manufacturing the defect this whole
project keeps finding -- an absence recorded as a measurement. A broken task produces failed episodes
that look exactly like a model that could not do the work, and those episodes then become `rejected`
examples teaching the model to avoid something that was never its fault.

Nineteen tasks were hand-written for this suite, by someone who was paying attention, and they still
shipped:

  * a `bundle.json` landing inside the agent's surface, which made EVERY judged run exit 2
  * two graders invoking a bare `python` the harness does not guarantee -- 10/10 failures caused
    by the grader, logged as a capability gap
  * a `command -v make` check fooled by a shim on PATH
  * a token-boundary bug the published check passed and the withheld one caught

At three hundred generated tasks that class of defect arrives at scale and silently. So each task
proves eight things, in a throwaway workspace, before it is allowed to exist:

  1. setup exits 0
  2. setup is DETERMINISTIC -- run twice, byte-identical
  3. the public check FAILS the untouched workspace
  4. the withheld check FAILS the untouched workspace
  5. the reference solution exits 0
  6. the public check PASSES after the reference solution
  7. the withheld check PASSES after the reference solution
  8. the public and withheld checks DISAGREE on a deliberately overfit solution

Eight is the one that matters most and the one a generator will most often fail. A withheld check
that agrees with the published one everywhere is not withholding anything: it adds cost, produces an
`overfit_rate` that is structurally zero, and creates the appearance of a second opinion where there
is one opinion twice. Checks 3 and 4 are its mirror -- a check that passes an untouched workspace is
measuring nothing at all, and would mark every episode a success.

Check 2 exists because a task whose setup varies between runs cannot have a withheld check pinned to
values, and the failure shows up much later as a check that passes on the author's machine. The
hand-written tasks solve this with a seeded LCG and a literal mtime; a generated one has to prove it.

Nothing here trusts a model's opinion about its own output. Every check is an exit code from a real
process against a real directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Long enough for a real setup that unpacks or generates data, short enough that a runaway generated
# script cannot hold a slot for an hour. A task that needs longer than this to build its own
# workspace is too heavy for a suite meant to run hundreds of episodes.
STEP_TIMEOUT_S = 120


class GateError(RuntimeError):
    """The gate could not be run at all, as distinct from a task failing it."""


@dataclass
class Candidate:
    """Everything a generated task must supply to be judged."""

    task_id: str
    setup: str
    verify: str
    withheld_verify: str
    reference_solution: str
    # A solution that satisfies the letter of the published check while doing none of the work --
    # hardcoding the expected answer, touching the output file, echoing a constant. Required, not
    # optional: without it check 8 cannot run, and check 8 is the one that decides whether the
    # withheld check is worth its cost.
    cheat_solution: str


@dataclass
class Verdict:
    """Why a task was accepted, or exactly which check killed it."""

    task_id: str
    accepted: bool
    failed_check: str = ""
    detail: str = ""
    checks_run: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "accepted": self.accepted,
            "failed_check": self.failed_check,
            "detail": self.detail[:400],
            "checks_run": list(self.checks_run),
        }


def _run(script: str, workspace: Path, *, timeout: int = STEP_TIMEOUT_S) -> tuple[int, str]:
    """One script against one workspace. Exit code and combined output, never an exception.

    `PATH` gets the workspace's own `bin/` prepended the same way the runner does, so a task that
    sabotages a tool is sabotaged here too -- a gate that ran in a different environment from the
    episodes would bless tasks that then behave differently under measurement.
    """
    env = dict(os.environ)
    env["PATH"] = f"{workspace / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        done = subprocess.run(
            ["/bin/sh", "-c", script],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except OSError as exc:  # pragma: no cover - only on a broken host
        raise GateError(f"could not run a gate step: {exc}") from exc
    return done.returncode, (done.stdout + done.stderr)[-4000:]


def _fingerprint(workspace: Path) -> str:
    """A digest of every file's path and bytes, so two setups can be compared exactly.

    Paths are sorted and hashed alongside the content: two workspaces holding the same bytes under
    different names are not the same workspace, and a check pinned to one would fail on the other.
    `__pycache__` is skipped because the interpreter writes it and its content varies by version --
    that is the harness's noise, not the task's.
    """
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        rel = path.relative_to(workspace).as_posix()
        digest.update(rel.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _build(candidate: Candidate, root: Path, timeout: int) -> tuple[Path, str]:
    workspace = Path(tempfile.mkdtemp(dir=root, prefix=f"{candidate.task_id}-"))
    code, output = _run(candidate.setup, workspace, timeout=timeout)
    if code != 0:
        raise _Failed("setup_exits_zero", f"setup exited {code}: {output}")
    return workspace, _fingerprint(workspace)


class _Failed(Exception):
    def __init__(self, check: str, detail: str) -> None:
        super().__init__(detail)
        self.check = check
        self.detail = detail


def gate(candidate: Candidate, *, root: Path | None = None, timeout_s: int = STEP_TIMEOUT_S) -> Verdict:
    """Run all eight checks. The first failure stops the rest and names itself.

    Stopping early is deliberate: once setup is broken, every later check is measuring the broken
    setup, and a verdict listing five failures where one caused the other four tells a reader less
    than a verdict naming the one.
    """
    verdict = Verdict(task_id=candidate.task_id, accepted=False)
    scratch = Path(tempfile.mkdtemp(prefix="taskgen-gate-")) if root is None else root
    made_scratch = root is None
    try:
        try:
            workspace, first = _build(candidate, scratch, timeout_s)
            verdict.checks_run.append("setup_exits_zero")

            # 2. Determinism. A task whose workspace differs between runs cannot carry a withheld
            # check pinned to values derived from it, and the symptom appears much later as a check
            # that only passes on the machine that wrote it.
            twin, second = _build(candidate, scratch, timeout_s)
            if first != second:
                raise _Failed(
                    "setup_is_deterministic",
                    "two runs of setup produced different workspaces; a withheld check cannot be "
                    "pinned to values that move. Use a seeded generator and literal timestamps, "
                    "never random or the clock",
                )
            shutil.rmtree(twin, ignore_errors=True)
            verdict.checks_run.append("setup_is_deterministic")

            # 3 and 4. A check that passes an untouched workspace marks every episode a success.
            for name, script in (
                ("public_fails_untouched", candidate.verify),
                ("withheld_fails_untouched", candidate.withheld_verify),
            ):
                code, output = _run(script, workspace, timeout=timeout_s)
                if code == 0:
                    raise _Failed(name, f"{name.split('_')[0]} check passed a workspace nobody had touched: {output}")
                verdict.checks_run.append(name)

            # 5, 6, 7. The reference solution is what proves the task is solvable at all. Without it
            # a failed episode is unattributable: the model may be wrong, or the task may be.
            solved = Path(tempfile.mkdtemp(dir=scratch, prefix=f"{candidate.task_id}-solved-"))
            shutil.rmtree(solved)
            shutil.copytree(workspace, solved)
            code, output = _run(candidate.reference_solution, solved, timeout=timeout_s)
            if code != 0:
                raise _Failed("reference_solution_runs", f"the reference solution exited {code}: {output}")
            verdict.checks_run.append("reference_solution_runs")

            for name, script in (
                ("public_passes_reference", candidate.verify),
                ("withheld_passes_reference", candidate.withheld_verify),
            ):
                code, output = _run(script, solved, timeout=timeout_s)
                if code != 0:
                    raise _Failed(
                        name, f"the reference solution did not satisfy the {name.split('_')[0]} check: {output}"
                    )
                verdict.checks_run.append(name)

            # 8. The one that decides whether the withheld check earns its cost.
            cheated = Path(tempfile.mkdtemp(dir=scratch, prefix=f"{candidate.task_id}-cheat-"))
            shutil.rmtree(cheated)
            shutil.copytree(workspace, cheated)
            _run(candidate.cheat_solution, cheated, timeout=timeout_s)
            public_code, _ = _run(candidate.verify, cheated, timeout=timeout_s)
            withheld_code, withheld_output = _run(candidate.withheld_verify, cheated, timeout=timeout_s)
            if public_code != 0:
                raise _Failed(
                    "checks_disagree_on_a_cheat",
                    "the cheat solution did not even pass the published check, so this task cannot "
                    "show that the withheld check catches anything the published one misses. Write a "
                    "cheat that satisfies the published assertions while doing none of the work",
                )
            if withheld_code == 0:
                raise _Failed(
                    "checks_disagree_on_a_cheat",
                    "the withheld check passed a solution that did none of the work, so it agrees "
                    f"with the published check everywhere and withholds nothing: {withheld_output}",
                )
            verdict.checks_run.append("checks_disagree_on_a_cheat")

            verdict.accepted = True
        except _Failed as failure:
            verdict.failed_check = failure.check
            verdict.detail = failure.detail
    finally:
        if made_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    return verdict


ALL_CHECKS = (
    "setup_exits_zero",
    "setup_is_deterministic",
    "public_fails_untouched",
    "withheld_fails_untouched",
    "reference_solution_runs",
    "public_passes_reference",
    "withheld_passes_reference",
    "checks_disagree_on_a_cheat",
)

__all__ = ["ALL_CHECKS", "STEP_TIMEOUT_S", "Candidate", "GateError", "Verdict", "gate"]

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
proves eleven things, in a throwaway workspace, before it is allowed to exist:

  0. no script reaches outside the workspace -- the one static check, see FORBIDDEN_ROOTS
  1. setup exits 0 (under `set -e`, so EVERY command in it must succeed)
  2. setup is DETERMINISTIC -- run twice, byte-identical
  3. the public check FAILS the untouched workspace
  4. the withheld check FAILS the untouched workspace
  5. the reference solution exits 0
  6. the public check PASSES after the reference solution
  7. the withheld check PASSES after the reference solution
  8. the public and withheld checks DISAGREE on a deliberately overfit solution
  8b. the protected paths exist and the reference solution does not modify them
  9. the withheld check ACCEPTS a second solution that reaches the same result another way

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
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Long enough for a real setup that unpacks or generates data, short enough that a runaway generated
# script cannot hold a slot for an hour. A task that needs longer than this to build its own
# workspace is too heavy for a suite meant to run hundreds of episodes.
STEP_TIMEOUT_S = 120


# Absolute paths a generated script may never touch, and the one guard here that is static rather
# than executed.
#
# Everything else this gate asserts is an exit code from a real process, on the principle that a
# model's claim about its own output is not evidence. This check is the exception because the
# evidence arrives too late to be useful: the prompt's own requirement 7 spells out the two
# outcomes -- "on a machine where that path is not writable your setup dies on its first line, and
# on one where it IS writable it writes into somebody's system" -- and only the first is
# observable. The second passes every check while escaping the workspace, and `_fingerprint`
# cannot see it because it walks the workspace and nothing else. Measured: a setup writing a
# counter to /tmp ran twice, incremented twice, and was accepted 9/9 by the determinism check.
#
# So this is the roots requirement 7 actually names, plus the ones the repo has measured itself
# failing on (`mkdir /workspace`, `mkdir /opt/myapp`). `/usr`, `/bin` and `/dev` are deliberately
# NOT here: those are overwhelmingly read-only interpreter and device use -- `/bin/sh`,
# `/usr/bin/env`, `/dev/null` -- and denying them would reject correct scripts, which costs a
# generation each.
FORBIDDEN_ROOTS = ("workspace", "tmp", "etc", "var", "opt", "root", "home", "srv", "mnt")

_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w./-])/(?:" + "|".join(FORBIDDEN_ROOTS) + r")(?:/|\b)")
# `~/` and `$HOME` are the same escape spelled two ways; a guard that caught one and not the other
# would be teaching the model which spelling to use.
_HOME_RE = re.compile(r"(?<![\w./-])~/|\$\{?HOME\b")
_SUDO_RE = re.compile(r"(?<![\w-])sudo(?![\w-])")
# `<<EOF`, `<<'EOF'`, `<<"EOF"`, `<<-EOF`: the word that ends a heredoc body.
_HEREDOC_OPEN_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _escapes_workspace(script: str) -> str:
    """The first command line of `script` that reaches outside the workspace, or "".

    Two kinds of line are not commands and are skipped. Comments, because prose explaining why a
    path is NOT used is not a use of it. And heredoc bodies, because they are DATA being written into
    the workspace, not instructions: a generated task that writes a runbook saying "run
    `ssh-add ~/.ssh/id_rsa`" is documenting a system, not touching the home directory -- and that
    exact task was rejected by the first version of this check, on real model output, 1 in 11.

    Heredoc tracking is deliberately simple: the terminator is the bare word on its own line, with
    leading tabs tolerated for `<<-`. That is what `/bin/sh` does, and a body the shell would not
    end is a setup that fails check 1 anyway.
    """
    terminator: str | None = None
    for number, line in enumerate(script.splitlines(), start=1):
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        bare = line.strip()
        if not bare or bare.startswith("#"):
            continue
        for pattern, what in (
            (_ABSOLUTE_PATH_RE, "an absolute path"),
            (_HOME_RE, "a home-relative path"),
            (_SUDO_RE, "sudo"),
        ):
            if pattern.search(line):
                return f"line {number} uses {what}: {bare[:120]}"
        opened = _HEREDOC_OPEN_RE.search(line)
        if opened:
            terminator = opened.group(2)
    return ""


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
    # A second correct solution that reaches the same outcome by a DIFFERENT route. Its only job is
    # to prove the withheld check grades the outcome rather than the method. One solution can never
    # show that: it was written in the same reply as the check and naturally satisfies it.
    alternate_solution: str = ""
    # Workspace files the agent must not modify -- the inputs its answer is derived FROM.
    #
    # `hermesbench` hashes these before and after an episode and disqualifies any run that changed
    # one. Without them the cheapest way to satisfy a check that pins a value is to edit the data
    # until the wrong answer is the right one, and nothing notices: `runner.py` skips the integrity
    # comparison entirely when the tuple is empty. Generated tasks shipped empty, so every one of
    # them was gradeable by tampering.
    protected_paths: tuple[str, ...] = ()


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


def _digest_file(path: Path) -> str | None:
    """Content digest of one file, or None when it is absent -- the shape `digest_paths` uses."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


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


def _copy_workspace(source: Path, root: Path, prefix: str) -> Path:
    """A copy of a built workspace, links included rather than followed.

    `symlinks=True` is load-bearing, not tidiness. These tasks create symlinks on purpose -- the very
    first one this gate accepted turned on whether a config path was linked to the real file -- and a
    setup that leaves a DANGLING link is a perfectly good workspace for a task about repairing it.
    Following links makes `copytree` open the target, and a dangling target raises. That crash took
    down a generation run at 116 accepted tasks: one malformed workspace ended the process rather
    than the attempt.
    """
    target = Path(tempfile.mkdtemp(dir=root, prefix=prefix))
    shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True)
    return target


def _build(candidate: Candidate, root: Path, timeout: int) -> tuple[Path, str]:
    """Build the workspace once, under `set -e`.

    `sh -c` reports the exit status of the LAST command, so without `set -e` "setup exits 0" is a
    much weaker statement than it reads as: a script whose first line fails and whose last line
    succeeds passes. Measured: `cp /nonexistent/file data/` followed by a valid setup was accepted
    9/9. Usually checks 3-7 catch the consequences -- but not when the line that failed was the one
    planting the trap, which yields a task accepted WITHOUT the thing that made it hard, and check
    2 cannot see it either because it fails identically both times.

    Requirement 1b of the synth prompt tells the model this, so a script that needs a command to be
    allowed to fail can say `|| true` rather than being rejected for it.
    """
    workspace = Path(tempfile.mkdtemp(dir=root, prefix=f"{candidate.task_id}-"))
    code, output = _run("set -e\n" + candidate.setup, workspace, timeout=timeout)
    if code != 0:
        raise _Failed("setup_exits_zero", f"setup exited {code}: {output}")
    return workspace, _fingerprint(workspace)


class _Failed(Exception):
    def __init__(self, check: str, detail: str) -> None:
        super().__init__(detail)
        self.check = check
        self.detail = detail


def gate(candidate: Candidate, *, root: Path | None = None, timeout_s: int = STEP_TIMEOUT_S) -> Verdict:
    """Run every check in `ALL_CHECKS`. The first failure stops the rest and names itself.

    Stopping early is deliberate: once setup is broken, every later check is measuring the broken
    setup, and a verdict listing five failures where one caused the other four tells a reader less
    than a verdict naming the one.
    """
    verdict = Verdict(task_id=candidate.task_id, accepted=False)
    scratch = Path(tempfile.mkdtemp(prefix="taskgen-gate-")) if root is None else root
    made_scratch = root is None
    try:
        try:
            # 0. Static, and first because it is the only check that is cheaper than running the
            # thing it judges -- and because by the time a path escape is observable it has already
            # happened.
            for name, script in (
                ("setup", candidate.setup),
                ("the published check", candidate.verify),
                ("the withheld check", candidate.withheld_verify),
                ("the reference solution", candidate.reference_solution),
                ("the alternate solution", candidate.alternate_solution),
                ("the cheat solution", candidate.cheat_solution),
            ):
                escape = _escapes_workspace(script)
                if escape:
                    raise _Failed(
                        "scripts_stay_in_the_workspace",
                        f"{name} reaches outside the scratch directory -- {escape}. Every path must "
                        "be relative to the current directory, which already IS the workspace. An "
                        "absolute path either dies on an unwritable system path or writes into the "
                        "host, and the second is silent",
                    )
            verdict.checks_run.append("scripts_stay_in_the_workspace")

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
            solved = _copy_workspace(workspace, scratch, f"{candidate.task_id}-solved-")
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
            cheated = _copy_workspace(workspace, scratch, f"{candidate.task_id}-cheat-")
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

            # 8b. Protected paths: declared, present, and not touched by a correct solution.
            #
            # The second half is what makes this executable rather than decorative. A task that
            # protects a file its own reference solution rewrites disqualifies every agent that
            # solves it properly -- the same shape as the defect check 9 exists for, arriving from
            # the task's own inputs. Checked against the reference only: the alternate is already
            # required to satisfy the withheld check, and a second full solution run to re-answer a
            # question the reference has answered is cost without evidence.
            if not candidate.protected_paths:
                raise _Failed(
                    "protected_paths_are_sound",
                    "the task declares no protected paths, so an agent may rewrite the very inputs "
                    "its answer is derived from and the published check cannot tell. Name the data "
                    "files the agent must not modify",
                )
            missing = [rel for rel in candidate.protected_paths if not (workspace / rel).is_file()]
            if missing:
                raise _Failed(
                    "protected_paths_are_sound",
                    f"protected path(s) {missing} do not exist after setup, so nothing is protected "
                    "and the runner would record them as deleted by the agent",
                )
            changed = [
                rel for rel in candidate.protected_paths if _digest_file(workspace / rel) != _digest_file(solved / rel)
            ]
            if changed:
                raise _Failed(
                    "protected_paths_are_sound",
                    f"the reference solution modifies protected path(s) {changed}, so every agent "
                    "that solves this task the intended way would be disqualified for tampering",
                )
            verdict.checks_run.append("protected_paths_are_sound")

            # 9. Method-pinning. The first generated task this gate accepted demanded a *symlink*
            # specifically: an agent that set an environment variable or copied the file would have
            # fixed the service for real and still failed, scoring as `overfit` while being correct.
            # Checks 6 and 7 cannot see this, because the reference solution comes from the same
            # reply as the check and uses the same method by construction.
            if candidate.alternate_solution.strip():
                other = _copy_workspace(workspace, scratch, f"{candidate.task_id}-alt-")
                code, output = _run(candidate.alternate_solution, other, timeout=timeout_s)
                if code != 0:
                    raise _Failed(
                        "withheld_accepts_a_different_method",
                        f"the alternate solution did not run ({code}), so nothing was learned about "
                        f"whether the withheld check grades the outcome or the method: {output}",
                    )
                code, output = _run(candidate.withheld_verify, other, timeout=timeout_s)
                if code != 0:
                    raise _Failed(
                        "withheld_accepts_a_different_method",
                        "a second solution that reaches the same outcome by a different route failed "
                        f"the withheld check, so the check grades the METHOD rather than the result. "
                        f"An agent solving this task correctly another way would score as overfit: {output}",
                    )
                verdict.checks_run.append("withheld_accepts_a_different_method")

            verdict.accepted = True
        except _Failed as failure:
            verdict.failed_check = failure.check
            verdict.detail = failure.detail
        except OSError as exc:
            # A generated setup can build a workspace this process cannot copy or read -- a dangling
            # link, a permission bit, a name the filesystem refuses. That is a fact about the task and
            # belongs in the histogram; letting it propagate ends the run instead of the attempt, and
            # a generation run that dies at task 116 of 150 has thrown away the queue behind it.
            verdict.failed_check = "workspace_is_unusable"
            verdict.detail = f"{type(exc).__name__}: {exc}"
    finally:
        if made_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    return verdict


ALL_CHECKS = (
    "scripts_stay_in_the_workspace",
    "setup_exits_zero",
    "setup_is_deterministic",
    "public_fails_untouched",
    "withheld_fails_untouched",
    "reference_solution_runs",
    "public_passes_reference",
    "withheld_passes_reference",
    "checks_disagree_on_a_cheat",
    "protected_paths_are_sound",
    "withheld_accepts_a_different_method",
)

# Not in ALL_CHECKS: that tuple is the ORDER the checks run in, and this is not a check. It is what a
# verdict says when the workspace itself could not be handled -- a dangling link, a permission bit, a
# name the filesystem refuses. A fact about the task, so it belongs in the histogram, but it is not a
# stage anything passes.
WORKSPACE_UNUSABLE = "workspace_is_unusable"

__all__ = ["ALL_CHECKS", "STEP_TIMEOUT_S", "Candidate", "GateError", "Verdict", "gate"]

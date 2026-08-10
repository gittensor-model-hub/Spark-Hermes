"""Does the suite still measure anything? A check that needs no model.

A benchmark can rot in ways its own unit tests never see. A task's setup can stop building
its workspace when a base image changes; a verifier can start passing before the agent has
done anything, because a file it looks for is now created by setup; a withheld check can
drift out of agreement with the published one. None of that shows up as a failing test, and
all of it shows up as a suspiciously good score.

So this runs the parts of the benchmark that do not need an agent, and asserts the one
property that makes a verifier a verifier:

**A published check must fail against a freshly-set-up workspace.** If it passes there, the
task is already solved before the agent starts, and every model scores a point for it. That
is the single most damaging defect a task can have, because it inflates the number in the
direction nobody questions.

**Withheld checks are held to the opposite standard where a solution exists.** A withheld
check that fails on an unsolved workspace is doing its job; one that passes there is
measuring nothing. Both are asserted, since a hidden test nobody has run is a hidden test
nobody knows the state of, and `overfit_rate` depends on it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from hermesbench import BENCH_VERSION
from hermesbench.tasks import Task, load_suite
from hermesbench.verify import setup_task, verify_hidden, verify_task
from hermesbench.withheld import WITHHELD_ROOT_ENV, WITHHELD_SALT_ENV, overlay, unscorable

# Interpreters a grader might invoke. Only these are reported -- see `missing_commands` for why
# the general "find every command" version was abandoned. `python` is the one that has actually
# bitten: the Dockerfile provides the alias, a stock host does not, and 14 of 19 graders already
# resolve it defensively while two did not and scored 0/10 on the first real baseline.
INTERPRETERS = ("python", "python2", "node", "ruby", "perl", "bash", "zsh")


def missing_commands(script: str) -> list[str]:
    """Executables a script invokes that do not exist on this machine.

    This exists because the fresh-workspace assertion below is structurally blind to a
    verifier that cannot run at all. `verify-speedup-claim` read

        test -f winner.txt && python - <<'EOF'

    and on a clean workspace the `test` failed first, so the command exited 1 -- which is
    exactly what this module asserts a verifier must do. The short-circuit satisfied the check
    while the grader was incapable of ever PASSING: the moment an agent created `winner.txt`
    the interpreter lookup ran and exited 127. It scored 0/10 on the first real baseline and
    read as a capability gap in the model.

    Proving a verifier CAN pass would need a known-good solution, and tasks here do not ship
    one. Proving its interpreter exists is weaker and cheap, and it catches the class that
    actually bit: an absent interpreter fails the verifier whatever the workspace contains, so
    no state an agent could reach would reveal it.

    Deliberately narrow: it only reports names in `INTERPRETERS`, not every command it can find.
    The first attempt tried to identify all commands and produced false positives immediately --
    `PYTHONPATH=deps python3 -c "` opens a multi-line double-quoted Python string, so a
    line-oriented reader treats `import`, `major` and `raise` as commands. Parsing shell properly
    is not the goal here, and a lint that cries wolf gets switched off, taking the real finding
    with it.

    Restricting the report to known interpreter names removes that whole class at once, because
    `import` is not an interpreter. It gives up on catching an absent `jq` or `zstd` -- those
    fail loudly and immediately when a verifier runs, unlike an interpreter alias, which fails
    identically to a model that cannot do the task. That is the asymmetry worth spending
    precision on.
    """
    # Drop heredoc bodies before looking for commands.
    lines: list[str] = []
    in_heredoc, terminator = False, ""
    for line in script.splitlines():
        if in_heredoc:
            if line.strip() == terminator:
                in_heredoc = False
            continue
        opener = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", line)
        if opener:
            in_heredoc, terminator = True, opener.group(1)
        lines.append(line)

    # `command -v python` is the RESOLVED form -- it tests for the interpreter rather than
    # invoking it, and flagging it would fail exactly the graders that have been repaired.
    text = re.sub(r"command\s+-v\s+\S+", "", "\n".join(lines))

    missing: list[str] = []
    for name in INTERPRETERS:
        # Command position: line start, or after a shell operator, or after a VAR=value prefix.
        # Followed by whitespace then something, so a bare mention in prose does not match.
        if not re.search(rf"(?:^|[;&|(]\s*|\s)(?:[A-Z_]+=\S*\s+)?{re.escape(name)}\s+\S", text, re.M):
            continue
        if shutil.which(name) is None and name not in missing:
            missing.append(name)
    return missing


def check_task(task: Task, root: Path) -> list[str]:
    """Run setup and the verifiers against an unsolved workspace. Returns problems."""
    problems: list[str] = []
    workspace = root / task.task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Before running anything. A verifier that invokes a command this machine does not have
    # fails for a reason no workspace state can reveal, and the fresh-workspace assertion below
    # would still pass -- which is exactly how an ungradeable verifier reached a live baseline.
    for label, script in (("published", task.verify), ("withheld", task.hidden_verify)):
        if not script.strip():
            continue
        absent = missing_commands(script)
        if absent:
            problems.append(
                f"{label} verifier invokes {absent}, which do not exist here; it can never pass on "
                "this machine, and a task whose grader cannot run scores 0 and reads as a capability "
                'gap in the model. Resolve the interpreter -- PY="$(command -v python3 || command -v '
                'python)" -- or install the tool.'
            )

    setup = setup_task(task, workspace)
    if setup is not None and not setup.passed:
        # Infrastructure breakage, and it disqualifies the task rather than the agent:
        # every episode on it would be scored as a failure the model never caused.
        problems.append(f"setup failed: {(setup.stderr or setup.stdout)[:200]}")
        return problems

    public = verify_task(task, workspace)
    if public.passed:
        problems.append(
            "published verifier PASSES an untouched workspace; the task is solved before the agent "
            "starts and every model scores a free point for it"
        )

    if task.has_hidden_tests:
        hidden = verify_hidden(task, workspace)
        if hidden is not None and hidden.passed:
            problems.append(
                "withheld verifier passes an untouched workspace; it is measuring nothing, and "
                "overfit_rate computed from it would be meaningless"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=BENCH_VERSION, help="bench versions: 'v1', 'v0,v1', or 'all'")
    parser.add_argument("--keep", action="store_true", help="keep the scratch workspaces for inspection")
    args = parser.parse_args(argv)

    tasks = load_suite(args.suite)
    # Attach the withheld checks when a private tree is configured. Without one this is a
    # public checkout, which is a legitimate state and must be reported as such rather than
    # silently checking half the suite.
    tasks = overlay(tasks)
    root = Path(tempfile.mkdtemp(prefix="suitecheck-"))
    failures = 0
    try:
        for task in tasks:
            problems = check_task(task, root)
            marker = "FAIL" if problems else "ok  "
            hidden = " [withheld]" if task.has_hidden_tests else ""
            print(f"{marker} {task.task_id}{hidden}")
            for problem in problems:
                print(f"       {problem}")
            failures += bool(problems)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    runnable = sum(1 for t in tasks if t.has_hidden_tests)
    declared = sum(1 for t in tasks if t.declares_hidden_tests)
    missing = unscorable(tasks)
    print(
        f"\n{len(tasks)} tasks, {declared} declaring withheld checks, {runnable} runnable here, "
        f"{failures} with problems",
        file=sys.stderr,
    )
    if missing:
        # The distinction the split exists for. "0 with withheld checks" and "16 declared,
        # none available" are the same number if you only ask `has_hidden_tests`, and the
        # first reads as a suite that never had an overfit signal rather than one whose
        # signal this checkout cannot compute.
        print(
            f"note: {len(missing)} task(s) declare a withheld check this checkout does not have, so "
            f"overfit_rate is UNAVAILABLE here, not 0.0. Set {WITHHELD_ROOT_ENV} and "
            f"{WITHHELD_SALT_ENV} to score them: {', '.join(missing[:4])}" + (" ..." if len(missing) > 4 else ""),
            file=sys.stderr,
        )
    elif not declared:
        # Genuinely no withheld checks anywhere -- a different and worse state.
        print(
            "warning: no task declares hidden_verify, so overfit_rate is structurally 0.0",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import shutil
import sys
import tempfile
from pathlib import Path

from hermesbench import BENCH_VERSION
from hermesbench.tasks import Task, load_suite
from hermesbench.verify import setup_task, verify_hidden, verify_task
from hermesbench.withheld import WITHHELD_ROOT_ENV, WITHHELD_SALT_ENV, overlay, unscorable


def check_task(task: Task, root: Path) -> list[str]:
    """Run setup and the verifiers against an unsolved workspace. Returns problems."""
    problems: list[str] = []
    workspace = root / task.task_id
    workspace.mkdir(parents=True, exist_ok=True)

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

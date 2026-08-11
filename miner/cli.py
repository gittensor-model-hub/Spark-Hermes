"""What a miner runs before spending anything.

    python -m miner init  --dir ./my-submission --skill step-budget
    python -m miner check --dir ./my-submission

`check` is the whole point of this module: it answers "will the validator take this, and what
will the model actually see" without contacting a model, touching a GPU, or opening a pull
request. Every refusal it prints is one a miner would otherwise learn from CI, after paying for
a run.

## It calls the validator's own code, not a copy of it

`hermes.miner_contract.MinerContract.check` decides which files are allowed;
`hermes.profile.assemble` is the function the runner calls before it contacts the model;
`hermes.profile.compose_system_prompt` is what folds a submission into the prompt. This module
calls all three. A second implementation that agreed with them today would drift, and the day it
drifted "it passed locally" would mean nothing -- which is worse than having no local check,
because a miner would trust it.

## Why it reports sizes

Measured on the 190-episode baseline: input is 94.9% of all tokens, and the median episode takes
34 turns. A submission's prose sits at the head of the prompt on every one of those turns.

That is *not* mainly a token bill -- with prefix caching on, the measured hit rate is 80.8%, so
the system prompt is served from cache after the first turn. It is a **context** cost, and that
is the tighter constraint: the window is 32,768 tokens, the model re-reads this text before every
decision, and prose that earns its place in one turn is prose the model must re-read 33 more
times. So the size is reported as what it is, without the arithmetic that would overstate it.

## What it cannot tell you

Whether the submission is any good. `check` proves a submission is *admissible*. The only thing
that establishes it *helps* is a paired run against a baseline, and the first submission written
for this repository -- careful, plausible, aimed squarely at the three `step_budget_exhausted`
challenges -- increased median tokens by 58.6% and took the pass rate from 1/3 to 0/3. It passed
`check` cleanly. That gap is the reason `evaluate` exists rather than this being the last step.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from hermes.acceptance import MIN_ATTEMPTS
from hermes.miner_contract import ContractError, Violation
from hermes.miner_contract import load as load_contract

SKILL_TEMPLATE = """---
name: {skill}
description: One line. Hermes reads this to decide whether to load the skill at all.
---

# {title}

State the situation this applies to in one sentence, then the steps.

1. **Do the cheap thing that removes uncertainty first.** Reading is cheaper than being wrong.
2. **Decide before acting.** If the plan needs a fact you do not have, get it in the same pass.
3. **Verify with the task's own check, and read the output.** A command that exited 0 is not
   evidence the change was correct; the check is.

Delete every line above that you did not mean. The model re-reads this before every decision, so
a sentence that does not change what it does is a sentence competing with the ones that do.
"""

SOUL_TEMPLATE = """# Operating identity

Two or three sentences on how this agent works. Not a persona -- a policy. What it does before
editing, what it does when a command fails, and when it stops.

Replace this text. A default SOUL.md that survives into a submission is a submission that has not
been written yet.
"""


def submission_paths(root: Path) -> list[str]:
    """Every file in the submission, as posix paths relative to its root.

    Directories are not listed and symlinks are not followed. A symlink is how a submission that
    contains only `.md` files comes to contain whatever it points at, and the contract checks
    names rather than targets.
    """
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            found.append(path.relative_to(root).as_posix())
            continue
        if path.is_file():
            found.append(path.relative_to(root).as_posix())
    return found


def symlinks_in(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_symlink()]


def check_submission(root: Path) -> tuple[list[str], list[Violation], list[str]]:
    """Returns (paths, contract violations, extra problems this layer found)."""
    problems: list[str] = []
    if not root.is_dir():
        return [], [], [f"{root} is not a directory"]

    paths = submission_paths(root)
    if not paths:
        problems.append(
            f"{root} contains no files. An empty submission is accepted by every path-based check "
            "there is, and would run as the unmodified baseline while looking like an entry."
        )

    for link in symlinks_in(root):
        # Reported here rather than left to the contract, which matches names. A symlink named
        # `notes.md` satisfies every extension and allowlist rule while resolving to anything.
        problems.append(f"{link}: is a symlink. A submission is files, not references to files.")

    contract = load_contract()
    return paths, contract.check(paths), problems


def _size_of(path: Path) -> int:
    """Bytes, without following a symlink.

    `stat()` resolves the link, so a submission containing `notes.md -> /etc/passwd` reported the
    size of the target -- which is both wrong and a read of a file the tool has no business
    touching. `lstat` describes the link itself, which is the thing that was submitted.
    """
    try:
        return path.lstat().st_size
    except OSError:
        return 0


def describe(root: Path, *, base_prompt: str = "") -> int:
    """Print the verdict. Returns a process exit code."""
    paths, violations, problems = check_submission(root)

    print(f"submission {root}")
    for path in paths:
        print(f"  {_size_of(root / path):>7,} B  {path}")
    print(f"  {sum(_size_of(root / p) for p in paths):>7,} B  total")

    if problems or violations:
        print("\nREFUSED")
        for problem in problems:
            print(f"  - {problem}")
        for violation in violations:
            print(f"  - {violation}")
        print(
            "\nEvery reason is listed rather than the first, because a miner who learns one problem "
            "per resubmission stops resubmitting."
        )
        return 1

    # The validator's own acceptance path. Reached only once the cheap checks pass, because its
    # failure modes are less legible and a contract violation should be reported as one.
    accepted, why, composed = _assemble(root, base_prompt)
    if not accepted:
        print(f"\nREFUSED by hermes.profile.assemble\n  - {why}")
        return 1

    print("\nACCEPTED by the contract and by hermes.profile.assemble")
    print(f"  composed system prompt: {len(composed):,} characters")
    print(
        "  This sits at the head of the prompt on every turn. With prefix caching on, the measured\n"
        "  hit rate is 80.8%, so it is mostly not a token bill -- it is context. The window is\n"
        "  32,768 tokens and the median episode takes 34 turns, so the model re-reads this text\n"
        "  before every decision it makes."
    )
    print(
        "\nThis says the submission is admissible, not that it helps. The first submission written\n"
        "for this repository passed this check cleanly and then increased median tokens by 58.6%\n"
        "while taking the pass rate from 1/3 to 0/3. Run a paired evaluation before submitting."
    )
    return 0


def _assemble(root: Path, base_prompt: str) -> tuple[bool, str, str]:
    """Run the validator's `assemble` and `compose_system_prompt`. Returns (ok, why, composed)."""
    from hermes.base_model import load as load_pin
    from hermes.profile import PINNED_CONFIG_KEYS, ProfileError, assemble, compose_system_prompt

    try:
        pin = load_pin()
        assemble(
            agent_repository="NousResearch/hermes-agent",
            agent_commit="0" * 40,
            model_repository=pin.repository,
            model_revision=pin.revision,
            config=dict.fromkeys(PINNED_CONFIG_KEYS, "pinned"),
            miner_dir=root,
            validator_executed=True,
        )
        return True, "", compose_system_prompt(base_prompt, root)
    except (ProfileError, ContractError) as exc:
        return False, str(exc), ""


def scaffold(root: Path, *, skill: str) -> list[Path]:
    """Write a minimal admissible submission. Refuses to overwrite."""
    soul = root / "SOUL.md"
    skill_file = root / "skills" / skill / "SKILL.md"
    existing = [p for p in (soul, skill_file) if p.exists()]
    if existing:
        raise FileExistsError(
            f"{', '.join(str(p) for p in existing)} already exists. Refusing to overwrite: a "
            "scaffold that clobbers a submission is a scaffold that loses work."
        )
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    soul.write_text(SOUL_TEMPLATE, encoding="utf-8")
    skill_file.write_text(
        SKILL_TEMPLATE.format(skill=skill, title=skill.replace("-", " ").capitalize()),
        encoding="utf-8",
    )
    return [soul, skill_file]


def _evaluate(args: Any) -> int:
    """Validate first, then run both arms, then judge with the validator's own gate."""
    from hermesbench import runner
    from miner.evaluate import arm_from_log, compare, render, runner_argv

    if not args.task or not args.model:
        print("miner: evaluate needs --task and --model", file=sys.stderr)
        return 2

    # Before any GPU time. A submission the validator would refuse is a submission whose
    # measurement is worthless, and the refusal costs nothing to find.
    code = describe(args.root)
    if code != 0:
        print("\nnot evaluating a submission the validator would refuse", file=sys.stderr)
        return code

    out = args.workspace_root
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for label, miner_dir in (("control", None), ("candidate", args.root)):
        log = out / f"{label}.jsonl"
        if log.exists():
            log.unlink()
        print(f"\n--- {label} arm: {args.repeats} attempt(s) on {args.task} ---", flush=True)
        # The runner prints a full suite-metrics JSON block per invocation. Two of those ahead of
        # the verdict buried the one thing a miner opened this tool for, so the arms' stdout is
        # captured and only surfaced when an arm fails -- where it is the diagnosis rather than
        # noise. Observed on the first live end-to-end run: ~25 lines of JSON before the summary.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            rc = runner.main(
                runner_argv(
                    task_id=args.task,
                    base_url=args.base_url,
                    model=args.model,
                    api_key_env=args.api_key_env,
                    workspace_root=out / f"ws-{label}",
                    episodes_out=log,
                    repeats=args.repeats,
                    miner_dir=miner_dir,
                    allow_unsandboxed=args.allow_unsandboxed,
                )
            )
        arm_output = captured.getvalue()
        if args.verbose and arm_output:
            print(arm_output, end="")
        if rc != 0:
            # Not swallowed. An arm that failed is the whole answer, so its output goes out in full.
            print(arm_output, end="", file=sys.stderr)
            print(f"miner: the {label} arm exited {rc}", file=sys.stderr)
            return rc
        summary = out / f"{label}-suite.json"
        summary.write_text(arm_output, encoding="utf-8")
        print(f"  done; runner output in {summary}")
        arms.append(arm_from_log(log, label=label))

    report = compare(*arms)
    print()
    print(render(report, task_id=args.task))
    if args.report:
        args.report.write_text(json.dumps(report.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0 if report.decision.accepted else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m miner",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("action", choices=["check", "init", "evaluate", "search"])
    parser.add_argument("--dir", type=Path, required=True, dest="root", help="the submission directory")
    parser.add_argument("--skill", default="my-strategy", help="skill directory name (init only)")
    parser.add_argument("--task", default="", help="the task id to evaluate on (evaluate only)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="", help="the served model name (evaluate only)")
    parser.add_argument("--api-key-env", default="NONE")
    parser.add_argument(
        "--repeats",
        type=int,
        default=MIN_ATTEMPTS,
        help="paired attempts per arm. Defaults to MIN_ATTEMPTS: one attempt tells you almost "
        "nothing and feels like it tells you everything",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("var/miner-eval"))
    parser.add_argument("--report", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print each arm's full runner output instead of saving it beside the report",
    )
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help="required to run model-authored shell commands; pass only inside a container, VM or disposable machine",
    )
    parser.add_argument(
        "--base-prompt",
        type=Path,
        default=None,
        help="a file holding the harness base prompt, to show the composed result at its real size",
    )
    args = parser.parse_args(argv)

    if args.action == "search":
        from miner.search import main as search_main

        forwarded = [
            "--dir",
            str(args.root),
            "--task",
            args.task,
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--api-key-env",
            args.api_key_env,
            "--repeats",
            str(args.repeats),
        ]
        if args.allow_unsandboxed:
            forwarded.append("--allow-unsandboxed")
        return search_main(forwarded)

    if args.action == "evaluate":
        return _evaluate(args)

    if args.action == "init":
        try:
            written = scaffold(args.root, skill=args.skill)
        except FileExistsError as exc:
            print(f"miner: {exc}", file=sys.stderr)
            return 2
        for path in written:
            print(f"wrote {path}")
        print(f"\nnow run: python -m miner check --dir {args.root}")
        return 0

    base = args.base_prompt.read_text(encoding="utf-8") if args.base_prompt else ""
    try:
        return describe(args.root, base_prompt=base)
    except ContractError as exc:
        print(f"miner: {exc}", file=sys.stderr)
        return 2


__all__ = ["check_submission", "describe", "main", "scaffold", "submission_paths", "symlinks_in"]


if __name__ == "__main__":
    raise SystemExit(main())

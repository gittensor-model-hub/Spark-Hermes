"""Run a submission against the baseline, paired, and report what the gate would say.

    python -m miner evaluate --dir ./my-submission --task tc-log-rotation-order \\
        --base-url http://127.0.0.1:8000/v1 --model qwen3.6-27b --repeats 10

`check` proves a submission is admissible. This is the only thing that establishes whether it
*helps*, and the gap between the two is not academic: the first submission written for this
repository passed `check` cleanly, then increased median tokens by 58.6% and took the pass rate
from 1/3 to 0/3.

## Paired, and re-measured locally

Both arms run here, now, on this machine: the control with no submission, the candidate with it.
Comparing a local candidate against the baseline *published in the packet* would fold every
difference between the miner's box and the validator's into the margin, and hardware variance is
the one thing pairing removes for free.

The consequence is stated in the output rather than buried: **a local win is evidence, not
acceptance.** The validator will re-run the baseline on its own hardware and compare against that,
so what transfers is the *effect*, not the numbers.

## It calls the runner's entry point rather than rebuilding it

`hermesbench.runner.main` is invoked with the same argv a validator uses, twice. Reassembling the
runner's setup here -- tool schemas, dialect, system prompt composition, executor -- would be a
second implementation of the thing being measured, and the dialect default alone was wrong for
long enough to make an entire run look like a capability failure. The submission is passed as
`--miner-dir`, which is the same flag and the same code path the validator's execution step uses.

## Why the default is not one repeat

One attempt tells you almost nothing and feels like it tells you everything. Measured: a 20% win
sitting exactly on the bar is unreachable at any sample size, and a 30% win at 30% run-to-run
spread needs roughly 50 paired repeats before the bootstrap interval clears. So the default is
`MIN_ATTEMPTS`, and when the interval does not clear the report prints `repeats_needed` -- a
number, because "run more repeats" leaves a miner guessing how much GPU time to buy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes.acceptance import MIN_ATTEMPTS, Arm, Decision, decide, reduction_interval, repeats_needed


class EvaluateError(RuntimeError):
    """The evaluation cannot be run or cannot be judged."""


@dataclass(frozen=True)
class ArmResult:
    """One side of the pair, as measured on this machine."""

    label: str
    arm: Arm
    steps: tuple[int, ...]
    dialects: tuple[str, ...]

    @property
    def median_tokens(self) -> float:
        from statistics import median

        return median(self.arm.tokens)


@dataclass(frozen=True)
class Report:
    control: ArmResult
    candidate: ArmResult
    decision: Decision
    interval: tuple[float, float]
    repeats_to_settle: int | None

    def to_record(self) -> dict[str, Any]:
        return {
            "control": {
                "passes": self.control.arm.passes,
                "attempts": self.control.arm.attempts,
                "median_tokens": self.control.median_tokens,
                "tokens": list(self.control.arm.tokens),
            },
            "candidate": {
                "passes": self.candidate.arm.passes,
                "attempts": self.candidate.arm.attempts,
                "median_tokens": self.candidate.median_tokens,
                "tokens": list(self.candidate.arm.tokens),
            },
            "decision": self.decision.to_record(),
            "reduction_interval": list(self.interval),
            "repeats_to_settle": self.repeats_to_settle,
            # Said in the record as well as on stdout. A report copied into a pull request without
            # this line is a claim about the validator's hardware that nobody measured.
            "a_local_win_is_evidence_not_acceptance": True,
        }


def arm_from_log(path: Path, *, label: str) -> ArmResult:
    """Build an `Arm` from one `--episodes-out` log.

    Reads through `hermes.challenge.episode_metrics_of`, which normalises the sink's nested shape.
    Reading the metric names off the top level is a real bug that shipped: every lookup missed,
    every default applied, and ten healthy episodes became ten zero-token failures without raising.
    """
    from hermes.challenge import episode_metrics_of
    from hermesbench.sink import read_episodes

    rows = [episode_metrics_of(row) for row in read_episodes(path)]
    if not rows:
        raise EvaluateError(f"{path} holds no episodes; the {label} arm produced nothing to judge")

    tokens = tuple(int(r.get("tokens_used") or 0) for r in rows)
    if not all(tokens):
        raise EvaluateError(
            f"the {label} arm reported a zero-token episode. That is not a cheap run -- it is a run "
            "that did not happen, and averaging it in would make the arm look free."
        )
    return ArmResult(
        label=label,
        arm=Arm(
            passes=sum(1 for r in rows if r.get("public_passed")),
            attempts=len(rows),
            tokens=tokens,
            tool_calls=tuple(int(r.get("tool_calls") or 0) for r in rows),
        ),
        steps=tuple(int(r.get("steps") or 0) for r in rows),
        dialects=tuple(str(r.get("dialect") or "") for r in rows),
    )


def compare(control: ArmResult, candidate: ArmResult) -> Report:
    """Judge the pair with the gate the validator uses. No model contact, no I/O."""
    if control.arm.attempts != candidate.arm.attempts:
        # Unequal arms are not a pairing. The interval would be computed over two sample sizes and
        # the smaller one silently dominates its width.
        raise EvaluateError(
            f"the arms are not paired: control ran {control.arm.attempts} attempts and candidate ran "
            f"{candidate.arm.attempts}. Re-run both with the same --repeats."
        )
    mismatched = {d for d in (*control.dialects, *candidate.dialects) if d}
    if len(mismatched) > 1:
        raise EvaluateError(
            f"the arms ran different wire dialects {sorted(mismatched)}. A dialect the model does not "
            "speak produces prose answers with zero tool calls and a clean protocol report, so this "
            "comparison would be measuring the harness rather than the submission."
        )
    return Report(
        control=control,
        candidate=candidate,
        decision=decide(candidate=candidate.arm, baseline=control.arm),
        interval=reduction_interval(control.arm.tokens, candidate.arm.tokens),
        repeats_to_settle=repeats_needed(control.arm.tokens, candidate.arm.tokens),
    )


def runner_argv(
    *,
    task_id: str,
    base_url: str,
    model: str,
    api_key_env: str,
    workspace_root: Path,
    episodes_out: Path,
    repeats: int,
    miner_dir: Path | None,
    allow_unsandboxed: bool,
    dialect: str = "",
    keep_trajectories: bool = True,
) -> list[str]:
    """The argv a validator would use.

    `dialect` is normally empty and the pin supplies it, which is right when the served model is
    the pinned one. It is passable because a validator can deliberately serve something else --
    during a base-model migration, or to compare two -- and the runner then instructs a wire format
    the model does not speak. That is not a loud failure: it shows up as malformed turns, or as a
    model that never calls a tool, both of which read as the model being bad.

    `keep_trajectories` defaults to TRUE here, unlike the runner, and the difference is deliberate.
    The runner serves ad-hoc benchmarking where a transcript is a cost nobody asked for. A judged
    round is the input to `validator.aggregate`, which builds every SFT row and preference pair
    from trajectories -- so a judged round without them yields no training data at all, and
    `aggregate` refuses with "no episode carries a trajectory" rather than writing an empty file.
    """
    argv = [
        "--base-url",
        base_url,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
        "--workspace-root",
        str(workspace_root),
        "--task-ids",
        task_id,
        "--suite",
        "all",
        "--repeats",
        str(repeats),
        "--episodes-out",
        str(episodes_out),
    ]
    if dialect:
        argv += ["--dialect", dialect]
    if keep_trajectories:
        argv += ["--keep-trajectories"]
    if miner_dir is not None:
        argv += ["--miner-dir", str(miner_dir)]
    if allow_unsandboxed:
        argv += ["--allow-unsandboxed"]
    return argv


def render(report: Report, *, task_id: str) -> str:
    """The human-facing summary. Separated from `compare` so the judgement is testable."""
    control, candidate = report.control, report.candidate
    lines = [
        f"task {task_id}",
        f"  control    {control.arm.passes}/{control.arm.attempts} passed   median {control.median_tokens:,.0f} tokens",
        f"  candidate  {candidate.arm.passes}/{candidate.arm.attempts} passed   "
        f"median {candidate.median_tokens:,.0f} tokens",
    ]
    base, cand = control.median_tokens, candidate.median_tokens
    if base > 0:
        change = (base - cand) / base
        word = "reduction" if change >= 0 else "INCREASE"
        lines.append(f"  token {word}: {abs(change):.1%}")
    low, high = report.interval
    lines.append(f"  95% interval on the reduction: [{low:.1%}, {high:.1%}]")
    if low > 0:
        lines.append("    the whole interval is above zero: the improvement is not noise")
    elif high < 0:
        lines.append("    the whole interval is below zero: this is confidently WORSE, not noise")
    else:
        lines.append("    the interval straddles zero: this evidence cannot tell the two apart")

    lines.append("")
    lines.append("VERDICT: " + ("would be ACCEPTED" if report.decision.accepted else "would be REFUSED"))
    for reason in report.decision.reasons:
        lines.append(f"  - {reason}")
    if report.repeats_to_settle is not None:
        lines.append(f"  about {report.repeats_to_settle} paired repeats would settle evidence like this")

    lines += [
        "",
        "A local win is evidence, not acceptance. Both arms ran on this machine, which is what makes",
        "the comparison fair -- pairing removes hardware variance for free. The validator will",
        "re-measure the baseline on its own hardware and compare against that, so what transfers is",
        "the effect, not these numbers.",
    ]
    return "\n".join(lines)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "MIN_ATTEMPTS",
    "ArmResult",
    "EvaluateError",
    "Report",
    "arm_from_log",
    "compare",
    "load_report",
    "render",
    "runner_argv",
]

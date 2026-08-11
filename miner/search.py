"""Search the surface space by ablation, and refuse to be fooled by the search itself.

    python -m miner search --dir ./my-surface --task tc-log-rotation-order \\
        --base-url http://127.0.0.1:8000/v1 --model qwen3.6-27b --repeats 10

A surface is a small pile of rules. This takes one apart, measures the pieces, and reports which
of them carry the result -- then re-measures the leader on episodes it has never seen, because the
act of picking a winner is itself a source of false wins.

## Why picking the best of k is a trap, and what is done about it

`hermesbench.run_suite` already states the shape of this for repeats: "keeping only the best would
turn the pass rate into a best-of-k order statistic -- `1-(1-p)^k`, which converges to 1.0 for any
p above zero". A search over surfaces is the same hazard one level up. Evaluate twenty candidates
at ten episodes each, keep the best, and its apparent margin is biased upward by however far the
luckiest of twenty draws lands above the mean -- with no candidate having improved anything.

The screening numbers are therefore reported as **screening** and never as a result. The leader is
re-run on a fresh sample and the confirmation is what a miner is told to believe. A margin that
survives an independent measurement is evidence; a margin that only ever appeared in the round
that selected it is the selection.

`selection_pressure` is reported with every search: the number of candidates the winner was chosen
from. One candidate is a measurement; twenty is a tournament, and the reader needs to know which
they are looking at.

## Why ablation rather than random search

At ten episodes a task's run-to-run spread swamps most differences -- measured at 23.5% on
`tc-log-rotation-order`, and `repeats_needed` puts a 30% win at 30% spread near fifty paired
repeats. Random search spends that budget on candidates that differ in ways nobody can attribute.

Leave-one-out answers a question the sample size can support: **which rule is carrying the
result**. Dropping one rule at a time from a surface that works isolates each rule's contribution
against the same background, and a rule whose removal changes nothing is a rule costing context
for free -- which is worth knowing even when no candidate wins.

## It never optimises against the withheld check

Candidates are scored on what a miner can see. The withheld verifier exists precisely because a
search that could read it would fit it, and `overfit_rate` would then measure nothing. This is a
tool for a miner; the validator's withheld half stays where it is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hermes.acceptance import Arm, reduction_interval

# A rule block in a SKILL.md: a numbered item, from its number to the next one or the end.
RULE_RE = re.compile(r"^\d+\.\s+.*?(?=^\d+\.\s|\Z)", re.M | re.S)


class SearchError(RuntimeError):
    """A search cannot be run or cannot be believed."""


@dataclass(frozen=True)
class Candidate:
    """One surface variant, and what was done to produce it."""

    name: str
    files: dict[str, str]
    dropped: tuple[str, ...] = ()

    def write(self, root: Path) -> Path:
        target = root / self.name
        for rel, text in self.files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return target


@dataclass
class Measured:
    candidate: Candidate
    arm: Arm
    episodes: int

    @property
    def median_tokens(self) -> float:
        from statistics import median

        return median(self.arm.tokens)

    @property
    def pass_rate(self) -> float:
        return self.arm.passes / self.arm.attempts


@dataclass
class SearchResult:
    baseline: Measured
    screened: list[Measured]
    leader: Measured | None
    confirmation: Measured | None
    confirmed_interval: tuple[float, float] | None
    episodes_spent: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def selection_pressure(self) -> int:
        """How many candidates the leader was chosen from. One is a measurement; twenty is a
        tournament, and the two deserve different amounts of belief."""
        return len(self.screened)

    def to_record(self) -> dict[str, Any]:
        return {
            "selection_pressure": self.selection_pressure,
            "episodes_spent": self.episodes_spent,
            "baseline": {"pass_rate": self.baseline.pass_rate, "median_tokens": self.baseline.median_tokens},
            "screened": [
                {
                    "name": m.candidate.name,
                    "dropped": list(m.candidate.dropped),
                    "pass_rate": m.pass_rate,
                    "median_tokens": m.median_tokens,
                }
                for m in self.screened
            ],
            "leader": self.leader.candidate.name if self.leader else None,
            "confirmation": (
                {"pass_rate": self.confirmation.pass_rate, "median_tokens": self.confirmation.median_tokens}
                if self.confirmation
                else None
            ),
            "confirmed_interval": list(self.confirmed_interval) if self.confirmed_interval else None,
            # Stated in the record because a screening table pasted somewhere without it reads as a
            # set of results rather than as the round that chose one.
            "screening_numbers_are_biased_by_selection": True,
            "notes": list(self.notes),
        }


def split_rules(skill_text: str) -> list[str]:
    """The numbered rule blocks of a SKILL.md, in order."""
    return [m.group(0).rstrip() + "\n" for m in RULE_RE.finditer(skill_text)]


def read_surface(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def ablations(files: dict[str, str]) -> list[Candidate]:
    """The full surface, then one candidate per rule with that rule removed.

    Leave-one-out rather than every subset. Ablating one rule at a time keeps the background fixed
    so a difference is attributable, and 1+n candidates is a budget a miner can actually afford --
    every subset of eight rules is 256 runs, which at ten episodes each is more GPU time than the
    baseline that produced the challenge.
    """
    skills = sorted(k for k in files if k.endswith("SKILL.md"))
    if not skills:
        raise SearchError("no SKILL.md in the surface; there are no rules to ablate")

    out = [Candidate(name="full", files=dict(files))]
    for skill in skills:
        rules = split_rules(files[skill])
        if len(rules) < 2:
            # Removing the only rule leaves a skill that says nothing, which is not an ablation of
            # a rule -- it is the surface without that skill, and that is a different experiment.
            continue
        for index, rule in enumerate(rules):
            reduced = files[skill].replace(rule, "", 1)
            first_line = rule.strip().splitlines()[0][:60]
            out.append(
                Candidate(
                    name=f"drop-{index + 1}",
                    files={**files, skill: reduced},
                    dropped=(f"{skill}:{first_line}",),
                )
            )
    return out


def leader_of(measured: list[Measured], baseline: Measured) -> Measured | None:
    """The candidate to re-measure: highest pass rate, then fewest tokens.

    Correctness first and lexicographically, mirroring `hermes.acceptance.decide` -- a candidate
    that is cheaper and wrong is not a candidate, so a combined score that could trade one for the
    other would rank something the gate will refuse outright.

    `None` when nothing beat the baseline. A search that always names a winner is a search that
    reports the luckiest draw when there was nothing to find.
    """
    better = [m for m in measured if (m.pass_rate, -m.median_tokens) > (baseline.pass_rate, -baseline.median_tokens)]
    if not better:
        return None
    return max(better, key=lambda m: (m.pass_rate, -m.median_tokens))


def search(
    *,
    surface: Path,
    run: Callable[[Candidate, Path], Arm],
    workspace: Path,
    repeats: int,
    baseline_arm: Arm | None = None,
    confirm: bool = True,
    max_candidates: int | None = None,
) -> SearchResult:
    """Screen the ablations, then re-measure the leader on fresh episodes.

    `run` is injected: the search logic, the selection handling and the confirmation step are the
    parts most likely to be wrong and none of them needs a served model.
    """
    files = read_surface(surface)
    if not files:
        raise SearchError(f"{surface} is empty")

    candidates = ablations(files)
    notes: list[str] = []
    if max_candidates is not None and len(candidates) > max_candidates:
        # Said out loud. A search that silently truncates its own space reports "the best of the
        # ablations" while having measured some of them.
        notes.append(
            f"capped at {max_candidates} of {len(candidates)} candidates; "
            f"{len(candidates) - max_candidates} ablation(s) were not measured"
        )
        candidates = candidates[:max_candidates]

    workspace.mkdir(parents=True, exist_ok=True)
    spent = 0
    screened: list[Measured] = []
    for candidate in candidates:
        arm = run(candidate, workspace / "screen")
        spent += arm.attempts
        screened.append(Measured(candidate, arm, arm.attempts))

    base = next((m for m in screened if m.candidate.name == "full"), None)
    if baseline_arm is not None:
        base = Measured(Candidate(name="baseline", files={}), baseline_arm, baseline_arm.attempts)
    if base is None:
        raise SearchError("no baseline to compare against")

    contenders = [m for m in screened if m.candidate.name != "full"]
    leader = leader_of(contenders, base)
    confirmation: Measured | None = None
    interval: tuple[float, float] | None = None

    if leader is not None and confirm:
        # Fresh episodes. The screening number is the maximum of many draws and is biased upward
        # by construction; this one was not selected on.
        arm = run(leader.candidate, workspace / "confirm")
        spent += arm.attempts
        confirmation = Measured(leader.candidate, arm, arm.attempts)
        interval = reduction_interval(base.arm.tokens, arm.tokens)
    elif leader is None:
        notes.append("no ablation beat the full surface; there is nothing to confirm")

    return SearchResult(
        baseline=base,
        screened=screened,
        leader=leader,
        confirmation=confirmation,
        confirmed_interval=interval,
        episodes_spent=spent,
        notes=notes,
    )


def render(result: SearchResult, *, repeats: int) -> str:
    lines = [
        f"screened {result.selection_pressure} candidate(s) at {repeats} episode(s) each "
        f"-- {result.episodes_spent} episodes total",
        "",
        f"  {'candidate':<14} {'pass':>6}  {'median tokens':>14}   dropped",
        f"  {'full':<14} {result.baseline.pass_rate:>5.0%}  {result.baseline.median_tokens:>14,.0f}",
    ]
    for m in sorted(result.screened, key=lambda x: (-x.pass_rate, x.median_tokens)):
        if m.candidate.name == "full":
            continue
        dropped = m.candidate.dropped[0].split(":", 1)[-1] if m.candidate.dropped else ""
        lines.append(f"  {m.candidate.name:<14} {m.pass_rate:>5.0%}  {m.median_tokens:>14,.0f}   {dropped}")

    lines.append("")
    if result.leader is None:
        lines.append("No ablation beat the full surface. Every rule is either helping or harmless.")
    elif result.confirmation is None:
        lines.append(f"Leader: {result.leader.candidate.name} (unconfirmed -- screening only)")
    else:
        conf = result.confirmation
        lines.append(f"Leader: {result.leader.candidate.name}")
        lines.append(f"  screening     {result.leader.pass_rate:>5.0%}  {result.leader.median_tokens:>12,.0f} tokens")
        lines.append(f"  confirmation  {conf.pass_rate:>5.0%}  {conf.median_tokens:>12,.0f} tokens   (fresh episodes)")
        if result.confirmed_interval:
            low, high = result.confirmed_interval
            lines.append(f"  95% interval on the reduction: [{low:.1%}, {high:.1%}]")
            if low > 0:
                lines.append("    above zero on a sample it was not selected on: this survived the search")
            else:
                lines.append("    straddles zero: the screening lead did not survive an independent measurement")

    for note in result.notes:
        lines.append(f"  note: {note}")
    lines += [
        "",
        f"The screening column is biased: the leader is the best of {result.selection_pressure} draws, so its",
        "apparent margin includes however far the luckiest landed above the mean. Believe the",
        "confirmation row. A margin that only ever appeared in the round that selected it is the",
        "selection, not the surface.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", dest="surface", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="NONE")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workspace", type=Path, default=Path("var/search"))
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--no-confirm", action="store_true", help="screen only; the leader stays unconfirmed")
    parser.add_argument("--allow-unsandboxed", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    from hermesbench import runner
    from miner.evaluate import arm_from_log
    from miner.evaluate import runner_argv as build_argv

    def run(candidate: Candidate, where: Path) -> Arm:
        root = candidate.write(where)
        log = where / f"{candidate.name}.jsonl"
        if log.exists():
            log.unlink()
        print(f"  running {candidate.name} ({args.repeats} episodes)...", flush=True)
        code = runner.main(
            build_argv(
                task_id=args.task,
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
                workspace_root=where / f"ws-{candidate.name}",
                episodes_out=log,
                repeats=args.repeats,
                miner_dir=root,
                allow_unsandboxed=args.allow_unsandboxed,
            )
        )
        if code != 0:
            raise SearchError(f"the runner exited {code} on {candidate.name}")
        return arm_from_log(log, label=candidate.name).arm

    try:
        result = search(
            surface=args.surface,
            run=run,
            workspace=args.workspace,
            repeats=args.repeats,
            confirm=not args.no_confirm,
            max_candidates=args.max_candidates,
        )
    except SearchError as exc:
        print(f"miner.search: {exc}", file=sys.stderr)
        return 2

    print()
    print(render(result, repeats=args.repeats))
    if args.report:
        args.report.write_text(json.dumps(result.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


__all__ = [
    "Candidate",
    "Measured",
    "SearchError",
    "SearchResult",
    "ablations",
    "leader_of",
    "read_surface",
    "render",
    "search",
    "split_rules",
]


if __name__ == "__main__":
    raise SystemExit(main())

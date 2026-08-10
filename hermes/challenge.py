"""Turning a baseline failure into something a miner can compete on.

A challenge is what the flywheel starts from: the frozen model attempted a task and failed,
looped, or succeeded far outside its resource envelope. Until something packages that, a
baseline run is a JSON file somebody reads by hand and the market has no input.

Three properties, and each exists because its absence has a specific cost.

**Confirmed over repeats, never opened on one run.** A single failure is often the sampling
noise of a stochastic decoder, and a challenge opened on it sends every miner in the subnet
to work on a task the baseline solves four times in five. `Baseline.confirmed` reports the
Wilson bound on the true pass rate rather than a count, because "failed once" and "fails
reliably" are the same integer and very different facts. The bound is the same arithmetic
`hermes.acceptance` uses for the other direction: at one attempt, observed 0/1 leaves the
true pass rate anywhere up to 79%.

**Classified from the trace, not asserted.** The failure class decides what a miner
optimises, so a mislabelled challenge wastes the whole round. Every class here is a
predicate over `EpisodeMetrics` and the step sequence -- nothing is passed in by a caller
who already believes the answer.

**Carries the commitment, never the withheld check.** A challenge is published to miners.
The withheld check is the only reason `overfit_rate` measures anything, so the packet
carries `hidden_verify_commitment` from the task and the body stays in the private tree. A
challenge that shipped the check would hand every miner the answer key on the way in.

## What a challenge deliberately does not contain

No acceptance thresholds. `MIN_TOKEN_REDUCTION` and the reduction interval in
`hermes.acceptance` are uncalibrated until a real baseline exists -- the run that produces
challenges is the same run that measures the spread. Baking a threshold into the packet
would freeze a guess at exactly the moment the data to replace it arrives. The packet
records the observed resource envelope and the gate reads the spread from it.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from hermes.harness import digest_mapping

# Why a task became a challenge. Ordered by what a miner would work on first: a task the
# baseline cannot do at all is worth more than one it does expensively.
PUBLIC_VERIFY_FAILED = "public_verify_failed"
HIDDEN_VERIFY_FAILED = "hidden_verify_failed"
OVERFIT = "overfit"
MALFORMED_PROTOCOL = "malformed_protocol"
NO_PROGRESS_LOOP = "no_progress_loop"
STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
SETUP_FAILED = "setup_failed"
EXPENSIVE_SUCCESS = "expensive_success"

# Not a challenge. Named so the classifier can return something rather than None, and so a
# caller cannot mistake "nothing wrong" for "unclassified".
HEALTHY = "healthy"


class ChallengeError(ValueError):
    """A challenge cannot be built from what was supplied."""


@dataclass(frozen=True)
class Attempt:
    """One baseline episode, reduced to what a challenge needs.

    A projection of `EpisodeMetrics` rather than the thing itself, so packaging does not
    drag the whole bench into anything that reads a challenge.
    """

    public_passed: bool
    hidden_passed: bool | None
    tokens: int
    tool_calls: int
    wall_time_s: float
    steps: int
    max_steps_hit: bool = False
    setup_failed: bool = False
    malformed_turns: int = 0
    repeated_actions: int = 0

    @property
    def verified(self) -> bool:
        """Passed what is published AND was not caught by what is withheld."""
        return self.public_passed and self.hidden_passed is not False

    @classmethod
    def from_metrics(cls, metrics: Any, *, repeated_actions: int = 0) -> Attempt:
        return cls(
            public_passed=bool(metrics.public_passed),
            hidden_passed=metrics.hidden_passed,
            tokens=int(metrics.tokens_used),
            tool_calls=int(metrics.tool_calls),
            wall_time_s=float(metrics.wall_time_s),
            steps=int(metrics.steps),
            max_steps_hit=bool(metrics.max_steps_hit),
            setup_failed=bool(metrics.setup_failed),
            malformed_turns=int(getattr(metrics, "malformed_turns", 0)),
            repeated_actions=repeated_actions,
        )


def classify(attempt: Attempt, *, envelope_tokens: int | None = None, envelope_multiple: float = 2.0) -> str:
    """Why this attempt is a challenge, or HEALTHY.

    Ordered deliberately. `setup_failed` first because it is infrastructure breakage
    masquerading as an agent failure -- a challenge opened on it sends miners to fix a
    broken workspace. `overfit` before `hidden_verify_failed` because the two are the same
    exit status and completely different findings: one is a model that could not do the
    task, the other is a model that learned the published check.
    """
    if attempt.setup_failed:
        return SETUP_FAILED
    if attempt.malformed_turns:
        # Before correctness: a run the protocol could not read was never really scored, and
        # the fix is the wire format rather than the strategy.
        return MALFORMED_PROTOCOL
    if attempt.public_passed and attempt.hidden_passed is False:
        return OVERFIT
    if attempt.hidden_passed is False:
        return HIDDEN_VERIFY_FAILED
    if not attempt.public_passed:
        if attempt.max_steps_hit:
            return STEP_BUDGET_EXHAUSTED
        if attempt.repeated_actions >= 3:
            return NO_PROGRESS_LOOP
        return PUBLIC_VERIFY_FAILED
    if envelope_tokens and attempt.tokens > envelope_multiple * envelope_tokens:
        return EXPENSIVE_SUCCESS
    return HEALTHY


@dataclass(frozen=True)
class Baseline:
    """Repeated attempts by the frozen model under the canonical strategy."""

    task_id: str
    attempts: tuple[Attempt, ...]

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ChallengeError(f"{self.task_id}: a baseline with no attempts establishes nothing")

    @property
    def passes(self) -> int:
        return sum(1 for a in self.attempts if a.verified)

    @property
    def pass_rate(self) -> float:
        return self.passes / len(self.attempts)

    @property
    def bound(self) -> tuple[float, float]:
        """95% interval on the true pass rate, from the repo's own Wilson implementation."""
        from hermesbench.repeats import wilson

        i = wilson(self.passes, len(self.attempts))
        return round(i.low, 4), round(i.high, 4)

    @property
    def median_tokens(self) -> int:
        return int(median([a.tokens for a in self.attempts]))

    @property
    def token_spread(self) -> float:
        """Observed relative spread, which is what an acceptance margin has to beat.

        Recorded here because the run that produces challenges is the only run that can
        measure it, and `hermes.acceptance` refuses a margin the spread swamps.
        """
        toks = [a.tokens for a in self.attempts]
        mid = median(toks)
        if len(toks) < 2 or mid <= 0:
            return float("inf")
        return round((max(toks) - min(toks)) / (2.0 * mid), 4)

    def confirmed(self, *, max_pass_rate: float = 0.5, min_attempts: int = 5) -> tuple[bool, str]:
        """Whether this is reliably a failure, or one unlucky run. Returns (verdict, reason).

        A single failure is often the sampling noise of a stochastic decoder. Opening a
        challenge on it sends every miner in the subnet to work on a task the baseline
        solves most of the time, and the round produces nothing anyone can learn from.
        """
        low, high = self.bound
        if len(self.attempts) < min_attempts:
            return False, (
                f"{self.passes}/{len(self.attempts)} attempts leaves the true pass rate anywhere in "
                f"[{low:.0%}, {high:.0%}]; a challenge opened on that may be one unlucky sample. "
                f"{min_attempts} attempts is the floor."
            )
        if self.pass_rate > max_pass_rate:
            return False, (
                f"the baseline passes {self.pass_rate:.0%} of attempts, above the {max_pass_rate:.0%} bar; "
                "this is a flaky task rather than a capability gap, and miners cannot tell the difference "
                "from inside a round"
            )
        return True, ""

    def dominant_class(self, *, envelope_tokens: int | None = None) -> str:
        """The class most attempts fell into. Ties resolve toward the earlier, worse class."""
        order = [
            SETUP_FAILED,
            MALFORMED_PROTOCOL,
            OVERFIT,
            HIDDEN_VERIFY_FAILED,
            STEP_BUDGET_EXHAUSTED,
            NO_PROGRESS_LOOP,
            PUBLIC_VERIFY_FAILED,
            EXPENSIVE_SUCCESS,
            HEALTHY,
        ]
        seen = [classify(a, envelope_tokens=envelope_tokens) for a in self.attempts]
        counts = {c: seen.count(c) for c in set(seen)}
        best = max(counts.values())
        return next(c for c in order if counts.get(c) == best)


# What a challenge may say about its task. An allowlist, for the same reason
# `hermes.miner_contract` is one: a denylist that filtered `hidden_verify` out would keep
# working right up until somebody added a second withheld field, and the failure mode is
# silent publication of the answer key.
#
# A caller handing in a whole task record is the expected case rather than an abuse -- the
# packet is the thing that gets published, so the packet does the stripping. Relying on
# every caller to pre-filter is the arrangement `redact_for_release` already exists because
# nobody reliably does.
PUBLISHABLE_TASK_KEYS = frozenset(
    {
        "task_id",
        "category",
        "prompt",
        "setup",
        "verify",
        "max_steps",
        "timeout_s",
        "env",
        "fingerprint",
        "lineage",
        "split",
        "hidden_verify_commitment",
        "has_hidden_tests",
        "declares_hidden_tests",
    }
)


@dataclass(frozen=True)
class Challenge:
    """One published, immutable challenge."""

    task_id: str
    failure_class: str
    baseline: Baseline
    epoch: dict[str, Any]
    task_pins: dict[str, Any] = field(default_factory=dict)

    @property
    def published_pins(self) -> dict[str, Any]:
        """Task fields this packet may carry. Anything unrecognised is dropped."""
        return {k: v for k, v in sorted(self.task_pins.items()) if k in PUBLISHABLE_TASK_KEYS}

    @property
    def dropped_task_keys(self) -> tuple[str, ...]:
        """Names of the fields stripped on the way out. Names only, never values.

        Reported so a maintainer who passed a full task record can see that the strip
        happened, instead of having to trust that it did.
        """
        return tuple(sorted(k for k in self.task_pins if k not in PUBLISHABLE_TASK_KEYS))

    @property
    def digest(self) -> str:
        """Content address. Excludes nothing, so two identical packets are one challenge."""
        return digest_mapping(self.to_record(with_digest=False))

    def to_record(self, *, with_digest: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": "spark-challenge-v1",
            "task_id": self.task_id,
            "failure_class": self.failure_class,
            "epoch": dict(sorted(self.epoch.items())),
            "task": self.published_pins,
            "baseline": {
                "attempts": len(self.baseline.attempts),
                "verified_passes": self.baseline.passes,
                "pass_rate": round(self.baseline.pass_rate, 4),
                "true_pass_rate_interval": list(self.baseline.bound),
                "median_tokens": self.baseline.median_tokens,
                "median_tool_calls": int(median([a.tool_calls for a in self.baseline.attempts])),
                "median_steps": int(median([a.steps for a in self.baseline.attempts])),
                # Recorded, never scored across nodes. Wall time is not reproducible, and a
                # bar that only rises would lock in whichever run got favourable scheduling.
                "median_wall_time_s": round(median([a.wall_time_s for a in self.baseline.attempts]), 3),
                "token_spread": self.baseline.token_spread,
            },
            # What a miner may NOT have, stated in the packet so nobody has to infer it from
            # an absence. The commitment proves which check will be used without revealing it.
            "withheld": {
                "hidden_verify_commitment": str(self.task_pins.get("hidden_verify_commitment") or ""),
                "body_included": False,
                "dropped_task_keys": list(self.dropped_task_keys),
            },
            "acceptance_thresholds_included": False,
            "why_no_thresholds": (
                "the run that produces challenges is the run that measures the token spread, so a "
                "threshold baked in here would freeze a guess at the moment the data to replace it "
                "arrives. hermes.acceptance reads the spread from baseline.token_spread instead."
            ),
        }
        if with_digest:
            record["challenge_digest"] = self.digest
        return record


def open_challenge(
    baseline: Baseline,
    *,
    epoch: dict[str, Any],
    task_pins: dict[str, Any] | None = None,
    envelope_tokens: int | None = None,
    max_pass_rate: float = 0.5,
    min_attempts: int = 5,
) -> Challenge:
    """Package a confirmed baseline failure. Refuses an unconfirmed one.

    Raising rather than returning a flag: an unconfirmed challenge that reaches miners costs
    a whole round, and the caller has no better information than this function does.
    """
    failure_class = baseline.dominant_class(envelope_tokens=envelope_tokens)

    # Classification-driven refusals come before the confirmation gate, because the gate
    # would otherwise misdiagnose them. A baseline that passes every attempt inside its
    # envelope trips the pass-rate bar and gets reported as "a flaky task" -- which is
    # exactly backwards, and the maintainer reading it goes looking for nondeterminism that
    # isn't there.
    if failure_class == SETUP_FAILED:
        raise ChallengeError(
            f"{baseline.task_id}: the task's own setup failed, which is infrastructure breakage rather "
            "than an agent failure. A challenge here sends miners to fix a broken workspace."
        )
    # Both remaining special cases require the baseline to pass CONSISTENTLY, and that
    # condition is doing real work rather than being belt-and-braces. `dominant_class` is a
    # majority vote, so a baseline that passes 7 of 10 returns HEALTHY -- and refusing that
    # as "nothing to improve" would bury a task that fails almost a third of the time.
    # Mixed baselines fall through to the confirmation gate, which names the reliability
    # problem instead.
    if baseline.pass_rate == 1.0:
        if failure_class == HEALTHY:
            raise ChallengeError(
                f"{baseline.task_id}: the baseline passes all {len(baseline.attempts)} attempts inside "
                "its resource envelope; there is nothing for a miner to improve and a challenge opened "
                "on it wastes a round"
            )
        if failure_class == EXPENSIVE_SUCCESS:
            # A resource challenge, so the pass-rate bar cannot apply -- it would refuse the
            # packet for the very thing that defines it. The attempt-count floor still does:
            # the envelope is a median, and one sample has no median worth publishing.
            if len(baseline.attempts) < min_attempts:
                raise ChallengeError(
                    f"{baseline.task_id}: "
                    f"{baseline.confirmed(max_pass_rate=max_pass_rate, min_attempts=min_attempts)[1]}"
                )
            return Challenge(
                task_id=baseline.task_id,
                failure_class=failure_class,
                baseline=baseline,
                epoch=epoch,
                task_pins=dict(task_pins or {}),
            )

    ok, reason = baseline.confirmed(max_pass_rate=max_pass_rate, min_attempts=min_attempts)
    if not ok:
        raise ChallengeError(f"{baseline.task_id}: {reason}")

    return Challenge(
        task_id=baseline.task_id,
        failure_class=failure_class,
        baseline=baseline,
        epoch=epoch,
        task_pins=dict(task_pins or {}),
    )


__all__ = [
    "EXPENSIVE_SUCCESS",
    "HEALTHY",
    "HIDDEN_VERIFY_FAILED",
    "MALFORMED_PROTOCOL",
    "NO_PROGRESS_LOOP",
    "OVERFIT",
    "PUBLIC_VERIFY_FAILED",
    "SETUP_FAILED",
    "STEP_BUDGET_EXHAUSTED",
    "PUBLISHABLE_TASK_KEYS",
    "Attempt",
    "Baseline",
    "Challenge",
    "ChallengeError",
    "CHALLENGES_DIR",
    "classify",
    "from_episode_log",
    "unverifiable_tasks",
    "open_challenge",
]

# --- opening challenges from a real baseline log ----------------------------------------------

CHALLENGES_DIR = Path("datasets/challenges")


def from_episode_log(
    rows: Iterable[dict[str, Any]],
    *,
    epoch: dict[str, Any],
    task_pins: dict[str, dict[str, Any]] | None = None,
    envelope_multiple: float = 2.0,
    min_attempts: int = 5,
) -> tuple[list[Challenge], list[tuple[str, str]]]:
    """Group a baseline's episodes by task and open what qualifies. Returns (opened, refused).

    Consumes exactly what `hermesbench.runner --episodes-out` writes -- one
    `EpisodeMetrics.to_record()` per line -- so the log a baseline already produces is the input
    here rather than a second format somebody has to export. That was the missing join: the
    baseline wrote metrics, this module could package them, and nothing carried one to the other.

    Refusals are returned rather than logged away. A task that did not become a challenge is the
    more common outcome and the reason matters -- "the baseline handles this" and "the baseline is
    flaky here" call for different work, and a caller that only sees the successes cannot tell
    which happened.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_id") or "")].append(row)

    opened: list[Challenge] = []
    refused: list[tuple[str, str]] = []
    for task_id, rows in sorted(grouped.items()):
        attempts = tuple(
            Attempt(
                public_passed=bool(r.get("public_passed")),
                hidden_passed=r.get("hidden_passed"),
                tokens=int(r.get("tokens_used") or 0),
                tool_calls=int(r.get("tool_calls") or 0),
                wall_time_s=float(r.get("wall_time_s") or 0.0),
                steps=int(r.get("steps") or 0),
                max_steps_hit=bool(r.get("max_steps_hit")),
                setup_failed=bool(r.get("setup_failed")),
                malformed_turns=int(r.get("malformed_turns") or 0),
            )
            for r in rows
        )
        baseline = Baseline(task_id=task_id, attempts=attempts)
        try:
            opened.append(
                open_challenge(
                    baseline,
                    epoch=epoch,
                    task_pins=(task_pins or {}).get(task_id),
                    envelope_tokens=baseline.median_tokens,
                    min_attempts=min_attempts,
                )
            )
        except ChallengeError as exc:
            refused.append((task_id, str(exc).split(": ", 1)[-1]))
    return opened, refused


def unverifiable_tasks(
    rows: Iterable[dict[str, Any]],
    *,
    current: dict[str, str],
    as_of: dict[str, str] | None = None,
    trust_unstamped: bool = False,
) -> dict[str, str]:
    """Tasks whose log cannot be shown to describe the grader in the tree now. task_id -> reason.

    This is the check that would have caught the problem at its source. `verify_digest` is
    stamped onto every episode by the runner; when the verify script has changed since, the log
    describes a grader that no longer exists and its pass rate says nothing about this suite.

    Per task rather than per run, because one repaired verifier should not invalidate eighteen
    good baselines.

    An unstamped log -- every log written before the field existed, including the first real
    baseline -- cannot be checked directly, and `as_of` recovers the missing stamp from git. With
    neither a stamp nor a recovered one, the task is unverifiable rather than assumed to match:
    absence of a mismatch is not evidence of agreement, and defaulting the other way is exactly
    how a stale log gets published.
    """
    as_of = as_of or {}
    problems: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        stamped = str(row.get("verify_digest") or "") or as_of.get(task_id, "")
        if not stamped:
            if not trust_unstamped:
                problems[task_id] = (
                    "its episodes carry no verify_digest, so nothing says which grader produced them. "
                    "Re-run the baseline, pass --baseline-ref to recover the stamp from git, or "
                    "--trust-unstamped if you have checked by hand that the verifier has not changed"
                )
        elif stamped != current.get(task_id, stamped):
            problems[task_id] = (
                f"the log was produced by a different grader: episodes stamp {stamped[:23]}... and the "
                f"suite now has {str(current.get(task_id))[:23]}.... The recorded pass rate describes a "
                "verifier that no longer exists; re-run the baseline for this task"
            )
    return problems


def _verify_digests_at(ref: str) -> dict[str, str]:
    """Digest every task's published verify script as it stood at a git ref.

    Reads the task YAML out of the old tree rather than importing it: importing a suite from
    another revision means running that revision's code, and the point here is to inspect a
    tree, not to trust it.

    Returns an empty mapping if git cannot answer -- a missing ref or a checkout without the
    history. Empty means "no stamp recovered", which leaves the unstamped-log refusal in force
    rather than silently passing everything.
    """
    import subprocess

    import yaml

    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref, "hermesbench/tasks/"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    digests: dict[str, str] = {}
    for path in listing:
        if not path.endswith((".yaml", ".yml")):
            continue
        try:
            blob = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
            spec = yaml.safe_load(blob) or {}
        except (subprocess.CalledProcessError, yaml.YAMLError):
            continue
        task_id = str(spec.get("task_id") or Path(path).stem)
        digests[task_id] = "sha256:" + hashlib.sha256(str(spec.get("verify") or "").encode("utf-8")).hexdigest()
    return digests


def main(argv: list[str] | None = None) -> int:
    import argparse
    import dataclasses
    import json

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, required=True, help="JSONL from `runner --episodes-out`")
    parser.add_argument("--out", type=Path, default=CHALLENGES_DIR, help="where to write challenge packets")
    parser.add_argument("--model-revision", required=True, help="the pinned model revision this baseline ran")
    parser.add_argument("--harness-digest", required=True, help="the harness digest this baseline ran")
    parser.add_argument("--min-attempts", type=int, default=5)
    parser.add_argument(
        "--baseline-ref",
        default="",
        help="git ref of the tree this log was produced from; recovers the missing verify_digest for an "
        "unstamped log by reading the task YAML out of that tree",
    )
    parser.add_argument(
        "--trust-unstamped",
        action="store_true",
        help="publish from a log with no verify_digest (pre-dating the field); you are asserting by hand "
        "that the verifiers have not changed since it was written",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would open, write nothing")
    args = parser.parse_args(argv)

    from hermesbench.runner import verify_digest as verify_digest_of
    from hermesbench.sink import read_episodes
    from hermesbench.suitecheck import missing_commands
    from hermesbench.tasks import load_suite

    rows = list(read_episodes(args.episodes))
    if not rows:
        print(f"hermes.challenge: no episodes in {args.episodes}", file=sys.stderr)
        return 2

    # The task definition comes from the suite rather than from the log. Two reasons, and they
    # are different reasons.
    #
    # The commitment, because an episode record does not carry one, and a challenge published
    # without it cannot say which withheld check will grade it -- the field `overfit_rate`
    # ultimately rests on.
    #
    # The prompt, setup and verify script, because a packet without them is not a work order.
    # `PUBLISHABLE_TASK_KEYS` is an allowlist of 14 keys and the first version of this passed 2,
    # producing a well-formed packet a miner could not have worked from. The allowlist is the
    # thing deciding what is publishable; a caller that hand-picks a subset is second-guessing
    # it, and the deny-by-default direction means the safe move is to hand over the whole record
    # and let `Challenge.to_record` drop what it must -- which it then reports in
    # `dropped_task_keys`, by name.
    pins = {}
    for t in load_suite("all"):
        record = {f.name: getattr(t, f.name) for f in dataclasses.fields(t)}
        record["hidden_verify_commitment"] = t.hidden_verify_commitment
        pins[t.task_id] = record
    # Prove the grader can run before publishing a challenge on the fact that it did not pass.
    #
    # This is not a hypothetical. The first real baseline log opened six challenges, and two --
    # `fix-failing-test` and `verify-speedup-claim` -- were tasks whose verifiers invoked a bare
    # `python`, which the harness does not guarantee. The agent finished, declared itself done,
    # and the grader failed it anyway; both scored 10/10 once the interpreter was resolved.
    # Publishing those would have sent miners to fix a verifier for two full rounds.
    #
    # The check is `suitecheck.missing_commands` rather than anything inferred from the metrics,
    # and the difference matters. The tempting signal was that neither task ever exhausted its
    # step budget while the four genuine challenges exhausted theirs 9 or 10 times out of 10 --
    # true of this run, and wrong as a rule: a model that writes a bad patch and stops is the
    # single most common real capability gap there is, and a step-budget heuristic refuses
    # exactly that. Asking whether the verifier's commands exist answers the actual question.
    #
    # It cannot catch every ungradeable verifier -- a grader can be present and still unable to
    # pass -- so it is a precondition, not a proof. An episode record carries no harness digest,
    # which is the gap underneath all of this: a log and the epoch it is published under can
    # disagree in silence.
    ungradeable: dict[str, str] = {}
    for task in load_suite("all"):
        absent = sorted({*missing_commands(task.verify), *missing_commands(task.hidden_verify or "")})
        if absent:
            ungradeable[task.task_id] = (
                f"its verifier invokes {absent}, which do not exist here, so it can never pass on this "
                "machine and every attempt scores 0 for a reason the model never caused"
            )

    gradeable_rows = [r for r in rows if r.get("task_id") not in ungradeable]
    epoch = {"model_revision": args.model_revision, "harness_digest": args.harness_digest}
    as_of = _verify_digests_at(args.baseline_ref) if args.baseline_ref else {}
    ungradeable.update(
        unverifiable_tasks(
            gradeable_rows,
            current={t.task_id: verify_digest_of(t) for t in load_suite("all")},
            as_of=as_of,
            trust_unstamped=args.trust_unstamped,
        )
    )

    gradeable = [r for r in gradeable_rows if r.get("task_id") not in ungradeable]
    opened, refused = from_episode_log(rows=gradeable, epoch=epoch, task_pins=pins, min_attempts=args.min_attempts)
    refused = sorted([*refused, *ungradeable.items()])

    print(f"{len(rows)} episodes over {len({r.get('task_id') for r in rows})} task(s)")
    for challenge in opened:
        b = challenge.baseline
        print(
            f"  OPEN    {challenge.task_id:<32} {challenge.failure_class:<22} "
            f"pass {b.passes}/{len(b.attempts)}  spread {b.token_spread:.1%}"
        )
    for task_id, why in refused:
        print(f"  refused {task_id:<32} {why[:96]}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for challenge in opened:
        path = args.out / f"{challenge.task_id}.json"
        path.write_text(json.dumps(challenge.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

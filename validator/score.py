"""Score a submitted surface against the challenge it was opened over.

    python -m validator.score --round r-001 --miner alice \\
        --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b --repeats 10

This is the step that turns an accepted pull request into a decision. `eval.strategy_track` decides
whether a surface may merge; this runs it and says whether it beat the baseline.

## The validator does not re-run a control arm, and that is the point

A miner rehearsing locally has to measure both arms, because their box is not the validator's. The
validator already holds the authoritative control: the challenge was *opened* from a measured
baseline, and `RoundWindow.snapshot()` carries every one of its individual attempts. That baseline
is the bar the round published, so re-measuring it would replace a published bar with a fresh one
and quietly move the target between submissions.

The cost of using a stored baseline is that it was measured at a different moment, so the
comparison is only meaningful while the epoch holds. `Challenge.epoch` records `model_revision` and
`harness_digest` for exactly this, and `epoch_issues` refuses a cross-epoch comparison rather than
producing a number. A candidate measured under a newer harness against a baseline measured under an
older one is a measurement of the harness change.

## Correctness here means the withheld check too

A miner's local rehearsal can only count `public_passed`; they do not hold the withheld verifiers,
which is the entire reason `overfit_rate` means anything. The validator counts
`Attempt.verified` -- published check passed *and* the withheld one did not fail -- so a surface
that learned the published assertions scores as the failure it is.

`overfit_attempts` is reported separately rather than folded into the pass count. "Failed the task"
and "passed the task it was shown and failed the one it was not" call for different responses: the
first is a capability gap, the second is a surface fitted to the benchmark, and averaging them into
one rate would hide the only signal that distinguishes them.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.acceptance import Arm, Decision, decide, reduction_interval, repeats_needed


class ScoreError(RuntimeError):
    """A submission cannot be scored."""


@dataclass(frozen=True)
class Scorecard:
    round_id: str
    miner_id: str
    task_id: str
    candidate: Arm
    baseline: Arm
    decision: Decision
    interval: tuple[float, float]
    repeats_to_settle: int | None
    overfit_attempts: int
    protocol_failures: int
    identity: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def to_record(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "policy": self.policy,
            "policy_hash": policy_hash(self.policy),
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "task_id": self.task_id,
            # `tool_calls` travels with the tokens because the crown compares both. Without it
            # `validator.crown` read zeros, and `dominates`' tool-call guard -- the one stopping a
            # challenger from buying a token win by collapsing thirty operations into one helper
            # call -- compared 0 against 0 and passed every time. A guard that cannot fire.
            "candidate": {
                "verified_passes": self.candidate.passes,
                "attempts": self.candidate.attempts,
                "tokens": list(self.candidate.tokens),
                "tool_calls": list(self.candidate.tool_calls),
            },
            "baseline": {
                "verified_passes": self.baseline.passes,
                "attempts": self.baseline.attempts,
                "tokens": list(self.baseline.tokens),
                "tool_calls": list(self.baseline.tool_calls),
            },
            "decision": self.decision.to_record(),
            "reduction_interval": list(self.interval),
            "repeats_to_settle": self.repeats_to_settle,
            # Kept out of the pass count on purpose. "Failed the task" and "passed the published
            # check and failed the withheld one" are different findings and only the second says a
            # surface was fitted to the benchmark.
            "overfit_attempts": self.overfit_attempts,
            "protocol_failures": self.protocol_failures,
            "baseline_is_the_published_bar_not_a_fresh_run": True,
        }


def epoch_issues(epoch: dict[str, Any], *, model_revision: str, harness_digest: str) -> list[str]:
    """Whether a candidate measured now is comparable to a baseline measured then.

    Both directions are named rather than summarised. A model change and a harness change produce
    the same wrong number and call for different repairs -- re-pin, or re-baseline -- and a caller
    told only "epoch mismatch" has to go and find out which.

    An empty field in the challenge is reported as unverifiable rather than treated as matching.
    Absence of a mismatch is not evidence of agreement, and every version of this mistake in this
    repository has looked like a clean result.
    """
    issues: list[str] = []
    want_model = str(epoch.get("model_revision") or "")
    want_harness = str(epoch.get("harness_digest") or "")

    if not want_model or not want_harness:
        issues.append(
            f"the challenge's epoch is incomplete (model_revision={want_model!r}, "
            f"harness_digest={want_harness!r}), so nothing can establish that this candidate is "
            "comparable to its baseline"
        )
        return issues
    if want_model != model_revision:
        issues.append(
            f"the baseline was measured on model {want_model} and this candidate on {model_revision}; "
            "comparing them measures the model change rather than the surface"
        )
    if want_harness != harness_digest:
        issues.append(
            f"the baseline was measured under harness {want_harness} and this candidate under "
            f"{harness_digest}; comparing them measures the harness change rather than the surface"
        )
    return issues


def policy_record() -> dict[str, Any]:
    from hermes import acceptance as a

    return {
        "version": "strategy-score-v1",
        "min_attempts": a.MIN_ATTEMPTS,
        "min_token_reduction": a.MIN_TOKEN_REDUCTION,
        "confidence": a.CONFIDENCE,
        "bootstrap_resamples": a.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": a.BOOTSTRAP_SEED,
        "correctness": "all_required_verifiers",
        "verification_unchanged": True,
        "token_statistic": "median",
        "interval_gate": "lower_bound_gte_margin",
    }


def policy_hash(value: Any) -> str:
    from validator.pr_admission import digest

    return digest(value)


def acceptance_decision(candidate: Arm, baseline: Arm, policy: dict[str, Any]) -> Decision:
    """The single versioned gate used by score production and crown consumption."""
    if policy != policy_record() or policy_hash(policy) != policy_hash(policy_record()):
        raise ScoreError("unknown, incomplete or changed score policy")
    return decide(candidate=candidate, baseline=baseline)


def private_check_required(challenge: Any) -> bool:
    pins = challenge.task_pins
    declaration = pins.get("private_check_required", pins.get("declares_hidden_tests"))
    if (
        "private_check_required" in pins
        and "declares_hidden_tests" in pins
        and pins["private_check_required"] is not pins["declares_hidden_tests"]
    ):
        raise ScoreError("conflicting private-check declarations")
    commitment = pins.get("hidden_verify_commitment")
    if declaration is not None and type(declaration) is not bool:
        raise ScoreError("private-check declaration must be boolean")
    if commitment:
        if declaration is False:
            raise ScoreError("private-free declaration contradicts withheld commitment")
        return True
    if declaration is False:
        return False
    raise ScoreError("task must explicitly declare its private-check requirement")


def _integer(row: dict[str, Any], key: str, *, positive: bool = False) -> int:
    value = row.get(key)
    if type(value) is not int or value < (1 if positive else 0):
        raise ScoreError(
            f"invalid {key}: expected {'positive' if positive else 'nonnegative'} measured integer; a zero-token run did not happen"
        )
    return value


def normalize_episode(row: dict[str, Any]) -> dict[str, Any]:
    from hermes.challenge import ChallengeError, episode_metrics_of

    try:
        return episode_metrics_of(row)
    except ChallengeError as exc:
        raise ScoreError(str(exc)) from exc


def validate_execution(row: dict[str, Any], *, private_required: bool) -> None:
    row = normalize_episode(row)
    for key in (
        "public_passed",
        "success",
        "setup_failed",
        "max_steps_hit",
        "disqualified",
        "integrity_clean",
        "integrity_fully_checked",
        "protocol_clean",
    ):
        if type(row.get(key)) is not bool:
            raise ScoreError(f"missing or non-boolean {key}")
    for key in ("truncated", "harness_final", "integrity_disqualified"):
        if key in row and row[key] is not False:
            raise ScoreError(f"invalid execution flag {key}")
    integrity = row.get("integrity")
    if "integrity" in row:
        if not isinstance(integrity, dict) or any(
            integrity.get(k) is not row[v]
            for k, v in (
                ("clean", "integrity_clean"),
                ("fully_checked", "integrity_fully_checked"),
                ("disqualified", "disqualified"),
            )
        ):
            raise ScoreError("contradictory or malformed integrity evidence")
    if "executed" in row and row["executed"] is not True:
        raise ScoreError("trajectory was not executed")
    if "trajectory_task_id" in row and row["trajectory_task_id"] != row.get("task_id"):
        raise ScoreError("trajectory task differs from episode task")
    if "private_check_required" in row and row["private_check_required"] is not private_required:
        raise ScoreError("episode private-check declaration contradicts task")
    hidden = row.get("hidden_passed")
    if private_required and type(hidden) is not bool:
        raise ScoreError("required private result is missing or non-boolean")
    if not private_required and hidden is not None:
        raise ScoreError("private-free task has unexpected private evidence")
    if (
        row["setup_failed"]
        or row["max_steps_hit"]
        or row["disqualified"]
        or not row["integrity_clean"]
        or not row["integrity_fully_checked"]
    ):
        raise ScoreError("execution failed integrity or was truncated/setup-failed")
    if _integer(row, "malformed_turns") or not row["protocol_clean"]:
        raise ScoreError("malformed protocol cannot receive credit")
    _integer(row, "tokens_used", positive=True)
    _integer(row, "tool_calls")
    _integer(row, "steps")
    latency = row.get("wall_time_s")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
        raise ScoreError("invalid measured wall_time_s")
    if row.get("cost") is not None:
        cost = row["cost"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ScoreError("invalid measured cost")
    verified = row["public_passed"] and (hidden is True if private_required else True)
    if row["success"] is not verified:
        raise ScoreError("success contradicts required verifier evidence")


def metrics_from_bytes(raw: bytes, *, source: str = "episode snapshot") -> list[dict[str, Any]]:
    """Normalize exactly the bytes checked for completeness and retained for hashing."""
    from hermes.challenge import episode_metrics_of
    from hermesbench.sink import decode_episodes

    try:
        return [episode_metrics_of(row) for row in decode_episodes(raw, source=source)]
    except (ValueError, RuntimeError, RecursionError) as exc:
        raise ScoreError(f"cannot score episode log: {exc}") from exc


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read one complete snapshot; never reopen between checking and decoding."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ScoreError(f"cannot read episode log: {exc}") from exc
    return metrics_from_bytes(raw, source=str(path))


def candidate_arm(rows: list[dict[str, Any]], *, private_required: bool = True) -> tuple[Arm, int, int]:
    if not rows:
        raise ScoreError("no episodes; there is nothing to score")
    rows = [normalize_episode(row) for row in rows]
    for row in rows:
        validate_execution(row, private_required=private_required)
    return (
        Arm(
            passes=sum(r["success"] for r in rows),
            attempts=len(rows),
            tokens=tuple(r["tokens_used"] for r in rows),
            tool_calls=tuple(r["tool_calls"] for r in rows),
        ),
        sum(r["public_passed"] and r.get("hidden_passed") is False for r in rows),
        0,
    )


def schedule(challenge: Any) -> list[str]:
    epoch = challenge.epoch
    attempts = epoch.get("attempt_ids")
    if not isinstance(epoch.get("epoch_id"), str) or not epoch["epoch_id"]:
        raise ScoreError("active epoch_id is required")
    if (
        not isinstance(attempts, list)
        or len(attempts) < 10
        or any(not isinstance(a, str) or not a for a in attempts)
        or len(set(attempts)) != len(attempts)
    ):
        raise ScoreError("epoch must declare at least 10 unique attempt_ids")
    return attempts


def validate_identity(
    rows: list[dict[str, Any]],
    challenge: Any,
    *,
    round_id: str | None = None,
    bundle_sha256: str | None = None,
    origin: dict[str, Any] | None = None,
) -> None:
    attempts = schedule(challenge)
    expected = {k: challenge.epoch[k] for k in ("epoch_id", "model_revision", "harness_digest")}
    expected["task_id"] = challenge.task_id
    if round_id is not None:
        expected["round_id"] = round_id
    if bundle_sha256 is not None:
        expected["bundle_sha256"] = bundle_sha256
    if origin is not None:
        expected["origin"] = origin
    verify = challenge.task_pins.get("verify_digest")
    if not verify:
        import hashlib

        script = challenge.task_pins.get("verify")
        if not isinstance(script, str) or not script:
            raise ScoreError("task public verifier identity is missing")
        verify = "sha256:" + hashlib.sha256(script.encode("utf-8")).hexdigest()
    expected["verify_digest"] = verify
    seen = []
    for row in rows:
        for key, value in expected.items():
            if row.get(key) != value:
                raise ScoreError(f"episode {key} does not match admitted round identity")
        seen.append(row.get("attempt_id"))
    if any(not isinstance(x, str) for x in seen) or len(seen) != len(attempts) or set(seen) != set(attempts):
        raise ScoreError("duplicate, missing or excess attempts; declared schedule must match exactly")


def baseline_arm(challenge: Any, *, origin: dict[str, Any] | None = None) -> Arm:
    attempts = challenge.baseline.attempts
    if len(attempts) < 2:
        raise ScoreError("single baseline attempt carries no information about its own variability")
    if challenge.baseline.task_id != challenge.task_id:
        raise ScoreError("baseline task differs from round task")
    rows = []
    for attempt in attempts:
        row = dataclasses.asdict(attempt)
        evidence = row.pop("evidence", None)
        if not isinstance(evidence, dict):
            raise ScoreError("baseline is missing original execution evidence; re-baseline required")
        evidence = normalize_episode(evidence)
        validate_execution(evidence, private_required=private_check_required(challenge))
        for key, value in row.items():
            source_key = "tokens_used" if key == "tokens" else key
            if source_key in evidence and evidence[source_key] != value:
                raise ScoreError(f"baseline {key} disagrees with original execution evidence")
        # Original booleans/counts take precedence: evidence cannot replace the oracle.
        row = {**evidence, **row, "tokens_used": attempt.tokens}
        rows.append(row)
    validate_identity(rows, challenge, origin=origin)
    return candidate_arm(rows, private_required=private_check_required(challenge))[0]


def score(
    *, window: Any, miner_id: str, rows: list[dict[str, Any]], model_revision: str, harness_digest: str
) -> Scorecard:
    rows = [normalize_episode(row) for row in rows]
    challenge = window.challenge
    mismatch = epoch_issues(challenge.epoch, model_revision=model_revision, harness_digest=harness_digest)
    if mismatch:
        raise ScoreError("; ".join(mismatch))
    from validator.pr_admission import admission_for

    try:
        admission = admission_for(window, miner_id)
    except ValueError as exc:
        raise ScoreError(str(exc)) from exc
    standing = window.submissions[miner_id]
    baseline = baseline_arm(challenge, origin=admission["origin"])
    validate_identity(
        rows, challenge, round_id=window.round_id, bundle_sha256=standing.payload_digest, origin=admission["origin"]
    )
    candidate, overfit, malformed = candidate_arm(rows, private_required=private_check_required(challenge))
    policy = policy_record()
    declared_policy = challenge.epoch.get("score_policy")
    if declared_policy != policy or policy_hash(declared_policy) != policy_hash(policy):
        raise ScoreError("unknown or changed score policy")
    return Scorecard(
        round_id=window.round_id,
        miner_id=miner_id,
        task_id=challenge.task_id,
        candidate=candidate,
        baseline=baseline,
        decision=acceptance_decision(candidate, baseline, policy),
        interval=reduction_interval(baseline.tokens, candidate.tokens),
        repeats_to_settle=repeats_needed(baseline.tokens, candidate.tokens),
        overfit_attempts=overfit,
        protocol_failures=malformed,
        policy=policy,
        identity={
            "epoch": challenge.epoch,
            "admission_id": admission["admission_id"],
            "bundle_sha256": standing.payload_digest,
            "origin": admission["origin"],
        },
    )


def render(card: Scorecard) -> str:
    from statistics import median

    lines = [
        f"round {card.round_id}   task {card.task_id}   miner {card.miner_id}",
        f"  baseline   {card.baseline.passes}/{card.baseline.attempts} verified   "
        f"median {median(card.baseline.tokens):,.0f} tokens   (the published bar)",
        f"  candidate  {card.candidate.passes}/{card.candidate.attempts} verified   "
        f"median {median(card.candidate.tokens):,.0f} tokens",
    ]
    if card.overfit_attempts:
        lines.append(
            f"  OVERFIT    {card.overfit_attempts} attempt(s) passed the published check and failed the withheld one"
        )
    if card.protocol_failures:
        lines.append(f"  protocol   {card.protocol_failures} attempt(s) had at least one unparseable turn")
    low, high = card.interval
    lines.append(f"  95% interval on the reduction: [{low:.1%}, {high:.1%}]")
    lines.append("")
    lines.append("VERDICT: " + ("ACCEPTED" if card.accepted else "REFUSED"))
    for reason in card.decision.reasons:
        lines.append(f"  - {reason}")
    if card.repeats_to_settle is not None:
        lines.append(f"  about {card.repeats_to_settle} paired repeats would settle evidence like this")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", dest="round_id", required=True)
    parser.add_argument("--miner", dest="miner_id", required=True)
    parser.add_argument("--episodes", type=Path, default=None, help="score an existing log instead of running")
    parser.add_argument("--harness-digest", default="", help="the harness digest this candidate ran under")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the scorecard as JSON")
    args = parser.parse_args(argv)

    from hermes.base_model import load as load_pin
    from validator.store import RoundStore

    if args.episodes is None:
        print(
            "validator.score: --episodes is required. Running the candidate is a separate step so a "
            "scorecard can be recomputed from a log without spending the GPU time again.",
            file=sys.stderr,
        )
        return 2
    try:
        window = RoundStore(args.store).load(args.round_id)
        rows = read_metrics(args.episodes)
        card = score(
            window=window,
            miner_id=args.miner_id,
            rows=rows,
            model_revision=load_pin().revision,
            harness_digest=args.harness_digest or str(window.challenge.epoch.get("harness_digest") or ""),
        )
    except ScoreError as exc:
        print(f"validator.score: {exc}", file=sys.stderr)
        return 2
    print(render(card))
    if args.out:
        args.out.write_text(json.dumps(card.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if card.accepted else 1


__all__ = [
    "ScoreError",
    "Scorecard",
    "baseline_arm",
    "candidate_arm",
    "epoch_issues",
    "main",
    "render",
    "score",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Rank accepted active-round scores and persist crown intent before any delivery.

Use ``validator.settlement activate`` to bind an active round first. ``select``
commits the outcome and outbox atomically; ``actions`` reads that saved intent.
Neither command contacts GitHub. See docs/competition-settlement.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.acceptance import Arm, Decision, reduction_interval
from hermes.evidence_json import evidence_object
from validator.score import ScoreError, acceptance_decision, policy_record

CROWNS = Path("datasets/crowns.json")
CROWN_LABEL = "crown"


class CrownError(RuntimeError):
    """A round cannot be settled."""


@dataclass(frozen=True)
class Contender:
    """One evaluated submission, with the bar it was measured against."""

    task_id: str
    miner_id: str
    round_id: str
    candidate: Arm
    baseline: Arm
    pr: int = 0
    received_at: float = 0.0
    decision: Decision | None = None
    policy: dict[str, Any] = field(default_factory=policy_record)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def median_tokens(self) -> float:
        from statistics import median

        return median(self.candidate.tokens)

    @property
    def median_tool_calls(self) -> float:
        from statistics import median

        return median(self.candidate.tool_calls)


@dataclass(frozen=True)
class Score:
    contender: Contender
    lower_bound: float
    interval: tuple[float, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.contender.task_id,
            "miner_id": self.contender.miner_id,
            "round_id": self.contender.round_id,
            "pr": self.contender.pr,
            "score": round(self.lower_bound, 4),
            "reduction_interval": [round(v, 4) for v in self.interval],
            "median_tokens": self.contender.median_tokens,
            "median_tool_calls": self.contender.median_tool_calls,
            "attempts": self.contender.candidate.attempts,
            "identity": self.contender.evidence.get("identity", {}),
            "policy": self.contender.policy,
            "policy_hash": self.contender.evidence.get("policy_hash"),
        }


@dataclass
class Outcome:
    winner: Score | None
    ranked: list[Score] = field(default_factory=list)
    ineligible: list[tuple[Contender, str]] = field(default_factory=list)

    @property
    def losers(self) -> list[Contender]:
        """Everyone who does not hold the crown, eligible or not, once each. Their round is over.

        Deduplicated because a contender can appear on both lists: when the best of the field does
        not clear zero it is ranked *and* recorded as ineligible with the reason. Without this the
        same pull request is closed twice, which the API tolerates and the action list does not --
        a duplicate reads as two miners losing when it is one.
        """
        keep = self.winner.contender if self.winner else None
        seen: set[tuple[str, str]] = set()
        out: list[Contender] = []
        for contender in [s.contender for s in self.ranked] + [c for c, _ in self.ineligible]:
            key = (contender.round_id, contender.miner_id)
            if contender is keep or key in seen:
                continue
            seen.add(key)
            out.append(contender)
        return out

    def to_record(self) -> dict[str, Any]:
        return {
            "winner": self.winner.to_record() if self.winner else None,
            "ranked": [s.to_record() for s in self.ranked],
            "ineligible": [
                {"miner_id": c.miner_id, "task_id": c.task_id, "pr": c.pr, "reason": why} for c, why in self.ineligible
            ],
            # Said in the record: an hour with no eligible entry is not an hour whose field was
            # ranked and found wanting, and a reader has to be able to tell those apart.
            "crowned": self.winner is not None,
        }


BARREN_ROUNDS_BEFORE_ROTATION = 2


@dataclass(frozen=True)
class Standing:
    """What carries from one hour to the next."""

    winner: dict[str, Any] | None = None
    task_id: str = ""
    barren_rounds: int = 0

    @classmethod
    def from_record(cls, record: dict[str, Any] | None) -> Standing:
        record = record or {}
        return cls(
            winner=record.get("winner"),
            task_id=str(record.get("task_id") or ""),
            barren_rounds=int(record.get("barren_rounds") or 0),
        )

    @property
    def pr(self) -> int:
        return int((self.winner or {}).get("pr") or 0)


def next_task(current: str, available: list[str]) -> str:
    """The task after this one, cycling. Empty when there is nothing to rotate to.

    Round-robin over the sorted list rather than random: a rotation nobody can predict is a
    rotation nobody can prepare for, and the point of moving on is to give the field a task it
    might actually beat.
    """
    if not available:
        return current
    ordered = sorted(available)
    if current not in ordered:
        return ordered[0]
    return ordered[(ordered.index(current) + 1) % len(ordered)]


def settle(
    previous: dict[str, Any] | None,
    outcome: Outcome,
    *,
    available_tasks: list[str] | None = None,
) -> tuple[Standing, list[tuple[str, int]]]:
    """The next standing, and the label moves. Returns (standing, label actions).

    Three cases and they are genuinely different:

    A winner takes the crown from whoever held it, and the barren counter resets. The task stays --
    someone beat it, so it is a task worth running again.

    No winner leaves the crown where it is and increments the counter. Nothing is removed: a barren
    hour is a fact about this hour's field, not about the miner who last cleared the bar.

    Two barren hours rotate the task and reset the counter. That is evidence about the task rather
    than the field, and a third run would spend an hour of everyone's GPU time to learn the same
    thing. The crown still carries: it was won, nothing has taken it, and stripping it because the
    subject changed would punish the holder for other people's failure.
    """
    standing = Standing.from_record(previous)
    task = standing.task_id or (outcome.winner.contender.task_id if outcome.winner else "")
    if not task and outcome.ranked:
        task = outcome.ranked[0].contender.task_id

    if outcome.winner is not None:
        actions: list[tuple[str, int]] = []
        new_pr = outcome.winner.contender.pr
        if standing.pr and standing.pr != new_pr:
            actions.append(("remove", standing.pr))
        if new_pr and standing.pr != new_pr:
            actions.append(("add", new_pr))
        return Standing(winner=outcome.winner.to_record(), task_id=task, barren_rounds=0), actions

    barren = standing.barren_rounds + 1
    if barren >= BARREN_ROUNDS_BEFORE_ROTATION:
        return Standing(winner=standing.winner, task_id=next_task(task, available_tasks or []), barren_rounds=0), []
    # The incumbent keeps the label: no actions at all, so the hourly job does not re-notify them.
    return Standing(winner=standing.winner, task_id=task, barren_rounds=barren), []


def available_tasks(challenges: Path) -> list[str]:
    """Task ids with a published challenge packet, which is what a round can be opened over."""
    return sorted(p.stem for p in challenges.glob("*.json")) if challenges.is_dir() else []


def eligibility(contender: Contender) -> tuple[bool, str]:
    """Consume the score producer's decision, checking it against the same policy."""
    if len(contender.baseline.tokens) < 2:
        return False, "its baseline has fewer than two measurements"
    try:
        decision = acceptance_decision(contender.candidate, contender.baseline, contender.policy)
    except (ValueError, ScoreError) as exc:
        return False, str(exc)
    if contender.decision is not None and contender.decision != decision:
        return False, "scorecard decision disagrees with its policy and evidence"
    return decision.accepted, "; ".join(decision.reasons)


def score_of(contender: Contender) -> Score:
    """Rank by the lower bound of the reduction against this submission's own baseline.

    Relative, so tasks with different bars are comparable. Lower bound rather than point estimate,
    so a large margin measured on a noisy task does not outrank a smaller one that is actually
    known -- which is the same reason `decide` gates on the bound instead of the estimate.
    """
    interval = reduction_interval(contender.baseline.tokens, contender.candidate.tokens)
    return Score(contender=contender, lower_bound=interval[0], interval=interval)


def select(contenders: list[Contender]) -> Outcome:
    """Rank accepted entries; ties: tool calls, receipt time, miner, PR, round, task."""
    ranked: list[Score] = []
    ineligible: list[tuple[Contender, str]] = []
    for contender in contenders:
        ok, why = eligibility(contender)
        if ok:
            ranked.append(score_of(contender))
        else:
            ineligible.append((contender, why))
    ranked.sort(
        key=lambda s: (
            -s.lower_bound,
            s.contender.median_tool_calls,
            s.contender.received_at,
            s.contender.miner_id,
            s.contender.pr,
            s.contender.round_id,
            s.contender.task_id,
        )
    )
    ineligible.sort(key=lambda pair: (pair[0].task_id, pair[0].miner_id))
    return Outcome(winner=ranked[0] if ranked else None, ranked=ranked, ineligible=ineligible)


def close_actions(outcome: Outcome, standing: Standing | None = None) -> list[tuple[int, str]]:
    """(pr, reason) for every pull request this round is finished with.

    Every challenger closes each hour. Two pull requests are spared: this hour's winner, and the
    standing crown holder if nobody took it from them -- theirs is the current result and closing
    it would leave the label pointing at a closed page.
    """
    spared = {outcome.winner.contender.pr} if outcome.winner else set()
    if standing is not None and standing.pr:
        spared.add(standing.pr)
    reasons = {c.miner_id: "the round is over" for c in outcome.losers}
    for contender, why in outcome.ineligible:
        reasons[contender.miner_id] = why
    return [(c.pr, reasons.get(c.miner_id, "the round is over")) for c in outcome.losers if c.pr and c.pr not in spared]


def contenders_from(
    scorecard_dir: Path,
    *,
    store: Any = None,
    round_id: str | None = None,
    registry: Path | None = None,
) -> list[Contender]:
    """Read only the explicit active round and bind cards to its trusted admissions.

    Registry input is retained for caller compatibility; PR identity comes exclusively
    from authenticated admissions. Missing/invalid active evidence aborts settlement.
    Historical files are never participants and cannot cause PR actions.
    """
    from hermes.round import GRADED, SETTLED
    from validator.pr_admission import admission_for
    from validator.score import baseline_arm, policy_hash, schedule
    from validator.store import RoundStore

    if not round_id:
        raise CrownError("an explicit active round is required")
    store = store or RoundStore()
    window = store.load(round_id)
    if window.state not in (GRADED, SETTLED):
        raise CrownError("active round must be graded before crown selection")
    expected_baseline = baseline_arm(window.challenge, origin=store.identity)
    attempts = schedule(window.challenge)
    out: list[Contender] = []
    for miner_id in sorted(window.submissions):
        admission = admission_for(window, miner_id)
        path = scorecard_dir / f"{round_id}-{miner_id}.json"
        try:
            card = evidence_object(path.read_bytes())
            identity = {
                "epoch": window.challenge.epoch,
                "admission_id": admission["admission_id"],
                "bundle_sha256": admission["bundle_sha256"],
                "origin": store.identity,
            }
            policy = policy_record()
            if any(
                card.get(k) != v
                for k, v in {
                    "round_id": round_id,
                    "task_id": window.task_id,
                    "miner_id": miner_id,
                    "identity": identity,
                    "policy": policy,
                    "policy_hash": policy_hash(policy),
                }.items()
            ) or policy_hash(window.challenge.epoch.get("score_policy")) != policy_hash(policy):
                raise ValueError("scorecard identity or policy differs from active round")

            def arm(value: Any) -> Arm:
                n, passes = value["attempts"], value["verified_passes"]
                if type(n) is not int or n != len(attempts) or type(passes) is not int or not 0 <= passes <= n:
                    raise ValueError("invalid scorecard attempt counts")
                for key, minimum in (("tokens", 1), ("tool_calls", 0)):
                    if (
                        not isinstance(value[key], list)
                        or len(value[key]) != n
                        or any(type(v) is not int or v < minimum for v in value[key])
                    ):
                        raise ValueError("invalid scorecard measurements")
                return Arm(passes, n, tuple(value["tokens"]), tuple(value["tool_calls"]))

            candidate, baseline = arm(card["candidate"]), arm(card["baseline"])
            decision = acceptance_decision(candidate, baseline, card["policy"])
            verdict = window.verdicts.get(miner_id)
            if (
                baseline != expected_baseline
                or card["decision"] != decision.to_record()
                or card["decision"].get("accepted") is not decision.accepted
                or verdict is None
                or verdict.passed is not decision.accepted
                or card["reduction_interval"] != list(reduction_interval(baseline.tokens, candidate.tokens))
                or card.get("protocol_failures") != 0
            ):
                raise ValueError("scorecard disagrees with baseline, verdict or acceptance decision")
            out.append(
                Contender(
                    task_id=window.task_id,
                    miner_id=miner_id,
                    round_id=round_id,
                    candidate=candidate,
                    baseline=baseline,
                    pr=admission["pr_number"],
                    received_at=window.submissions[miner_id].received_at,
                    decision=decision,
                    policy=card["policy"],
                    evidence=card,
                )
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CrownError(f"invalid active scorecard for {miner_id}: {exc}") from exc
    return out


def pr_numbers(registry: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """(round_id, miner_id) -> {pr, received_at}, from the strategy registry.

    Optional on the record. A submission with no number is reported and not labelled or closed:
    guessing which pull request belongs to a miner is how the wrong one gets closed, and closing is
    not reversible by this job.
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}
    if not registry.is_file():
        return found
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("pr") or 0):
            found[(str(record.get("round_id") or ""), str(record.get("miner_id") or ""))] = {
                "pr": int(record["pr"]),
                "received_at": float(record.get("received_at") or 0.0),
            }
    return found


def render(outcome: Outcome) -> str:
    lines = [f"{len(outcome.ranked) + len(outcome.ineligible)} submission(s) evaluated this round", ""]
    if outcome.ranked:
        lines.append(f"  {'#':<3} {'miner':<14} {'task':<28} {'score':>8}  {'tokens':>9}  interval")
        for place, s in enumerate(outcome.ranked, start=1):
            c = s.contender
            lines.append(
                f"  {place:<3} {c.miner_id:<14} {c.task_id:<28} {s.lower_bound:>7.1%}  "
                f"{c.median_tokens:>9,.0f}  [{s.interval[0]:.1%}, {s.interval[1]:.1%}]"
            )
        lines.append("")
        lines.append("  score is the LOWER BOUND of the reduction against each submission's own baseline:")
        lines.append("  relative, so tasks with different bars compare, and noise-aware, so a large margin")
        lines.append("  on a variable task does not outrank a smaller one that is actually known.")
    for contender, why in outcome.ineligible:
        lines.append(f"  --  {contender.miner_id:<14} {contender.task_id:<28} {why[:70]}")

    lines.append("")
    if outcome.winner:
        w = outcome.winner
        lines.append(f"CROWN: {w.contender.miner_id} on {w.contender.task_id} (#{w.contender.pr or '?'})")
        lines.append(f"  {w.lower_bound:.1%} reduction, lower bound, over {w.contender.candidate.attempts} attempts")
    else:
        lines.append("NO CROWN this round. Nothing cleared the bar, and crowning the best of a bad")
        lines.append("field would make an hourly reward that always pays out and therefore says nothing.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from validator.settlement import crown_main

    return crown_main(argv)


__all__ = [
    "CROWNS",
    "CROWN_LABEL",
    "Contender",
    "CrownError",
    "Outcome",
    "Score",
    "close_actions",
    "contenders_from",
    "eligibility",
    "Standing",
    "available_tasks",
    "next_task",
    "settle",
    "main",
    "pr_numbers",
    "render",
    "score_of",
    "select",
]


if __name__ == "__main__":
    raise SystemExit(main())

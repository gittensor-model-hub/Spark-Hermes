"""Execution eligibility for crossed release, distinct from measured task success.

This reads complete runner/sink envelopes. It does not apply the competition
protocol-clean threshold or the training requirement to demonstrate tool use:
a naturally completed, verified failure (including a recovered malformed turn)
still contributes its actual outcome and every measured resource cost.
"""

from __future__ import annotations

import math
from typing import Any

from admin.artifacts import StageError, canonical
from admin.serving_identity import validate_completion_usage
from hermes.challenge import ChallengeError, episode_metrics_of
from hermes.cost import OPENAI, Usage, usage_from_provider
from hermes.state import ReasoningState
from hermes.trajectory import FINAL, STEP_KINDS, TOOL_CALL, TOOL_RESULT, AgentTrajectory


def _trajectory(raw: Any, metrics: dict[str, Any], task: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise StageError("missing complete executed trajectory")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise StageError("unsupported trajectory schema")
    if raw.get("task_id") != task["task_id"] or raw.get("task") != task["prompt"]:
        raise StageError("trajectory task differs from frozen task")
    if canonical(raw.get("tools_available", [])) != canonical(task["tools"]):
        raise StageError("trajectory tools differ from frozen task")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("executed") is not True:
        raise StageError("trajectory was not executed")
    if type(metadata.get("harness_final")) is not bool:
        raise StageError("missing trajectory completion evidence")
    if type(raw.get("success")) is not bool or raw["success"] is not metrics["success"]:
        raise StageError("trajectory success contradicts measured outcome")
    if type(raw.get("abstention", False)) is not bool or any(
        raw.get(key) is not None and not isinstance(raw[key], str) for key in ("system", "source")
    ):
        raise StageError("malformed trajectory attributes")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps or any(not isinstance(s, dict) for s in steps):
        raise StageError("trajectory has no complete executed steps")
    seen, pending = set(), set()
    for index, step in enumerate(steps):
        if step.get("kind") not in STEP_KINDS or ("content" in step and not isinstance(step["content"], str)):
            raise StageError("malformed trajectory step")
        if "state" in step and step["state"] is not None:
            state = step["state"]
            if not isinstance(state, dict) or any(
                key in state and not isinstance(state[key], str)
                for key in ("goal", "action", "hypothesis", "expected_signal", "observed_signal", "decision")
            ):
                raise StageError("malformed structured reasoning state")
            for key in ("known", "unknown"):
                if key in state and (
                    not isinstance(state[key], list) or any(not isinstance(item, str) for item in state[key])
                ):
                    raise StageError("malformed structured reasoning state facts")
            ReasoningState.from_record(state)
        if step["kind"] == FINAL and index != len(steps) - 1:
            raise StageError("trajectory continued after its final response")
        if step["kind"] in (TOOL_CALL, TOOL_RESULT):
            call_id = step.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise StageError("trajectory tool step lacks exact call identity")
            if step["kind"] == TOOL_CALL:
                if call_id in seen or not isinstance(step.get("args", {}), dict):
                    raise StageError("duplicate or malformed trajectory tool call")
                if step.get("tool") not in task["tools"]:
                    raise StageError("trajectory used an unavailable tool")
                seen.add(call_id)
                pending.add(call_id)
            else:
                if call_id not in pending or type(step.get("ok")) is not bool:
                    raise StageError("trajectory lacks paired observed tool results")
                pending.remove(call_id)
    if pending or steps[-1]["kind"] != FINAL or not steps[-1].get("content", "").strip():
        raise StageError("incomplete trajectory/final response")
    if metrics["steps"] != len(steps) or metrics["tool_calls"] != len(seen):
        raise StageError("trajectory steps/tool calls contradict measurements")
    # Parse the complete supported schema, including optional structured state.
    # Training's validate() also requires tool use, which is not release policy.
    parsed = AgentTrajectory.from_record(raw)
    if parsed.abstention and parsed.tool_calls:
        raise StageError("trajectory abstention contradicts executed tool calls")


def assess_episode(episode: dict[str, Any], *, task: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Retain explicit refusal reasons without rewriting or censoring outcomes."""
    reasons: list[str] = []
    try:
        metrics = episode_metrics_of(episode)
    except (ChallengeError, ValueError, TypeError, RecursionError) as exc:
        return {"eligible": False, "reasons": [f"invalid original episode evidence: {exc}"]}

    for key in (
        "public_passed",
        "success",
        "setup_failed",
        "max_steps_hit",
        "disqualified",
        "integrity_clean",
        "integrity_fully_checked",
        "protocol_clean",
        "hidden_passed",
    ):
        if type(metrics.get(key)) is not bool:
            reasons.append(f"missing or non-boolean {key}")
    for key in (
        "setup_failed",
        "max_steps_hit",
        "disqualified",
        "harness_final",
        "truncated",
        "integrity_disqualified",
    ):
        if key in metrics and metrics[key] is not False:
            reasons.append(f"execution {key}")
    for key in ("integrity_clean", "integrity_fully_checked"):
        if metrics.get(key) is not True:
            reasons.append(f"execution lacks {key}")
    if not isinstance(episode.get("integrity"), dict) or not isinstance(episode.get("verification"), dict):
        reasons.append("missing original integrity/verification reports")
    verification = episode.get("verification")
    if isinstance(verification, dict) and (
        any(type(verification.get(k)) is not bool for k in ("passed", "timed_out"))
        or "exit_code" not in verification
        or (verification["exit_code"] is not None and type(verification["exit_code"]) is not int)
        or any(not isinstance(verification.get(k), str) for k in ("stdout", "stderr"))
    ):
        reasons.append("incomplete original verifier report")
    for key in ("tokens_used", "tool_calls", "steps", "malformed_turns"):
        if type(metrics.get(key)) is not int or metrics[key] < 0:
            reasons.append(f"invalid measured integer {key}")
    malformed = metrics.get("malformed_turns")
    if type(malformed) is int and metrics.get("protocol_clean") is not (malformed == 0):
        reasons.append("protocol summary contradicts observed malformed turns")
    for key in ("wall_time_s", "cost"):
        value: Any = metrics.get(key)
        if key == "cost" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            reasons.append(f"invalid measured {key}")
    if all(type(metrics.get(k)) is bool for k in ("public_passed", "hidden_passed", "success", "disqualified")):
        verified = metrics["public_passed"] and metrics["hidden_passed"] and not metrics["disqualified"]
        if metrics["success"] is not verified:
            reasons.append("success contradicts required verifier/integrity evidence")
    try:
        _trajectory(episode.get("trajectory"), metrics, task)
    except (StageError, KeyError, TypeError, ValueError) as exc:
        reasons.append(f"invalid trajectory evidence: {exc}")
    # Per-response OUTPUT limits cannot be recovered from episode totals. Missing
    # originals fail closed; large prompts and cumulative valid outputs are allowed.
    responses = metrics.get("completion_usage")
    statuses = metrics.get("completion_status")
    if not isinstance(statuses, list) or not statuses:
        reasons.append("missing per-response completion status")
    else:
        if not isinstance(responses, list) or len(statuses) != len(responses):
            reasons.append("completion status count contradicts observed responses")
        for index, status in enumerate(statuses):
            if not isinstance(status, str) or status not in ("stop", "tool_calls"):
                reasons.append(f"response {index + 1} has incomplete/unsupported provider finish reason: {status!r}")
        if statuses[-1] != "stop":
            reasons.append("last provider response did not complete normally")
    if not isinstance(responses, list) or not responses:
        reasons.append("missing per-response measured completion usage")
    else:
        total = Usage()
        try:
            for usage in responses:
                validate_completion_usage(usage, max_tokens=budget["max_tokens"])
                total = total + usage_from_provider(usage, shape=OPENAI)
            if total.total != metrics.get("tokens_used") or canonical(total.to_record()) != canonical(
                metrics.get("usage")
            ):
                raise StageError("response usage contradicts episode resource totals")
        except (StageError, ValueError, TypeError, OverflowError) as exc:
            reasons.append(str(exc))
    return {"eligible": not reasons, "reasons": reasons}

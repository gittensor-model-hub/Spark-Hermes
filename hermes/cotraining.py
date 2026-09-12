"""Strict paired-family co-training statistics; reports alone grant no authority."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean
from typing import Any

from admin.artifacts import StageError, canonical, content_digest

CELLS = ("Q00", "Q10", "Q01", "Q11")
DEFAULT_POLICY = {
    "version": "spark-crossed-quality-v1",
    "metric": "verified_task_success",
    "unit": "fraction",
    "direction": "higher",
    "min_attempts": 10,
    "min_families": 6,
    "resamples": 2000,
    "confidence": 0.95,
    "seed": 20260912,
    "min_gain": 0.0,
    "family_regression_tolerance": 0.0,
    "max_token_ratio": 1.0,
    "max_latency_ratio": 1.0,
}


def validate_policy(policy: dict[str, Any]) -> str:
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_POLICY):
        raise StageError("quality policy has missing/unknown fields")
    configurable = {"min_gain", "family_regression_tolerance", "max_token_ratio", "max_latency_ratio"}
    for key, expected in DEFAULT_POLICY.items():
        value = policy[key]
        if key not in configurable and (type(value) is not type(expected) or value != expected):
            raise StageError(f"unsupported protocol field: {key}")
        if key in configurable:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise StageError(f"invalid threshold: {key}")
            if key.startswith("max_") and value <= 0:
                raise StageError(f"invalid resource ratio: {key}")
            if key in {"min_gain", "family_regression_tolerance"} and value > 1:
                raise StageError(f"invalid fraction threshold: {key}")
    return content_digest(policy)


def _quantile(values: list[float], q: float) -> float:
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def crossed_report(matrix: dict[str, Any], *, require_execution: bool = False) -> dict[str, Any]:
    """Validate exact paired cells, then bootstrap families (never individual attempts)."""
    policy = matrix["policy"]
    if validate_policy(policy) != matrix.get("policy_hash"):
        raise StageError("quality policy hash differs")
    cells = matrix.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(CELLS):
        raise StageError("all four cells are required")
    schedule = matrix.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise StageError("missing paired schedule")
    expected = {}
    attempts: dict[str, set[str]] = defaultdict(set)
    task_families = {}
    for item in schedule:
        if not isinstance(item, dict) or set(item) != {"task_id", "family_id", "attempt_id", "seed"}:
            raise StageError("incomplete schedule identity")
        if any(not isinstance(item[k], str) or not item[k] for k in ("task_id", "family_id", "attempt_id")):
            raise StageError("empty schedule identity")
        if type(item["seed"]) is not int or item["seed"] < 0:
            raise StageError("invalid attempt seed")
        key = (item["task_id"], item["attempt_id"])
        if key in expected or (item["task_id"], item["seed"]) in {(x["task_id"], x["seed"]) for x in expected.values()}:
            raise StageError("duplicate attempt/seed identity")
        expected[key] = item
        if item["task_id"] in task_families and task_families[item["task_id"]] != item["family_id"]:
            raise StageError("task has inconsistent family")
        task_families[item["task_id"]] = item["family_id"]
        attempts[item["task_id"]].add(item["attempt_id"])
    summaries = {}
    execution_reasons = []
    fixed = {k: matrix[k] for k in ("evaluator", "workload", "environment", "budget", "sampling", "policy_hash")}
    factors = matrix["factors"]
    for name in CELLS:
        cell = cells[name]
        if any(canonical(cell.get(k)) != canonical(v) for k, v in fixed.items()):
            raise StageError(f"{name}: changed evaluator/workload/environment/budget/sampling/policy")
        if cell.get("agent") != factors["agents"][int(name[1])] or cell.get("model") != factors["models"][int(name[2])]:
            raise StageError(f"{name}: wrong agent/model factors")
        rows = cell.get("rows")
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise StageError(f"{name}: incomplete attempts")
        seen = set()
        by_task = defaultdict(list)
        ineligible = []
        for row in rows:
            if type(row.get("seed")) is not int:
                raise StageError("attempt seed must retain its declared integer type")
            key = (row.get("task_id"), row.get("attempt_id"))
            if key in seen or key not in expected or any(row.get(k) != v for k, v in expected[key].items()):
                raise StageError(f"{name}: unpaired/duplicate attempt identity")
            seen.add(key)
            if type(row.get("success")) is not bool:
                raise StageError("verified task success must be boolean")
            for resource in ("tokens", "latency"):
                value = row.get(resource)
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise StageError("negative/nonfinite/missing measured resources")
            by_task[key[0]].append(row["success"])
            if require_execution or "execution" in row:
                execution = row.get("execution")
                if (
                    not isinstance(execution, dict)
                    or set(execution) != {"eligible", "reasons"}
                    or type(execution["eligible"]) is not bool
                    or not isinstance(execution["reasons"], list)
                    or any(not isinstance(r, str) or not r for r in execution["reasons"])
                    or execution["eligible"] is not (not execution["reasons"])
                ):
                    raise StageError(f"{name}: missing/contradictory execution eligibility")
                if not execution["eligible"]:
                    ineligible.append({**expected[key], "reasons": execution["reasons"]})
        tasks = {task: mean(values) for task, values in by_task.items()}
        families: dict[str, list[float]] = defaultdict(list)
        for task, value in tasks.items():
            families[task_families[task]].append(value)
        family_means = {family: mean(values) for family, values in sorted(families.items())}
        summaries[name] = {
            "tasks": tasks,
            "families": family_means,
            "quality": mean(family_means.values()),
            "tokens": mean(row["tokens"] for row in rows),
            "latency": mean(row["latency"] for row in rows),
        }
        if require_execution or any("execution" in row for row in rows):
            invalid_families = {r["family_id"] for r in ineligible}
            summaries[name]["execution"] = {
                "eligible_attempts": len(rows) - len(ineligible),
                "complete_families": sorted(set(family_means) - invalid_families),
                "ineligible_attempts": ineligible,
            }
            if ineligible:
                detail = sorted({reason for r in ineligible for reason in r["reasons"]})
                execution_reasons.append(f"{name}: ineligible execution ({'; '.join(detail)})")
    baseline, joint = summaries["Q00"], summaries["Q11"]
    deltas = [joint["families"][f] - baseline["families"][f] for f in baseline["families"]]
    rng = random.Random(policy["seed"])
    bootstrap = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(policy["resamples"]))
    tail = (1 - policy["confidence"]) / 2
    interval = [_quantile(bootstrap, tail), _quantile(bootstrap, 1 - tail)]
    reasons = execution_reasons
    if min(map(len, attempts.values())) < policy["min_attempts"]:
        reasons.append("insufficient attempts per task")
    if len(deltas) < policy["min_families"]:
        reasons.append("insufficient independent families")
    if interval[0] <= policy["min_gain"]:
        reasons.append("joint gain lower bound does not exceed min_gain")
    if any(delta < -policy["family_regression_tolerance"] for delta in deltas):
        reasons.append("per-family success regression")
    for resource, threshold in (("tokens", "max_token_ratio"), ("latency", "max_latency_ratio")):
        if baseline[resource] <= 0:
            reasons.append(f"{resource} baseline must be positive")
        elif joint[resource] > baseline[resource] * policy[threshold]:
            reasons.append(f"mean {resource} regression")
    q = {name: summaries[name]["quality"] for name in CELLS}
    return {
        "schema": "spark-crossed-report-v1",
        "policy": policy,
        "policy_hash": matrix["policy_hash"],
        "cells": summaries,
        "agent_effect": q["Q10"] - q["Q00"],
        "model_effect": q["Q01"] - q["Q00"],
        "interaction": q["Q11"] - q["Q10"] - q["Q01"] + q["Q00"],
        "joint_gain": q["Q11"] - q["Q00"],
        "joint_interval": interval,
        "eligible": not reasons,
        "reasons": reasons,
        "authorizes_activation": False,
    }

"""Deterministic requests for new verified experience from measured failures.

No generator, teacher or model is called. Requests are inputs to the existing task
generator and its verification gate; they never become positive training examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, content_digest, file_digest, read_record, write_record
from admin.replay import ReplayStore

DEFAULT_CONFIG = {"version": "spark-curriculum-v1", "count": 6, "min_families": 3, "max_per_family": 2}


def classify(row: dict[str, Any], *, private_required: bool) -> str:
    """Never confuse a broken harness or censored run with a measured learning failure."""
    from validator.aggregate import admit_episode
    from validator.score import normalize_episode

    try:
        metrics = normalize_episode(row)
        if metrics.get("setup_failed") is True:
            return "infrastructure"
        if metrics.get("max_steps_hit") is True or metrics.get("truncated") is True:
            return "truncation"
        episode = admit_episode(
            row, round_id="diagnostic", miner_id="operator", private_required=private_required, provenance={}
        )
    except (ValueError, RuntimeError, KeyError, TypeError):
        return "invalid_evidence"
    if episode.verified:
        return "success"
    if episode.overfit:
        return "withheld_generalization"
    if any(s.get("kind") == "tool_result" and s.get("ok") is False for s in episode.trajectory["steps"]):
        return "tool_recovery"
    return "task_correctness"


def select_requests(feedback: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if (
        set(config) != set(DEFAULT_CONFIG)
        or config["version"] != DEFAULT_CONFIG["version"]
        or any(type(config[k]) is not int or config[k] < 1 for k in config if k != "version")
    ):
        raise StageError("curriculum requires complete versioned positive-integer configuration")
    known = {
        "success",
        "task_correctness",
        "withheld_generalization",
        "tool_recovery",
        "infrastructure",
        "truncation",
        "invalid_evidence",
    }
    seen = set()
    groups: dict[str, list[dict[str, Any]]] = {}
    excluded: Counter[str] = Counter()
    for row in feedback:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("id", "family", "task_id", "version")):
            raise StageError("feedback requires exact task/family/version/evidence identity")
        if row["id"] in seen:
            raise StageError("duplicate curriculum feedback identity")
        seen.add(row["id"])
        if row.get("category") not in known:
            raise StageError("unknown measured feedback category")
        if row["category"] in {"infrastructure", "truncation", "invalid_evidence"}:
            excluded[row["category"]] += 1
            continue
        groups.setdefault(row["family"], []).append(row)
    if len(groups) < config["min_families"] or config["count"] < config["min_families"]:
        raise StageError("insufficient genuine family breadth for requested curriculum")
    if len(groups) * config["max_per_family"] < config["count"]:
        raise StageError("curriculum budget cannot meet per-family caps")
    stats = {}
    for family, rows in sorted(groups.items()):
        counts = Counter(row["category"] for row in rows)
        failures = len(rows) - counts["success"]
        categories = sorted((k for k in counts if k != "success"), key=lambda k: (-counts[k], k))
        stats[family] = {
            "attempts": len(rows),
            "failures": failures,
            "all_fail": failures == len(rows),
            "categories": categories,
            "counts": dict(sorted(counts.items())),
        }
    ranking = sorted(groups, key=lambda f: (-stats[f]["failures"] / stats[f]["attempts"], f))
    requests = []
    for turn in range(config["max_per_family"]):
        for family in ranking:
            if len(requests) == config["count"]:
                break
            stat = stats[family]
            categories = stat["categories"] or ["breadth"]
            request = {
                "family_id": family,
                "category": categories[turn % len(categories)],
                "action": "collect-new-verified-experience",
                "all_fail": stat["all_fail"],
                "ordinal": turn,
                "source_ids": sorted(r["id"] for r in groups[family]),
                "requires_execution_and_verification": True,
                "automatic_positive": False,
            }
            requests.append({"request_id": content_digest(request), **request})
    payload = {
        "schema": "spark-curriculum-v1",
        "config": config,
        "input_hash": content_digest(sorted(feedback, key=lambda r: r["id"])),
        "families": stats,
        "excluded": dict(sorted(excluded.items())),
        "requests": requests,
    }
    return {"id": content_digest(payload), **payload}


def build_curriculum(
    replay: ReplayStore,
    *,
    config: dict[str, Any],
    diagnostics: Path | None = None,
    identifiers: list[str] | None = None,
) -> dict[str, Any]:
    from admin.data_policy import DataPolicy
    from hermesbench.sink import decode_episodes

    settings = replay.configuration()
    policy = DataPolicy(Path(settings["policy"]), identity=replay.identity)
    if identifiers is not None and (not identifiers or len(set(identifiers)) != len(identifiers)):
        raise StageError("curriculum requires distinct explicit experience IDs")
    episodes, inputs = replay.experiences(identifiers)
    feedback = []
    for episode in episodes:
        p = episode.provenance
        category = (
            "success" if episode.verified else "withheld_generalization" if episode.overfit else "task_correctness"
        )
        if not episode.verified and any(
            s.get("kind") == "tool_result" and s.get("ok") is False for s in episode.trajectory["steps"]
        ):
            category = "tool_recovery"
        feedback.append(
            {
                "id": content_digest(p),
                "family": p["membership"]["canonical_family"],
                "task_id": episode.task_id,
                "version": p["task_version"],
                "category": category,
            }
        )
    diagnostic_sources = []
    if diagnostics:
        declaration = read_record(diagnostics)
        if declaration.get("origin") != replay.identity:
            raise StageError("diagnostic declaration must belong to configured replay operator")
        for source in declaration["logs"]:
            path = Path(source["path"])
            if file_digest(path) != source["sha256"]:
                raise StageError("diagnostic source changed")
            if type(source.get("private_required")) is not bool:
                raise StageError("diagnostics require explicit private-check declaration")
            for row in decode_episodes(path.read_bytes(), source=str(path)):
                # Diagnostics only identify broken infrastructure/censored data. Genuine
                # learning counts come exclusively from committed settled experience.
                category = classify(row, private_required=source["private_required"])
                if category not in {"infrastructure", "truncation", "invalid_evidence"}:
                    raise StageError("unsettled diagnostic logs cannot supply learning outcomes")
                task_id = row.get("task_id", row.get("metrics", {}).get("task_id"))
                member = policy.membership(
                    task_id=task_id,
                    repository=source["repository"],
                    version=source["versions"][task_id],
                    purpose="curriculum",
                )
                feedback.append(
                    {
                        "id": content_digest({"source": source, "row": row}),
                        "task_id": task_id,
                        "version": member["version"],
                        "family": member["canonical_family"],
                        "category": category,
                    }
                )
            diagnostic_sources.append(source)
    result = select_requests(feedback, config)
    payload = {
        **result,
        "origin": replay.identity,
        "settled_inputs": inputs,
        "diagnostic_sources": diagnostic_sources,
        "policy_hash": settings["policy_sha256"],
        "authorizes_model_promotion": False,
    }
    committed = replay.authority.put("curriculum", payload)
    return {**payload, "authority_id": committed["id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_curriculum(
            ReplayStore(args.replay_root), config=read_record(args.config), diagnostics=args.diagnostics
        )
        write_record(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"curriculum: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

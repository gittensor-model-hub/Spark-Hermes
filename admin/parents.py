"""Consume durable release decisions as SFT parent authority.

The future release gate is the sole production writer of `release-decision` records.
There is deliberately no production JSON-approval import or approval CLI. The labelled
fixture command writes only to an immutable fixture authority for CPU handoff tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from admin.artifacts import (
    AuthorityStore,
    StageError,
    checkpoint_files,
    content_digest,
    file_digest,
    read_record,
    same_domain,
)
from admin.runtime_protocol import candidate_writer


def merged_identity(record_path: Path, *, identity: dict[str, str]) -> dict[str, Any]:
    record = read_record(record_path)
    same_domain(identity, record.get("origin", {}))
    path = Path(record["merged"]).resolve()
    if Path(record["merged"]).is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
        raise StageError("approved parent cannot contain symlinked checkpoint artifacts")
    files = checkpoint_files(path)
    if files != record.get("files"):
        raise StageError("approved merged checkpoint files changed")
    if "tokenizer_config.json" not in files or not any(
        k in files for k in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
    ):
        raise StageError("approved parent requires complete tokenizer artifacts")
    recipe = Path(record.get("recipe", ""))
    if not recipe.is_file() or file_digest(recipe) != record.get("recipe_sha256"):
        raise StageError("merged checkpoint recipe provenance is missing or changed")
    if record.get("stage") != "sft":
        raise StageError("approved parent must be a merged SFT checkpoint")
    for key in ("profile", "base_model", "revision"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise StageError("merged parent profile/base/revision identity is incomplete")
    return {
        "record": str(record_path.resolve()),
        "record_sha256": file_digest(record_path),
        "merged": str(path),
        "files": files,
        "origin": record["origin"],
        **{k: record[k] for k in ("profile", "base_model", "revision", "recipe", "recipe_sha256", "stage")},
    }


class ParentAuthority:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.store = AuthorityStore(root, role="release", mode=mode, namespace=namespace)
        self.identity = self.store.identity

    def approved_parent(
        self, identifier: str, *, identity: dict[str, str], profile: str, repository: str, revision: str
    ) -> tuple[Path, dict[str, Any]]:
        same_domain(identity, self.identity)
        if self.store.kind(identifier) == "runtime-parent":
            from admin.release import ReleaseAuthority
            from admin.runtime_transition import baseline_parent

            decision = baseline_parent(ReleaseAuthority(self.store.root), identifier)
            candidate = decision["candidate"]
            if any(
                candidate[k] != v for k, v in (("profile", profile), ("base_model", repository), ("revision", revision))
            ):
                raise StageError("baseline parent belongs to another model profile/base/revision")
            return Path(candidate["merged"]), {
                "root": str(self.store.root),
                "identity": self.identity,
                "approval_id": identifier,
                "decision": decision,
            }
        issued = self.store.get(identifier, kind="release-decision")
        decision = issued["payload"]
        if decision.get("schema") != "spark-release-decision-v1" or decision.get("result") != "accepted":
            raise StageError("parent has no accepted release decision")
        if decision.get("origin") != self.identity:
            raise StageError("release decision issuer mismatch")
        if self.identity["mode"] == "production" and decision.get("fixture_only") is not False:
            raise StageError("production parent requires a production release decision")
        if decision.get("strict") is True:
            from admin.release import ReleaseAuthority

            ReleaseAuthority(self.store.root).resolve_decision(identifier)
        candidate = decision.get("candidate")
        if not isinstance(candidate, dict):
            raise StageError("release decision has no exact merged candidate identity")
        if candidate != merged_identity(Path(candidate["record"]), identity=self.identity):
            raise StageError("approved parent checkpoint/record changed")
        if any(
            candidate[k] != v for k, v in (("profile", profile), ("base_model", repository), ("revision", revision))
        ):
            raise StageError("approved parent belongs to another model profile/base/revision")
        if not isinstance(decision.get("agent"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", decision["agent"]):
            raise StageError("release must bind the exact agent digest")
        if (
            not isinstance(decision.get("policy"), dict)
            or not decision["policy"]
            or content_digest(decision["policy"]) != decision.get("policy_hash")
        ):
            raise StageError("release decision policy binding is missing or changed")
        evaluation = decision.get("evaluation", {})
        cells = evaluation.get("cells") if isinstance(evaluation, dict) else None
        if (
            not isinstance(cells, list)
            or not all(isinstance(c, str) for c in cells)
            or sorted(cells) != ["Q00", "Q01", "Q10", "Q11"]
        ):
            raise StageError("release requires a four-cell evaluation identity")
        if not isinstance(evaluation.get("id"), str) or not evaluation["id"]:
            raise StageError("release evaluation identity is missing")
        artifact = Path(evaluation.get("path", ""))
        if not artifact.is_file() or file_digest(artifact) != evaluation.get("sha256"):
            raise StageError("release evaluation artifact changed")
        same_domain(self.identity, evaluation.get("origin", {}))
        from admin.pipeline import Workspace, _require
        from admin.replay import verify_corpus

        data = decision.get("corpus", {})
        if not isinstance(data, dict) or not isinstance(data.get("workspace"), str):
            raise StageError("release corpus identity is missing")
        ws = Workspace(Path(data["workspace"]))
        same_domain(self.identity, ws.identity)
        corpus = _require(ws, "corpus")
        verify_corpus(ws, corpus)
        if corpus["authority"] != data.get("authority"):
            raise StageError("approved parent training corpus identity changed")
        return Path(candidate["merged"]), {
            "root": str(self.store.root),
            "identity": self.identity,
            "approval_id": identifier,
            "decision": decision,
        }

    @candidate_writer
    def fixture_approve(self, *, merged_record: Path, workspace: Any, agent: str, evaluation: Path) -> dict[str, Any]:
        if self.identity["mode"] != "fixture":
            raise StageError("fixture approval cannot be issued by a production authority")
        from admin.pipeline import _require
        from admin.replay import verify_corpus

        same_domain(self.identity, workspace.identity)
        corpus = _require(workspace, "corpus")
        verify_corpus(workspace, corpus)
        candidate = merged_identity(merged_record, identity=self.identity)
        policy = {"version": "cpu-parent-handoff-v1", "fixture_only": True}
        decision = {
            "schema": "spark-release-decision-v1",
            "origin": self.identity,
            "result": "accepted",
            "candidate": candidate,
            "agent": agent,
            "policy": policy,
            "policy_hash": content_digest(policy),
            "evaluation": {
                "id": "sha256:" + file_digest(evaluation),
                "path": str(evaluation.resolve()),
                "sha256": file_digest(evaluation),
                "origin": self.identity,
                "cells": ["Q00", "Q10", "Q01", "Q11"],
            },
            "corpus": {"workspace": str(workspace.root), "authority": corpus["authority"]},
            "fixture_only": True,
            "authorizes_production_promotion": False,
        }
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", agent):
            raise StageError("fixture agent must have an exact digest")
        return self.store.put("release-decision", decision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixture-approve",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--merged-record", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from admin.pipeline import Workspace

        authority = ParentAuthority(args.root, mode="fixture", namespace=args.namespace)
        record = authority.fixture_approve(
            merged_record=args.merged_record,
            workspace=Workspace(args.workspace),
            agent=args.agent,
            evaluation=args.evaluation,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"parents: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

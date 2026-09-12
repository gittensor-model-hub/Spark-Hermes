"""Content-bound candidates consuming original replay and prepared-training authority."""

from __future__ import annotations

import argparse
import json
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
from admin.parents import merged_identity
from admin.pipeline import Workspace, _require
from admin.replay import verify_corpus
from admin.runtime_protocol import candidate_runtime, candidate_writer, historical_checked
from admin.training import training_recipe
from hermes.harness import crossed_runtime_identity
from hermes.protocol import DIALECTS


def file_identity(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise StageError(f"missing or symlinked artifact: {path}")
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def checked_file(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if file_identity(path) != record:
        raise StageError("artifact bytes/path changed")
    return path


def bound_bytes(record: dict[str, Any]) -> bytes:
    """Capture once against the previously committed expectation, before parsing."""
    import hashlib

    path = Path(record["path"])
    if path.is_symlink() or str(path.resolve()) != record["path"]:
        raise StageError("artifact path changed")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise StageError("artifact changed at final byte capture")
    return raw


def bound_record(record: dict[str, Any]) -> dict[str, Any]:
    from hermes.evidence_json import evidence_object

    return evidence_object(bound_bytes(record))


def agent_record(path: Path) -> dict[str, Any]:
    record = read_record(path)
    if set(record) != {"schema", "system", "dialect", "tool_schemas", "native_tool_messages"}:
        raise StageError("agent artifact has missing/unknown fields")
    if record["schema"] != "spark-agent-v1" or record["dialect"] not in DIALECTS:
        raise StageError("unsupported agent schema/dialect")
    if not isinstance(record["system"], str) or not record["system"].strip():
        raise StageError("agent needs actual executed system prompt")
    if not isinstance(record["tool_schemas"], dict) or type(record["native_tool_messages"]) is not bool:
        raise StageError("agent tool/message configuration is incomplete")
    for name, schema in record["tool_schemas"].items():
        if not isinstance(schema, dict) or schema.get("name") != name or not isinstance(schema.get("parameters"), dict):
            raise StageError("agent requires concrete named tool schemas")
    return record


def check_initial_parent(parent: Path, prepared: dict[str, Any], *, identity: dict[str, str]) -> None:
    """Resolve the actual pinned Hub cache without assuming metadata survives config serialization."""
    if identity["mode"] == "fixture":
        config = read_record(parent / "config.json")
        if config.get("_commit_hash") != prepared["revision"] or config.get("_name_or_path") != prepared["base_model"]:
            raise StageError("fixture initial parent requires its explicit repository/revision binding")
        return
    from huggingface_hub import snapshot_download

    try:
        cached = Path(snapshot_download(prepared["base_model"], revision=prepared["revision"], local_files_only=True))
    except Exception as exc:
        raise StageError(
            "initial parent requires the actual pinned local Hub snapshot; no download was attempted"
        ) from exc
    if cached.name != prepared["revision"] or cached.resolve() != parent.resolve():
        raise StageError("initial parent is not the exact pinned Hub cache snapshot used by preparation")


class CandidateStore:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.store = AuthorityStore(root, role="candidate", mode=mode, namespace=namespace)
        self.identity = self.store.identity

    @candidate_writer
    def register(
        self, *, workspace: Workspace, merged_record: Path, agent: Path, workload: Path, parent: Path
    ) -> dict[str, Any]:
        same_domain(self.identity, workspace.identity)
        recipe = training_recipe(workspace, "sft")
        prepared_path = workspace.models / "sft/prepared.json"
        prepared = read_record(prepared_path)
        model = merged_identity(merged_record, identity=self.identity)
        if model["origin"] != workspace.identity or model["recipe"] != str(recipe.resolve()):
            raise StageError("candidate merged model does not come from this preparation")
        merged = read_record(merged_record)
        corpus = _require(workspace, "corpus")
        verify_corpus(workspace, corpus)
        if merged.get("corpus_authority") != corpus["authority"]:
            raise StageError("candidate merged model has wrong corpus authority")
        if any(model[k] != prepared[k] for k in ("profile", "base_model", "revision")):
            raise StageError("candidate model profile differs from training preparation")
        parent = parent.resolve()
        if prepared.get("parent") and Path(prepared["parent"]["decision"]["candidate"]["merged"]).resolve() != parent:
            raise StageError("candidate parent is not the approved training parent")
        if not prepared.get("parent"):
            check_initial_parent(parent, prepared, identity=self.identity)
        import yaml

        template = yaml.safe_load(recipe.read_text())["chat_template_jinja"]
        tokenizer = read_record(Path(model["merged"]) / "tokenizer_config.json")
        template_file = Path(model["merged"]) / "chat_template.jinja"
        if tokenizer.get("chat_template") != template and (
            not template_file.is_file() or template_file.read_text() != template
        ):
            raise StageError("candidate tokenizer/template differs from the actual prepared recipe")
        parent_files = checkpoint_files(parent)
        if "tokenizer_config.json" not in parent_files or not any(
            k in parent_files for k in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
        ):
            raise StageError("parent requires tokenizer identity")
        payload = {
            "schema": "spark-candidate-v1",
            "origin": self.identity,
            "workspace": str(workspace.root.resolve()),
            "workspace_identity": workspace.identity,
            "model": model,
            "model_id": content_digest({"representation": "merged-sft", "files": model["files"]}),
            "representation": "merged-sft",
            "agent": file_identity(agent),
            "agent_record": agent_record(agent),
            "agent_id": "sha256:" + file_digest(agent),
            "workload": file_identity(workload),
            "prepared": file_identity(prepared_path),
            "recipe": file_identity(recipe),
            "corpus": {"workspace": str(workspace.root.resolve()), "authority": corpus["authority"]},
            "parent": {"path": str(parent), "files": parent_files, "approval": prepared.get("parent")},
            "runtime": crossed_runtime_identity(),
        }
        return self.store.put("candidate", payload)

    @historical_checked
    def resolve(self, identifier: str) -> dict[str, Any]:
        if self.store.kind(identifier) == "runtime-baseline":
            from admin.runtime_transition import baseline_candidate

            return baseline_candidate(self.store, identifier)
        record = self.store.get(identifier, kind="candidate")
        candidate = record["payload"]
        if candidate.get("origin") != self.identity:
            raise StageError("candidate issuer changed")
        ws = Workspace(Path(candidate["workspace"]))
        same_domain(self.identity, ws.identity)
        if ws.identity != candidate["workspace_identity"]:
            raise StageError("candidate workspace issuer changed")
        recipe = training_recipe(ws, "sft")
        if str(recipe.resolve()) != candidate["recipe"]["path"]:
            raise StageError("candidate prepared recipe changed")
        for key in ("agent", "workload", "prepared", "recipe"):
            checked_file(candidate[key])
        if agent_record(Path(candidate["agent"]["path"])) != candidate["agent_record"]:
            raise StageError("candidate executed agent changed")
        if merged_identity(Path(candidate["model"]["record"]), identity=self.identity) != candidate["model"]:
            raise StageError("candidate model artifacts changed")
        if checkpoint_files(Path(candidate["parent"]["path"])) != candidate["parent"]["files"]:
            raise StageError("candidate parent artifacts changed")
        candidate_runtime(candidate, crossed_runtime_identity())
        return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "show"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", choices=("production", "fixture"))
    parser.add_argument("--namespace")
    parser.add_argument("--id")
    for field in ("workspace", "merged-record", "agent", "workload", "parent"):
        parser.add_argument("--" + field, type=Path)
    args = parser.parse_args(argv)
    try:
        store = CandidateStore(args.root, mode=args.mode, namespace=args.namespace)
        if args.command == "show":
            result = store.resolve(args.id)
        else:
            if any(getattr(args, k) is None for k in ("workspace", "merged_record", "agent", "workload", "parent")):
                raise StageError(
                    "register requires workspace, merged-record, agent, workload and actual parent checkpoint"
                )
            result = store.register(
                workspace=Workspace(args.workspace),
                merged_record=args.merged_record,
                agent=args.agent,
                workload=args.workload,
                parent=args.parent,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"candidates: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

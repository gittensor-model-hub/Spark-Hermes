"""Prepare and launch operator training from a verified, fingerprinted corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from admin.artifacts import (
    AuthorityStore,
    StageError,
    canonical,
    checkpoint_files,
    file_digest,
    read_record,
    write_record,
)
from admin.pipeline import Workspace, _require
from admin.runtime_protocol import check_workspace, preparation_writer, verification_call
from admin.split import refuse_eval_tasks
from hermes.base_model import load as load_base

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "hermes/recipes/spark-hermes-3.8-27b"


def merged_base(workspace: Workspace, *, profile: str, repository: str, revision: str) -> tuple[Path, dict[str, Any]]:
    path = workspace.models / "sft/adapter/merged"
    record_path = workspace.models / "sft/merged.json"
    if not record_path.is_file():
        raise StageError("DPO requires a recorded merged SFT checkpoint; run merge --training-stage sft first")
    record = read_record(record_path)
    if any(
        record.get(key) != value
        for key, value in (("profile", profile), ("base_model", repository), ("revision", revision))
    ):
        raise StageError("merged SFT checkpoint belongs to a different model profile or revision")
    if record.get("recipe_sha256") != file_digest(training_recipe(workspace, "sft")):
        raise StageError("SFT recipe changed since merging; the DPO reference policy is stale")
    if record.get("files") != checkpoint_files(path):
        raise StageError("merged SFT checkpoint changed since merging")
    return path, record


def _rows(path: Path, *, expected_digest: str | None = None) -> list[dict[str, Any]]:
    from hermesbench.sink import decode_episodes

    raw = path.read_bytes()
    if expected_digest is not None and hashlib.sha256(raw).hexdigest() != expected_digest:
        raise StageError("corpus changed at final training-row capture")
    rows = []
    for number, row in enumerate(decode_episodes(raw, source=str(path)), 1):
        if not isinstance(row, dict) or not row.get("task_id"):
            raise StageError(f"{path}:{number}: expected a row with task_id")
        rows.append(row)
    refuse_eval_tasks([r["task_id"] for r in rows])
    return rows


def render_preference(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Render whole trajectories under their shared initial prompt.

    Last-response chat loaders discard the actions we are comparing. Render before
    Axolotl loads the pair so reasoning, calls, schemas and observations all survive.
    This is trajectory-level DPO: tool-result tokens also contribute to its loss.
    """
    chosen, rejected = row["chosen"], row["rejected"]
    first = next((i for i, m in enumerate(chosen) if m["role"] == "assistant"), len(chosen))
    if not first or first == len(chosen) or chosen[:first] != rejected[:first]:
        raise StageError(f"{row['task_id']}: preference sides do not share an initial prompt")
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise StageError(f"{row['task_id']}: preference trajectory is incomplete")
    kwargs = {"tools": row.get("tools"), "tokenize": False, "add_generation_prompt": False}
    prompt = tokenizer.apply_chat_template(chosen[:first], **kwargs)
    sides = [tokenizer.apply_chat_template(messages, **kwargs) for messages in (chosen, rejected)]
    if not all(side.startswith(prompt) for side in sides):
        raise StageError("chat template changed the prompt while rendering the preference completion")
    if sides[0] == sides[1]:
        raise StageError(f"{row['task_id']}: preference sides render identically")
    return {
        "task_id": row["task_id"],
        "prompt": prompt,
        "chosen": sides[0][len(prompt) :],
        "rejected": sides[1][len(prompt) :],
    }


@preparation_writer
def prepare_training(
    workspace: Workspace,
    *,
    stage: str = "sft",
    sequence_len: int | None = None,
    profile: str = "bf16",
    max_steps: int | None = None,
    local_files_only: bool = False,
    tokenizer: Any = None,
    parent_approval: str | None = None,
    release_root: Path | None = None,
) -> dict[str, Any]:
    if profile not in {"bf16", "rtx5090-poc"}:
        raise StageError("unknown training profile")
    overrides = yaml.safe_load((RECIPES / "rtx5090-poc.yaml").read_text()) if profile == "rtx5090-poc" else {}
    sequence_len = sequence_len if sequence_len is not None else int(overrides.get("sequence_len", 8192))
    if max_steps is not None and max_steps < 1:
        raise StageError("max-steps must be positive")
    if stage not in {"sft", "dpo"} or sequence_len < 1:
        raise StageError("stage must be sft or dpo and sequence length must be positive")
    corpus = _require(workspace, "corpus")
    from admin.replay import verify_corpus

    verify_corpus(workspace, corpus)
    if tokenizer is not None and workspace.identity["mode"] != "fixture":
        raise StageError("injected tokenizer requires an immutable fixture workspace")
    source = workspace.corpus / ("sft.jsonl" if stage == "sft" else "preference.jsonl")
    if not source.is_file() or corpus.get("sha256", {}).get(source.name) != file_digest(source):
        raise StageError("corpus is missing or changed; rerun corpus before preparing training")
    rows = _rows(source, expected_digest=corpus["sha256"][source.name])
    if not rows:
        raise StageError(f"no {stage} examples; collect more verified rollouts")
    pin = load_base()
    repository = overrides.get("base_model", pin.repository)
    revision = overrides.get("base_model_revision", pin.revision)
    if any(row["task_id"] not in corpus.get("accepted", []) for row in rows):
        raise StageError("training row is outside the accepted corpus task set; rebuild corpus")
    directory = workspace.models / stage
    if list((directory / "adapter").glob("adapter_model.*")):
        raise StageError("this stage already has a trained adapter; prepare a new run to preserve its provenance")
    parent = (
        merged_base(workspace, profile=profile, repository=repository, revision=revision) if stage == "dpo" else None
    )
    if parent_approval is not None or release_root is not None:
        if stage != "sft" or not parent_approval or release_root is None:
            raise StageError("repeated SFT requires both parent approval ID and configured release root")
        from admin.parents import ParentAuthority

        parent = ParentAuthority(release_root).approved_parent(
            parent_approval, identity=workspace.identity, profile=profile, repository=repository, revision=revision
        )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = (
            AutoTokenizer.from_pretrained(str(parent[0]), trust_remote_code=False, local_files_only=True)
            if parent_approval and parent
            else AutoTokenizer.from_pretrained(
                repository, revision=revision, trust_remote_code=True, local_files_only=local_files_only
            )
        )
    # Both validation and training render the exact committed template.
    template = (
        ROOT
        / "hermes/templates"
        / ("chat-template-qwen35-4b.jinja" if profile == "rtx5090-poc" else "chat-template-qwen35.jinja")
    )
    tokenizer.chat_template = template.read_text(encoding="utf-8")
    lengths = []
    prepared_rows = []
    for row in rows:
        if stage == "sft":
            if row.get("metadata", {}).get("executed") is not True:
                raise StageError(f"{row['task_id']}: SFT requires an executed trajectory")
            messages = row.get("messages")
            if not messages or messages[-1].get("role") != "assistant":
                raise StageError(f"{row['task_id']}: SFT trajectory is incomplete")
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tools=row.get("tools"),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            ]
            prepared_rows.append(row)
        else:
            rendered = render_preference(row, tokenizer)
            texts = [rendered["prompt"] + rendered[side] for side in ("chosen", "rejected")]
            prepared_rows.append(rendered)
        length = max(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        lengths.append(length)
        if length > sequence_len:
            raise StageError(
                f"{row['task_id']}: {length} tokens exceeds sequence_len={sequence_len}; "
                "raise --sequence-len after measuring GPU headroom, or collect shorter complete trajectories"
            )
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "train.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prepared_rows), encoding="utf-8")
    recipe_name = "stage-c-tools.yaml" if stage == "sft" else "stage-d-preference.yaml"
    cfg = yaml.safe_load((RECIPES / recipe_name).read_text(encoding="utf-8"))
    cfg.update(overrides)
    if stage == "dpo":
        # Hardware overrides must not replace the preference objective's learning rate.
        cfg["learning_rate"] = 5.0e-6
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    cfg.pop("lora_model_dir", None)  # The operator's first SFT pass starts directly from the pin.
    if parent is not None:
        assert parent is not None
        cfg["base_model"] = str(parent[0])
        cfg.pop("base_model_revision", None)
    cfg["chat_template"] = "jinja"
    cfg["chat_template_jinja"] = template.read_text(encoding="utf-8")
    cfg["sequence_len"] = sequence_len
    cfg["sample_packing"] = False
    cfg["pad_to_sequence_len"] = False
    cfg["val_set_size"] = 0  # Evaluation uses separate tasks, never a random split of the same task's pairs.
    cfg.pop("evals_per_epoch", None)
    cfg["eval_strategy"] = "no"
    cfg["output_dir"] = str((directory / "adapter").resolve())
    cfg["datasets"] = [
        {
            "path": str(data.resolve()),
            "type": "chat_template",
            "field_messages": "messages",
            "field_tools": "tools",
            "roles_to_train": ["assistant"],
        }
    ]
    if stage == "dpo":
        cfg["datasets"] = [
            {
                "path": str(data.resolve()),
                "type": {
                    "field_prompt": "prompt",
                    "field_chosen": "chosen",
                    "field_rejected": "rejected",
                    "prompt_format": "{prompt}",
                    "chosen_format": "{chosen}",
                    "rejected_format": "{rejected}",
                },
            }
        ]
    recipe = directory / "source.yaml"
    cache_key = hashlib.sha256((file_digest(data) + json.dumps(cfg, sort_keys=True)).encode()).hexdigest()[:20]
    cfg["dataset_prepared_path"] = str((directory / "cache" / cache_key).resolve())
    recipe.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    from eval.train_prep import prepare_train_recipe

    result = prepare_train_recipe(
        recipe_path=recipe, distill_root=ROOT, out_path=directory / "train.yaml", enforce_canonical=False
    )
    report = {
        **result,
        "stage": stage,
        "rows": len(rows),
        "max_tokens": max(lengths),
        "source": str(source.resolve()),
        "source_sha256": file_digest(source),
        "data": str(data.resolve()),
        "data_sha256": file_digest(data),
        "recipe_sha256": file_digest(Path(result["prepared_recipe"])),
        "base_model": repository,
        "revision": revision,
        "profile": profile,
        "max_steps": max_steps,
        "parent": parent[1] if parent else None,
        "parent_approval": parent_approval,
        "origin": workspace.identity,
        "corpus_authority": corpus["authority"],
    }
    authority = AuthorityStore(
        workspace.root / "preparation-authority",
        role="preparation",
        mode=workspace.identity["mode"],
        namespace=workspace.identity["namespace"],
    )
    issued = authority.put("prepared-training", report)
    report["preparation_id"] = issued["id"]
    write_record(directory / "prepared.json", report)
    workspace.record("prepare", {"stage": stage, "summary": f"{stage}: {len(rows)} rows, max {max(lengths)} tokens"})
    return report


@verification_call
def training_recipe(workspace: Workspace, stage: str) -> Path:
    corpus = _require(workspace, "corpus")
    from admin.replay import verify_corpus

    verify_corpus(workspace, corpus)
    manifest = workspace.models / stage / "prepared.json"
    if not manifest.is_file():
        raise StageError(f"prepare --training-stage {stage} must run first")
    record = read_record(manifest)
    authority = AuthorityStore(
        workspace.root / "preparation-authority",
        role="preparation",
        mode=workspace.identity["mode"],
        namespace=workspace.identity["namespace"],
    )
    issued = authority.get(record.get("preparation_id", ""), kind="prepared-training")["payload"]
    if canonical({k: v for k, v in record.items() if k != "preparation_id"}) != canonical(issued):
        raise StageError("prepared training manifest changed from committed authority")
    if record.get("origin") != workspace.identity or record.get("corpus_authority") != corpus["authority"]:
        raise StageError("training namespace/corpus authority changed")
    for key, digest in (("source", "source_sha256"), ("data", "data_sha256"), ("prepared_recipe", "recipe_sha256")):
        if not isinstance(record.get(key), str) or not isinstance(record.get(digest), str):
            raise StageError(f"{manifest}: missing {key} or {digest}; rerun prepare")
        path = Path(record[key])
        if not path.is_file() or file_digest(path) != record[digest]:
            raise StageError(f"{key} changed after preparation; rerun prepare")
    if corpus.get("sha256", {}).get(Path(record["source"]).name) != record["source_sha256"]:
        raise StageError("prepared data no longer belongs to the recorded corpus")
    if stage == "dpo":
        _, parent = merged_base(
            workspace, profile=record["profile"], repository=record["base_model"], revision=record["revision"]
        )
        if parent != record.get("parent"):
            raise StageError("DPO reference checkpoint changed; rerun prepare")
    if record.get("parent_approval"):
        from admin.parents import ParentAuthority

        authority = ParentAuthority(Path(record["parent"]["root"]))
        _, current = authority.approved_parent(
            record["parent_approval"],
            identity=workspace.identity,
            profile=record["profile"],
            repository=record["base_model"],
            revision=record["revision"],
        )
        if current != record["parent"]:
            raise StageError("approved parent authority changed since preparation")
    return Path(record["prepared_recipe"])


def axolotl_command(action: str, recipe: Path) -> list[str]:
    executable = Path(sys.executable).parent / "axolotl"
    binary = str(executable) if executable.is_file() else shutil.which("axolotl")
    if not binary:
        raise StageError("Axolotl is not installed; run scripts/install_train.sh on the GPU host")
    return [binary, action, str(recipe)]


def train(workspace: Workspace, *, stage: str, dry_run: bool = False, fixture_executor: Any = None) -> dict[str, Any]:
    check_workspace(workspace)
    recipe = training_recipe(workspace, stage)
    if dry_run:
        return {"command": ["axolotl", "train", str(recipe)], "dry_run": True}
    if fixture_executor is not None and workspace.identity["mode"] != "fixture":
        raise StageError("fixture training executor requires an immutable fixture workspace")
    command = ["fixture-axolotl", "train", str(recipe)] if fixture_executor else axolotl_command("train", recipe)
    output = Path(yaml.safe_load(recipe.read_text())["output_dir"])
    if list(output.glob("adapter_model.*")):
        raise StageError("adapter already exists; use a new run root to preserve the checkpoint")
    done = (fixture_executor or subprocess.run)(command, cwd=ROOT)
    if done.returncode:
        raise StageError(f"training exited {done.returncode}; no completion recorded")
    from hermes.merge import Adapter, Recipe, check_base_agrees, check_shape_agrees, check_weights_exist

    adapter_recipe = Recipe.load(recipe)
    adapter = Adapter.load(output)
    issues = [
        *check_weights_exist(adapter),
        *check_base_agrees(adapter, adapter_recipe),
        *check_shape_agrees(adapter, adapter_recipe),
    ]
    if issues:
        raise StageError("trained adapter failed validation: " + "; ".join(issues))
    prepared = json.loads((workspace.models / stage / "prepared.json").read_text())
    return {
        "summary": f"{stage} adapter trained",
        "recipe": str(recipe),
        "adapter": str(output),
        "stage": stage,
        "profile": prepared.get("profile", "bf16"),
        "max_steps": prepared.get("max_steps"),
    }


def merge(workspace: Workspace, *, stage: str, fixture_executor: Any = None) -> dict[str, Any]:
    check_workspace(workspace)
    recipe = training_recipe(workspace, stage)
    config = yaml.safe_load(recipe.read_text())
    if config.get("load_in_4bit") or config.get("adapter") == "qlora":
        raise StageError(
            "QLoRA output is an adapter: evaluate it with the same 4-bit base; bf16 merging changes the evaluated model"
        )
    from hermes.merge import plan

    result = plan(recipe)
    if not result.ok:
        raise StageError("merge refused: " + "; ".join(result.issues))
    if fixture_executor is not None and workspace.identity["mode"] != "fixture":
        raise StageError("fixture merge executor requires an immutable fixture workspace")
    command = (
        ["fixture-axolotl", "merge-lora", str(recipe)] if fixture_executor else axolotl_command("merge-lora", recipe)
    ) + [f"--lora-model-dir={result.adapter.path}"]
    done = (fixture_executor or subprocess.run)(command)
    if done.returncode:
        raise StageError(f"merge exited {done.returncode}")
    prepared = read_record(workspace.models / stage / "prepared.json")
    path = result.recipe.merged_dir
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
        raise StageError("merged checkpoint cannot contain symlinked artifacts")
    record = {
        "merged": str(path),
        "files": checkpoint_files(path),
        "recipe_sha256": file_digest(recipe),
        "recipe": str(recipe.resolve()),
        "origin": workspace.identity,
        "stage": stage,
        "corpus_authority": prepared["corpus_authority"],
        **{key: prepared[key] for key in ("profile", "base_model", "revision")},
    }
    write_record(workspace.models / stage / "merged.json", record)
    workspace.record("merge", {"summary": f"{stage} merged", **record})
    return record


def doctor(workspace: Workspace, *, profile: str = "bf16", software_only: bool = False) -> dict[str, Any]:
    """Offline observations, separating CPU readiness from real deployment prerequisites."""
    from admin.pipeline import status
    from admin.readiness import prerequisites

    report = {**prerequisites(workspace, profile=profile), "pipeline": status(workspace)}
    if software_only:
        from admin.selfcheck import software_status

        software = software_status()
        report = {**report, **software, "production_blockers": report["blockers"]}
    return report

"""Offline prerequisite observations; fixture success never certifies production readiness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

from admin.artifacts import StageError, checkpoint_files, file_digest, read_record
from admin.pipeline import Workspace, withheld_environment
from admin.selfcheck import software_status
from admin.training import RECIPES, training_recipe
from hermes.base_model import load as load_base


def _identity(root: Path) -> dict[str, str] | None:
    """Observe identity without initializing an empty root as a side effect."""
    if not (root / ".identity").is_file():
        return None
    record = read_record(root / ".identity")
    if (
        set(record) != {"mode", "namespace", "issuer"}
        or record.get("mode") not in ("production", "fixture")
        or any(not isinstance(value, str) or not value for value in record.values())
    ):
        raise StageError("invalid workspace identity")
    return record


def profiles() -> dict[str, dict[str, str]]:
    from eval.hf_pin import check_revision

    base = load_base()
    poc = yaml.safe_load((RECIPES / "rtx5090-poc.yaml").read_text())
    if poc.get("base_model") != "Qwen/Qwen3.5-4B" or check_revision(poc.get("base_model_revision")):
        raise StageError("4B proof-of-concept recipe requires its pinned Qwen3.5-4B identity")
    return {
        "rtx5090-poc": {
            "repository": poc["base_model"],
            "revision": poc["base_model_revision"],
            "target_hardware": "RTX 5090 32 GB; real memory and compatibility measurement pending",
        },
        "bf16": {
            "repository": base.repository,
            "revision": base.revision,
            "target_hardware": "RTX PRO 6000 Blackwell 96 GB; real memory and compatibility measurement pending",
        },
    }


def _check(ready: bool, summary: str, **observations: Any) -> dict[str, Any]:
    return {"ready": ready, "summary": summary, **observations}


def prerequisites(workspace: Workspace, *, profile: str = "bf16") -> dict[str, Any]:
    """Inspect local artifacts with their production verifiers, without provider requests.

    Existing authorities are revalidated; absent roots/authorities are never initialized.
    Local files cannot establish live serving, actual optimizer execution, attestation
    or subnet enrollment. These are separately named pending external checks.
    """
    known_profiles = profiles()
    if profile not in known_profiles:
        raise StageError("unknown training profile")
    selected = known_profiles[profile]
    identity = _identity(workspace.root)
    fixture = identity is not None and identity["mode"] == "fixture"
    production = identity is not None and identity["mode"] == "production"
    software = software_status()
    checks: dict[str, dict[str, Any]] = {
        "software": _check(software["ready"], "CPU dependencies and packaged runtime assets", details=software),
    }
    corpus_valid = False
    corpus = workspace.manifest_of("corpus")
    corpus_summary = "no verified corpus; build reviewed licensed data with corpus/replay"
    if corpus is not None:
        try:
            from admin.replay import verify_corpus

            binding = corpus.get("authority")
            if identity is None or not isinstance(binding, dict) or not isinstance(binding.get("root"), str):
                raise StageError("corpus has no initialized workspace/authority binding")
            authority_root = Path(binding["root"])
            if _identity(authority_root) is None or not (authority_root / "authority.sqlite3").is_file():
                raise StageError("corpus authority store is absent")
            verify_corpus(workspace, corpus)
            corpus_valid = True
            corpus_summary = "rights, family/exposure metadata, source authority and corpus hashes verified"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            corpus_summary = f"corpus refused: {exc}"
    checks["corpus"] = _check(
        corpus_valid and production,
        corpus_summary + ("; fixture corpus cannot qualify real training" if fixture else ""),
        artifact_valid=corpus_valid,
        fixture=fixture,
    )
    private_valid = False
    private_summary = "original private evaluation checks and matching salt required"
    try:
        from hermesbench.tasks import load_suite

        tasks = load_suite("v0,v1")
        withheld_environment(tasks)
        private_valid = True
        private_summary = "packaged evaluation task commitments match configured private checks and salt"
    except (OSError, ValueError, RuntimeError) as exc:
        private_summary += f": {exc}"
    checks["private_checks"] = _check(
        private_valid and production,
        private_summary + ("; fixture namespace cannot certify private production evaluation" if fixture else ""),
        commitments_valid=private_valid,
    )
    prepared_valid = False
    prepared_summary = "prepare a verified corpus with the selected pinned profile"
    if corpus_valid and (workspace.root / "preparation-authority/authority.sqlite3").is_file():
        try:
            recipe = training_recipe(workspace, "sft")
            prepared = read_record(workspace.models / "sft/prepared.json")
            if (
                prepared.get("profile") != profile
                or prepared.get("base_model") != selected["repository"]
                or prepared.get("revision") != selected["revision"]
            ):
                raise StageError("prepared recipe belongs to a different selected model profile")
            prepared_valid = True
            prepared_summary = f"committed SFT recipe and parent/input hashes verified: {recipe}"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            prepared_summary = f"preparation refused: {exc}"
    checks["prepared_training"] = _check(
        prepared_valid and production,
        prepared_summary + ("; fixture tokenizer/inputs do not validate real training" if fixture else ""),
        artifact_valid=prepared_valid,
    )
    model_valid = False
    model_summary = "no verified merged SFT artifact; execute and record real training/merge on the training host"
    merged_path = workspace.models / "sft/merged.json"
    if prepared_valid and merged_path.is_file():
        try:
            merged = read_record(merged_path)
            path = workspace.models / "sft/adapter/merged"
            if (
                merged.get("origin") != identity
                or merged.get("profile") != profile
                or merged.get("base_model") != selected["repository"]
                or merged.get("revision") != selected["revision"]
                or merged.get("files") != checkpoint_files(path)
                or merged.get("recipe_sha256") != file_digest(training_recipe(workspace, "sft"))
                or merged.get("corpus_authority") != (corpus or {}).get("authority")
            ):
                raise StageError("merged artifact identity, recipe or checkpoint bytes changed")
            model_valid = True
            model_summary = "merged artifact hashes verified; local files alone do not prove real optimizer execution"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            model_summary = f"merged artifact refused: {exc}"
    checks["real_training"] = _check(
        False,
        model_summary + ("; CPU fixture weights are not a trained model" if fixture else ""),
        artifact_valid=model_valid,
        real_training_verified=False,
    )
    checks["trusted_serving"] = _check(
        False,
        "live HTTPS identity handshake and per-completion binding to exact model/deployment required; "
        "an OpenAI model alias or serving JSON alone is insufficient",
        checked_online=False,
    )
    training_dependencies = {
        name: importlib.util.find_spec(name) is not None for name in ("torch", "axolotl", "transformers")
    }
    checks["hardware"] = _check(
        False,
        selected["target_hardware"] + "; CPU software checks do not validate training or serving hardware",
        dependencies=training_dependencies,
        gpu_execution_performed=False,
    )
    checks["attestation"] = _check(
        False,
        "optional confidential deployment requires authenticated GPU/TDX evidence bound to the run "
        "and an independently approved guest measurement; no approved measurement is pinned here",
        required_for="confidential deployment claims",
    )
    checks["sn74"] = _check(
        False,
        "external repository approval, read-only Gittensor App, miner identity/eligibility, merged PRs "
        "and subnet-controlled reward configuration required; local labels/outbox do not authorize payout",
        registration_verified=False,
        payout_verified=False,
    )
    return {
        "ready": False,
        "scope": "production-prerequisites",
        "origin": identity,
        "mode": identity["mode"] if identity else "uninitialized (production default)",
        "profile": profile,
        "base_model": selected["repository"],
        "revision": selected["revision"],
        "profiles": known_profiles,
        "prerequisites": checks,
        "blockers": [f"{name}: {check['summary']}" for name, check in checks.items() if not check["ready"]],
        "model_training_executed": False,
        "external_requests_performed": False,
    }

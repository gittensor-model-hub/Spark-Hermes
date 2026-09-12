"""Learning boundaries exercised on real settled CPU trajectories and CLI artifacts."""

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml
from learning_support import NAMESPACE, cli, policy_config, produce_round

from admin.artifacts import (
    AuthorityStore,
    StageError,
    checkpoint_files,
    file_digest,
    read_record,
    write_record,
)
from admin.curriculum import DEFAULT_CONFIG, build_curriculum, classify, select_requests
from admin.data_policy import DataPolicy
from admin.parents import ParentAuthority
from admin.pipeline import Workspace
from admin.replay import DEFAULT_MIXTURE, ReplayStore, verify_corpus
from admin.training import prepare_training, training_recipe
from validator.aggregate import admit_episode
from validator.persistence import state_identity


def setup_replay(tmp_path, *, second=False, **kwargs):
    a = produce_round(tmp_path / "source", **kwargs)
    a["family"] = "counter"
    produced = [a]
    if second:
        b = produce_round(tmp_path / "source", "r-2")
        b["family"] = "counter"
        produced.append(b)
    config, _ = policy_config(tmp_path, produced)
    replay = ReplayStore(tmp_path / "replay", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    return replay, produced


def fixture_workspace(path):
    state_identity(path, mode="fixture", namespace=NAMESPACE)
    return Workspace(path)


def fixture_merge(ws, prepared):
    """Labelled byte stand-in exercises shard/tokenizer identity, never an optimizer."""
    merged = ws.models / "sft/adapter/merged"
    merged.mkdir(parents=True)
    write_record(merged / "config.json", {"model_type": "qwen3_5", "fixture_only": True})
    write_record(
        merged / "model.safetensors.index.json",
        {"weight_map": {"layer.one": "part-1.safetensors", "layer.two": "part-2.safetensors"}},
    )
    (merged / "part-1.safetensors").write_bytes(b"CPU_FIXTURE_NOT_MODEL_WEIGHTS_1")
    (merged / "part-2.safetensors").write_bytes(b"CPU_FIXTURE_NOT_MODEL_WEIGHTS_2")
    write_record(merged / "tokenizer_config.json", {"fixture_only": True})
    write_record(merged / "tokenizer.json", {"fixture_only": True})
    record = {
        "merged": str(merged),
        "files": checkpoint_files(merged),
        "origin": ws.identity,
        "stage": "sft",
        "recipe": prepared["prepared_recipe"],
        "recipe_sha256": prepared["recipe_sha256"],
        **{key: prepared[key] for key in ("profile", "base_model", "revision")},
    }
    path = ws.models / "sft/merged.json"
    write_record(path, record)
    return path


def test_two_round_cli_replay_and_repeated_sft(tmp_path):
    a = produce_round(tmp_path / "source")
    b = produce_round(tmp_path / "source", "r-2")
    a["family"] = b["family"] = "counter"
    policy_config(tmp_path, [a, b])
    root, first, second = tmp_path / "replay", tmp_path / "cycle-1", tmp_path / "cycle-2"
    cli(
        tmp_path,
        "admin.replay",
        "init",
        "--root",
        root,
        "--mode",
        "fixture",
        "--namespace",
        NAMESPACE,
        "--config",
        tmp_path / "config.json",
    )
    cli(tmp_path, "admin.replay", "import", "--root", root, "--source", "source", "--round", "r-1")
    cli(
        tmp_path,
        "admin.replay",
        "freeze",
        "--root",
        root,
        "--workspace",
        first,
        "--mode",
        "fixture",
        "--namespace",
        NAMESPACE,
    )
    first_report = json.loads(
        cli(
            tmp_path,
            "admin.cli",
            "prepare",
            "--root",
            first,
            "--profile",
            "rtx5090-poc",
            "--fixture-tokenizer",
            "--sequence-len",
            "65536",
        ).stdout
    )
    cfg = yaml.safe_load(Path(first_report["prepared_recipe"]).read_text())
    assert cfg["base_model"] == "Qwen/Qwen3.5-4B" and len(cfg["base_model_revision"]) == 40
    merged = fixture_merge(Workspace(first), first_report)
    evaluation = tmp_path / "fixture-evaluation.json"
    write_record(
        evaluation, {"fixture_only": True, "cells": ["Q00", "Q10", "Q01", "Q11"], "not_measured_learning": True}
    )
    approval = json.loads(
        cli(
            tmp_path,
            "admin.parents",
            "fixture-approve",
            "--root",
            tmp_path / "release",
            "--namespace",
            NAMESPACE,
            "--workspace",
            first,
            "--merged-record",
            merged,
            "--agent",
            a["admission"]["bundle_sha256"],
            "--evaluation",
            evaluation,
        ).stdout
    )
    assert approval["issuer"]["mode"] == "fixture"
    cli(
        tmp_path,
        "validator.aggregate",
        "--replay-root",
        root,
        "--source",
        "source",
        "--round",
        "r-2",
        "--out",
        second,
        "--mode",
        "fixture",
        "--namespace",
        NAMESPACE,
    )
    ws = Workspace(second)
    manifest = ws.manifest_of("corpus")
    assert manifest["sft_rows"] == 1 and manifest["preference_pairs"] == 1
    assert manifest["selection"]["sft"]["duplicates"] == 9
    sft = json.loads((ws.corpus / "sft.jsonl").read_text())
    assert len(sft["provenance"]) == 10
    assert {p["round_id"] for p in sft["provenance"]} == {"r-1", "r-2"}
    assert all(p["rights"] and p["evaluator"] and p["receipt_origin"] != p["origin"] for p in sft["provenance"])
    original = (ws.corpus / "stage.json").read_bytes()
    cli(tmp_path, "admin.cli", "replay", "freeze", "--root", root, "--workspace", second)
    assert (ws.corpus / "stage.json").read_bytes() == original
    cli(tmp_path, "admin.replay", "verify", "--root", root, "--workspace", second)
    report = json.loads(
        cli(
            tmp_path,
            "admin.cli",
            "prepare",
            "--root",
            second,
            "--profile",
            "rtx5090-poc",
            "--fixture-tokenizer",
            "--sequence-len",
            "65536",
            "--release-root",
            tmp_path / "release",
            "--parent-approval",
            approval["id"],
        ).stdout
    )
    cfg = yaml.safe_load(Path(report["prepared_recipe"]).read_text())
    assert cfg["base_model"] == read_record(merged)["merged"] and "base_model_revision" not in cfg
    assert report["parent"]["decision"]["candidate"]["files"] == checkpoint_files(Path(cfg["base_model"]))
    cli(tmp_path, "admin.cli", "train", "--root", second, "--dry-run")
    cli(
        tmp_path,
        "admin.cli",
        "prepare",
        "--root",
        second,
        "--profile",
        "bf16",
        "--fixture-tokenizer",
        "--release-root",
        tmp_path / "release",
        "--parent-approval",
        approval["id"],
        expected=2,
    )
    (Path(cfg["base_model"]) / "part-2.safetensors").write_bytes(b"changed CPU fixture")
    cli(tmp_path, "admin.cli", "train", "--root", second, "--dry-run", expected=2)
    assert all(x["status"] == "pending" for x in a["settlement"].actions())


@pytest.mark.parametrize(
    "change",
    ["log", "receipt", "round-origin", "policy", "rights", "task-rights", "family", "source-role", "uncommitted"],
)
def test_ingress_refuses_changed_or_unauthorized_sources(tmp_path, change):
    replay, (a,) = setup_replay(tmp_path)
    if change == "log":
        path = Path(a["record"]["entries"][0]["episodes"]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "receipt":
        (a["intake"].bundle_dir(a["intake"].read_receipts()[0]) / "SOUL.md").write_text("mutation proposal only")
    elif change == "round-origin":
        path = a["store"].path_for("r-1")
        row = read_record(path)
        del row["store_identity"]
        write_record(path, row)
    elif change in {"policy", "rights", "task-rights", "family"}:
        path = tmp_path / "policy.json"
        p = read_record(path)
        if change == "rights":
            p["rights"] = []
        elif change == "task-rights":
            p["rights"][0]["derivatives"] = False
        elif change == "family":
            p["memberships"][0]["partition"] = "sealed-release"
        else:
            p["version"] = "changed"
        write_record(path, p)
        # Both frozen-policy mutation and freshly configured disallowed policies refuse.
        if change != "policy":
            replay = ReplayStore(tmp_path / "other-replay", mode="fixture", namespace=NAMESPACE)
            replay.configure(read_record(tmp_path / "config.json"))
    elif change == "source-role":
        path = a["intake"].root / ".identity"
        write_record(path, a["store"].identity)
    else:
        with a["settlement"].transaction() as db:
            db.execute("DELETE FROM outbox")
            db.execute("DELETE FROM rounds")
    with pytest.raises((StageError, ValueError, RuntimeError)):
        replay.import_round("source", "r-1")
    assert not replay.authority.records(kind="settled-experience")


@pytest.mark.parametrize("poison", ["missing", "unexecuted", "unpaired", "counts", "final", "mutation"])
def test_settled_metrics_never_substitute_for_trajectory(tmp_path, poison):
    def change(rows):
        for row in rows:
            if poison == "missing":
                row.pop("trajectory")
            elif poison == "unexecuted":
                # Absence remains scoreable metrics, but cannot claim learning execution.
                row["trajectory"]["metadata"].pop("executed")
            elif poison == "unpaired":
                row["trajectory"]["steps"][1]["call_id"] = "unknown"
            elif poison == "counts":
                row["trajectory"]["steps"].insert(0, {"kind": "thinking", "content": "extra"})
            elif poison == "final":
                row["trajectory"]["steps"][-1]["kind"] = "thinking"
            else:
                row["trajectory"]["steps"] = [{"kind": "final", "content": "Suggested mutation; never executed"}]

    replay, _ = setup_replay(tmp_path, poison=change)
    with pytest.raises((StageError, ValueError, RuntimeError)):
        replay.import_round("source", "r-1")


def test_all_fail_creates_no_positives_and_requests_new_experience(tmp_path):
    a = produce_round(tmp_path / "source", successes=0)
    b = produce_round(tmp_path / "source", "r-2", task_id="gen-breadth", successes=5)
    config, _ = policy_config(tmp_path, [a, b])
    replay = ReplayStore(tmp_path / "replay", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    replay.import_round("source", "r-1")
    manifest = replay.freeze(fixture_workspace(tmp_path / "empty"))
    assert manifest["sft_rows"] == manifest["preference_pairs"] == 0
    replay.import_round("source", "r-2")
    config = {**DEFAULT_CONFIG, "count": 4, "min_families": 2}
    report = build_curriculum(replay, config=config)
    assert len(report["requests"]) == 4
    assert {r["family_id"] for r in report["requests"]} == {"family-0", "family-1"}
    assert report["families"]["family-0"]["all_fail"]
    assert all(r["automatic_positive"] is False for r in report["requests"])
    assert report == build_curriculum(replay, config=config)
    write_record(tmp_path / "curriculum.json", config)
    cli(
        tmp_path,
        "admin.curriculum",
        "--replay-root",
        replay.root,
        "--config",
        tmp_path / "curriculum.json",
        "--out",
        tmp_path / "requests.json",
    )


def test_replay_rechecks_history_before_preparation(tmp_path):
    replay, _ = setup_replay(tmp_path, second=True)
    replay.import_round("source", "r-1")
    replay.import_round("source", "r-2")
    ws = fixture_workspace(tmp_path / "workspace")
    manifest = replay.freeze(ws)
    verify_corpus(ws, manifest)
    path = tmp_path / "source/episodes/r-1/alice.jsonl"
    path.write_bytes(path.read_bytes().replace(b"CPU fixture complete.", b"copied unexecuted answer"))
    with pytest.raises(StageError, match="source bytes changed"):
        prepare_training(ws)


def test_fixture_cannot_cross_namespace_or_gain_parent_authority(tmp_path):
    replay, _ = setup_replay(tmp_path)
    replay.import_round("source", "r-1")
    with pytest.raises(StageError, match="mode/namespace"):
        replay.freeze(Workspace(tmp_path / "production"))
    ws = fixture_workspace(tmp_path / "fixture")
    manifest = replay.freeze(ws)
    production = Workspace(tmp_path / "production")
    shutil.copytree(ws.corpus, production.corpus)
    with pytest.raises(StageError):
        verify_corpus(production, manifest)
    authority = ParentAuthority(tmp_path / "release", mode="fixture", namespace=NAMESPACE)
    write_record(tmp_path / "approval.json", {"approved": True, "id": "pretend"})
    with pytest.raises(StageError, match="no committed authority"):
        authority.approved_parent(
            "pretend", identity=ws.identity, profile="rtx5090-poc", repository="Qwen/Qwen3.5-4B", revision="a" * 40
        )
    with pytest.raises((StageError, ValueError), match="immutable"):
        ParentAuthority(authority.store.root, mode="production")
    with pytest.raises(StageError, match="role/issuer"):
        AuthorityStore(replay.root, role="release")


def test_family_aliases_and_release_exposure(tmp_path):
    replay, (a,) = setup_replay(tmp_path)
    raw = read_record(tmp_path / "policy.json")
    member = raw["memberships"][0]
    alias = {
        **member,
        "task_id": "reworded-unrelated-id",
        "repository": "elsewhere/fork",
        "version": "another-version",
        "family_id": "counter-fork",
        "partition": "sealed-release",
        "exposure": [],
    }
    raw["family_aliases"]["counter-fork"] = "counter"
    raw["memberships"].append(alias)
    write_record(tmp_path / "policy.json", raw)
    p = DataPolicy(tmp_path / "policy.json", identity=replay.identity)
    with pytest.raises(StageError, match="sealed family"):
        p.membership(task_id=a["task"].task_id, repository="example/spark", version=a["version"], purpose="training")
    with pytest.raises(StageError, match="fresh release"):
        p.membership(
            task_id=alias["task_id"], repository=alias["repository"], version=alias["version"], purpose="release"
        )
    with pytest.raises(StageError, match="missing exact"):
        p.membership(task_id="renamed", repository=alias["repository"], version=alias["version"], purpose="training")
    raw["memberships"] = [alias]
    write_record(tmp_path / "policy.json", raw)
    p = DataPolicy(tmp_path / "policy.json", identity=replay.identity)
    assert p.membership(
        task_id=alias["task_id"], repository=alias["repository"], version=alias["version"], purpose="release"
    )
    for exposure in ("public", "disclosed", "selection", "training"):
        raw["memberships"][0]["exposure"] = [exposure]
        write_record(tmp_path / "policy.json", raw)
        with pytest.raises(StageError, match="fresh release"):
            DataPolicy(tmp_path / "policy.json", identity=replay.identity).membership(
                task_id=alias["task_id"], repository=alias["repository"], version=alias["version"], purpose="release"
            )


def test_caps_and_feedback_failure_classes(tmp_path):
    from test_admin_training import episode

    from admin.replay import _dedup_cap

    good = episode()
    assert classify(good, private_required=True) == "success"
    for key, category in (("setup_failed", "infrastructure"), ("max_steps_hit", "truncation")):
        row = copy.deepcopy(good)
        row["metrics"][key] = True
        assert classify(row, private_required=True) == category
        with pytest.raises((ValueError, RuntimeError)):
            admit_episode(row, round_id="r", miner_id="m", private_required=True, provenance={})
    incomplete = copy.deepcopy(good)
    del incomplete["metrics"]["integrity_clean"]
    assert classify(incomplete, private_required=True) == "invalid_evidence"
    feedback = [
        {"id": str(i), "family": f"f{i % 3}", "task_id": f"task{i}", "version": "v1", "category": c}
        for i, c in enumerate(
            ["task_correctness", "success", "withheld_generalization", "infrastructure", "truncation"]
        )
    ]
    r = select_requests(feedback, DEFAULT_CONFIG)
    assert r == select_requests(list(reversed(feedback)), DEFAULT_CONFIG)
    assert r["excluded"] == {"infrastructure": 1, "truncation": 1}
    rows = [
        {
            "messages": [{"role": "assistant", "content": str(i)}],
            "provenance": [
                {"membership": {"canonical_family": "one" if i < 3 else "two"}, "miner_id": "alice" if i < 4 else "bob"}
            ],
        }
        for i in range(6)
    ]
    kept, stats = _dedup_cap(
        rows, kind="sft", config={**DEFAULT_MIXTURE, "max_sft_per_family": 2, "max_sft_per_miner": 2}
    )
    for key in ("family", "miner"):
        counts = {}
        for row in kept:
            p = row["provenance"][0]
            label = p["miner_id"] if key == "miner" else p["membership"]["canonical_family"]
            counts[label] = counts.get(label, 0) + 1
        assert max(counts.values()) <= 2
    assert stats["capped"] > 0


@pytest.mark.parametrize(
    "change",
    [
        "shard",
        "tokenizer",
        "empty",
        "missing-approval",
        "wrong-issuer",
        "namespace",
        "origin",
        "recipe",
        "evaluation",
        "strip-parent",
        "rewrite-profile",
    ],
)
def test_parent_and_launch_revalidate_durable_authority(tmp_path, change):
    from admin.selfcheck import FixtureTokenizer

    replay, (a,) = setup_replay(tmp_path)
    replay.import_round("source", "r-1")
    ws1 = fixture_workspace(tmp_path / "first")
    replay.freeze(ws1)
    first = prepare_training(ws1, profile="rtx5090-poc", tokenizer=FixtureTokenizer(), sequence_len=65536)
    merged_record = fixture_merge(ws1, first)
    evaluation = tmp_path / "evaluation.json"
    write_record(evaluation, {"fixture_only": True})
    authority = ParentAuthority(tmp_path / "release", mode="fixture", namespace=NAMESPACE)
    approval = authority.fixture_approve(
        merged_record=merged_record, workspace=ws1, agent=a["admission"]["bundle_sha256"], evaluation=evaluation
    )
    ws2 = fixture_workspace(tmp_path / "second")
    replay.freeze(ws2)
    report = prepare_training(
        ws2,
        profile="rtx5090-poc",
        tokenizer=FixtureTokenizer(),
        sequence_len=65536,
        parent_approval=approval["id"],
        release_root=authority.store.root,
    )
    merged = Path(read_record(merged_record)["merged"])
    if change == "shard":
        (merged / "part-2.safetensors").unlink()
    elif change == "tokenizer":
        (merged / "tokenizer.json").write_text('{"changed":true}')
    elif change == "empty":
        (merged / "part-1.safetensors").write_bytes(b"")
    elif change == "missing-approval":
        with authority.store.connect() as db:
            db.execute("DELETE FROM records")
    elif change == "wrong-issuer":
        (authority.store.root / ".identity").unlink()
    elif change == "namespace":
        identity = ws2.identity
        write_record(ws2.root / ".identity", {**identity, "namespace": "foreign"})
    elif change == "origin":
        record = read_record(merged_record)
        del record["origin"]
        write_record(merged_record, record)
    elif change == "recipe":
        Path(first["prepared_recipe"]).write_text("changed recipe")
    elif change == "evaluation":
        evaluation.write_text('{"fixture_only":false}')
    else:
        path = ws2.models / "sft/prepared.json"
        record = read_record(path)
        if change == "strip-parent":
            record.pop("parent_approval")
            record.pop("parent")
        else:
            record["profile"] = "bf16"
        write_record(path, record)
    with pytest.raises((StageError, ValueError, OSError, RuntimeError)):
        training_recipe(ws2, "sft")
    assert report["parent_approval"] == approval["id"]


def test_committed_settlement_survives_projection_crash(tmp_path):
    replay, (a,) = setup_replay(tmp_path)
    path = a["store"].path_for("r-1")
    original = read_record(path)
    # SQLite is the authority; the final snapshot is only a recoverable projection.
    original["state"] = "graded"
    write_record(path, original)
    record = replay.import_round("source", "r-1")
    assert len(record["payload"]["episodes"]) == 10


@pytest.mark.parametrize(
    "bad", [b'{"schema":"spark-data-policy-v1","schema":"other"}', b'{"value":NaN}', b'{"value":1e999}', b"[]", b"{}{}"]
)
def test_strict_authority_json(tmp_path, bad):
    path = tmp_path / "record.json"
    path.write_bytes(bad)
    with pytest.raises(StageError):
        read_record(path)


def test_training_rows_bind_final_captured_bytes(tmp_path, monkeypatch):
    from admin.training import _rows

    path = tmp_path / "sft.jsonl"
    path.write_text('{"task_id":"gen-safe","messages":[]}\n')
    digest = file_digest(path)
    actual = Path.read_bytes

    def changed(file):
        raw = actual(file)
        return raw.replace(b"gen-safe", b"gen-poisoned") if file == path else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(StageError, match="final training-row capture"):
        _rows(path, expected_digest=digest)

"""Offline operator readiness must retain real prerequisites after fixture success."""

import json

import pytest
from test_admin_training import Tokenizer, workspace

from admin.artifacts import checkpoint_files, file_digest, write_record
from admin.cli import main
from admin.pipeline import Workspace, build_corpus
from admin.training import doctor, prepare_training


def test_empty_root_remains_uncreated_and_lists_both_exact_profiles(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HERMESBENCH_WITHHELD_SALT", raising=False)
    root = tmp_path / "absent"
    assert main(["doctor", "--software-only", "--root", str(root), "--profile", "rtx5090-poc"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] and report["scope"] == "software"
    assert not report["training_readiness_checked"]
    assert report["profiles"]["rtx5090-poc"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert report["profiles"]["bf16"]["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert report["production_blockers"]
    for name, check in report["prerequisites"].items():
        assert check["ready"] is (name == "software")
    assert main(["doctor", "--root", str(root)]) == 1
    assert main(["status", "--root", str(root)]) == 0
    assert "sn74" in capsys.readouterr().out
    assert not root.exists()


def prepared_fixture(root):
    ws = workspace(root)
    ws.record("corpus", build_corpus(ws))
    prepared = prepare_training(ws, tokenizer=Tokenizer(), profile="rtx5090-poc")
    merged = ws.models / "sft/adapter/merged"
    merged.mkdir(parents=True)
    write_record(merged / "config.json", {"model_type": "cpu_fixture", "not_a_model": True})
    (merged / "model.safetensors").write_bytes(b"CPU_FIXTURE_NOT_MODEL_WEIGHTS")
    write_record(
        ws.models / "sft/merged.json",
        {
            "merged": str(merged),
            "files": checkpoint_files(merged),
            "recipe_sha256": file_digest(ws.models / "sft/train.yaml"),
            "origin": ws.identity,
            "corpus_authority": prepared["corpus_authority"],
            **{key: prepared[key] for key in ("profile", "base_model", "revision")},
        },
    )
    return ws


def test_populated_fixture_verifies_artifacts_without_clearing_real_readiness(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HERMESBENCH_WITHHELD_SALT", raising=False)
    ws = prepared_fixture(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["doctor", "--root", str(ws.root), "--profile", "rtx5090-poc"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "fixture"
    checks = report["prerequisites"]
    for name in ("corpus", "prepared_training", "real_training"):
        assert checks[name]["artifact_valid"], checks[name]["summary"]
        assert checks[name]["ready"] is False
    for name in ("private_checks", "trusted_serving", "hardware", "attestation", "sn74"):
        assert checks[name]["ready"] is False
    assert not report["model_training_executed"]
    assert not report["external_requests_performed"]
    assert main(["status", "--root", str(ws.root), "--profile", "rtx5090-poc"]) == 0
    output = capsys.readouterr().out
    assert "Mode: fixture" in output
    assert "CPU fixture weights are not a trained model" in output
    assert "subnet-controlled" in output
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("artifact", ["corpus/sft.jsonl", "models/sft/adapter/merged/model.safetensors"])
def test_changed_artifact_is_not_ready(tmp_path, artifact):
    ws = prepared_fixture(tmp_path)
    (ws.root / artifact).write_bytes(b"changed")
    report = doctor(ws, profile="rtx5090-poc")
    check = "corpus" if artifact.startswith("corpus") else "real_training"
    assert report["prerequisites"][check]["artifact_valid"] is False
    assert "refused" in report["prerequisites"][check]["summary"]


def test_wrong_selected_profile_refuses_existing_preparation(tmp_path):
    ws = prepared_fixture(tmp_path)
    report = doctor(ws, profile="bf16")
    assert not report["prerequisites"]["prepared_training"]["artifact_valid"]
    assert "different selected model profile" in report["prerequisites"]["prepared_training"]["summary"]


def test_caller_authored_corpus_does_not_create_an_authority(tmp_path):
    root = tmp_path / "workspace"
    fake_authority = tmp_path / "fake-authority"
    ws = Workspace(root)
    ws.record("corpus", {"authority": {"root": str(fake_authority)}})
    report = doctor(ws)
    assert not report["prerequisites"]["corpus"]["artifact_valid"]
    assert not (root / ".identity").exists()
    assert not fake_authority.exists()


def test_invalid_workspace_identity_is_a_clean_cli_refusal(tmp_path, capsys):
    write_record(tmp_path / ".identity", {"mode": [], "namespace": "fixture", "issuer": "untrusted"})
    assert main(["doctor", "--root", str(tmp_path)]) == 2
    assert "invalid workspace identity" in capsys.readouterr().err


def test_fixture_tokenizer_cannot_be_ignored_on_another_command():
    with pytest.raises(SystemExit) as exc:
        main(["train", "--fixture-tokenizer"])
    assert exc.value.code == 2


def test_root_help_lists_integrated_groups_and_no_command_is_implicit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in ("cycle", "replay", "curriculum", "parents", "candidates", "cotraining", "release"):
        assert command in output
    # Whitespace-collapsed, because argparse re-wraps the epilog to the terminal width and
    # `--mode fixture` lands across a line break at 80 columns. Asserting the raw substring
    # made this test pass or fail on COLUMNS rather than on what the help actually says.
    assert "--mode fixture" in " ".join(output.split())
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []

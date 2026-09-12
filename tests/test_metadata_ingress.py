"""CPU metadata ingress controls; transport fixtures never confer proof or payout authority."""

import json
from types import SimpleNamespace

import pytest

from eval import registry_gate, rollout_track, rollout_track_cli, training_track_gate
from hermes.evidence_json import evidence_object, evidence_value
from validator.pr_admission import AdmissionError, GitHubSource
from validator.settlement import GitHubActions

BAD = [
    '{"miner":"other","miner":"alice"}',
    '{"miner":"alice","mi\\u006eer":"alice"}',
    '{"claims":{"passed":false,"passed":true}}',
    '{"rows":NaN,"rows":10}',
    '{"rows":Infinity,"rows":10}',
    '{"rows":-Infinity,"rows":10}',
    '{"rows":1e999,"rows":10}',
    '{"rows":01,"rows":10}',
    '{"miner":',
    "[]",
    "null",
    "true",
]
LITERAL = 'NaN Infinity {"passed":false,"passed":true}\u2028ordinary string'


def no_effect(*args, **kwargs):
    pytest.fail("malformed metadata reached a proof download, verification or external effect")


@pytest.mark.parametrize("raw", BAD)
@pytest.mark.parametrize("where", ["head", "base"])
def test_data_registry_cli_refuses_original_bytes_before_proof(tmp_path, monkeypatch, raw, where):
    valid = json.dumps(
        {
            "miner": "alice",
            "hf_url": "https://huggingface.co/datasets/org/data",
            "trajectories_sha256": "a" * 64,
            "rows_total": 25,
            "dataset_version": 1,
            "gpu_architecture": "hopper",
        }
    )
    base, head = ("", raw) if where == "head" else (raw + "\n", raw + "\n" + valid)
    monkeypatch.setattr(registry_gate, "_git_show", lambda ref, path: base if ref == "base" else head)
    monkeypatch.setattr(registry_gate, "verify_dataset_submission", no_effect)
    monkeypatch.setattr(registry_gate, "compute_rows_selected_for_entry", no_effect)
    out = tmp_path / "report.json"
    assert (
        registry_gate.main(
            [
                "--base-ref",
                "base",
                "--head-ref",
                "head",
                "--out",
                str(out),
                "--sparkproof-root",
                str(tmp_path),
                "--skip-mining-publish",
            ]
        )
        == 1
    )
    report = json.loads(out.read_text())
    assert not report["verified"] and not report["merge_eligible"] and not report["reward_eligible"]
    assert report["label"] == "dataset:REJECT" and not report["submissions"]
    path = tmp_path / "registry.jsonl"
    path.write_text(raw)
    with pytest.raises(ValueError):
        registry_gate._load_registry_lines(path)
    assert path.read_text() == raw


@pytest.mark.parametrize("raw", BAD)
@pytest.mark.parametrize("where", ["head", "base", "round"])
def test_rollout_cli_rejects_ambiguous_registry_and_round_before_fetch(tmp_path, monkeypatch, raw, where):
    valid = json.dumps({"round_id": "r1", "miner_id": "alice"})
    base, head = ("", raw) if where == "head" else (raw + "\n", raw + "\n" + valid)
    if where == "round":
        base, head = "", valid

    def read(ref, path):
        if path.endswith("rollouts.jsonl"):
            return base if ref == "base" else head
        return raw

    monkeypatch.setattr(rollout_track_cli, "git_show", read)
    monkeypatch.setattr(rollout_track_cli, "fetch_exports", no_effect)
    paths = tmp_path / "paths"
    paths.write_text("datasets/rollouts.jsonl\n")
    out = tmp_path / "report.json"
    assert (
        rollout_track_cli.main(
            [
                "--base-ref",
                "base",
                "--head-ref",
                "head",
                "--changed-paths-file",
                str(paths),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert json.loads(out.read_text())["verdict"] == rollout_track.REJECT
    if where == "base":
        assert rollout_track.check_novelty({"round_id": "r1", "miner_id": "alice"}, base)


@pytest.mark.parametrize("raw", BAD + ["", " ", None])
def test_training_attestation_ingress_rejects_before_bundle_download(tmp_path, monkeypatch, raw):
    monkeypatch.setattr(training_track_gate, "_git_show", lambda ref, path: raw)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_effect)
    issues, label = training_track_gate.verify_remote_proof_bundle_scores(
        "org/proof",
        head_ref="head",
        changed_paths=["runs/r1/attestation.json"],
    )
    # Ingress failure returns issues and no computed tier. The training gate
    # includes those issues in training:REJECT.
    assert label is None and any("invalid JSON" in issue for issue in issues)


@pytest.mark.parametrize("raw", BAD)
def test_rollout_attestation_ingress_refuses_without_authority(tmp_path, raw):
    (tmp_path / "attestation.json").write_text(raw)
    assert rollout_track_cli.load_attestation(tmp_path) is None


def test_valid_sibling_objects_preserve_literal_payload_and_pin(tmp_path, monkeypatch):
    raw = json.dumps({"round_id": "old", "miner_id": "alice", "notes": LITERAL}, ensure_ascii=False)
    new = json.dumps({"round_id": "next", "miner_id": "bob", "notes": LITERAL}, ensure_ascii=False)
    for parse in (registry_gate.parse_added_registry_lines, rollout_track.added_lines):
        assert parse(raw + "\n", raw + "\n" + new) == [json.loads(new)]
        assert parse(raw + "\n", raw + "\n" + raw) == [json.loads(raw)]
    monkeypatch.setattr(rollout_track_cli, "git_show", lambda ref, path: raw)
    assert rollout_track_cli.load_round("old", "base")["notes"] == LITERAL
    path = tmp_path / "attestation.json"
    path.write_text(raw)
    assert rollout_track_cli.load_attestation(tmp_path)["notes"] == LITERAL
    path = tmp_path / "registry.jsonl"
    path.write_text(raw + "\n")
    assert registry_gate._load_registry_lines(path)[0]["notes"] == LITERAL
    monkeypatch.setattr(training_track_gate, "_git_show", lambda ref, path: raw)
    downloaded = []

    def snapshot(**kwargs):
        downloaded.append(kwargs)
        return tmp_path

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    # No manifest means no verification authority; this tests only ingress + pinned transport.
    report, attestation, error, path = training_track_gate._download_and_verify_bundle(
        "org/proof",
        head_ref="head",
        changed_paths=["runs/r1/attestation.json"],
        revision="a" * 40,
    )
    assert report is None and error is None and attestation is None
    assert downloaded[0]["revision"] == "a" * 40 and path == tmp_path


@pytest.mark.parametrize("check", [registry_gate.validate_append_only_registry, rollout_track.check_append_only])
def test_prior_registry_bytes_cannot_be_reformatted(check):
    base = '{ "miner_id": "alice" }\n'
    assert check(base, '{"miner_id":"alice"}\n{}\n')
    assert check(base, base + "{}\n") == []
    assert check("{}", "{} {}")


@pytest.mark.parametrize("raw", ["{}", '{"miner_id":[],"round_id":"r"}'])
def test_historical_rollout_identity_refuses(raw):
    assert rollout_track.check_novelty({"round_id": "new", "miner_id": "bob"}, raw)


@pytest.mark.parametrize("where", ["user", "pr", "labels", "reviews"])
@pytest.mark.parametrize("assertion", ['"x":0,"x":0', '"x":NaN,"x":0', '"x":0,"\\u0078":1'])
def test_action_adapter_refuses_ambiguous_provider_bytes_without_mutation(monkeypatch, where, assertion):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    action = {
        "repository": "org/repo",
        "pr": 7,
        "author": "alice",
        "head_sha": "a" * 40,
        "kind": "review" if where == "reviews" else "label_remove",
        "key": "fixture-key",
        "label": "crown",
        "body": LITERAL,
    }
    mutations = []

    def transport(argv, **kwargs):
        if argv[5] != "GET":
            mutations.append(argv)
            pytest.fail("refused response reached mutation")
        endpoint = argv[6]
        if endpoint == "user":
            which, value = "user", {"login": "bot"}
        elif "labels?" in endpoint:
            which, value = "labels", [{"name": "crown"}]
        elif "reviews?" in endpoint:
            which, value = "reviews", [{"body": "old", "user": {"login": "bot"}}]
        else:
            which, value = (
                "pr",
                {
                    "number": 7,
                    "base": {"repo": {"full_name": "org/repo"}},
                    "head": {"sha": action["head_sha"]},
                    "user": {"login": "alice"},
                    "state": "open",
                    "merged": False,
                    "draft": False,
                },
            )
        raw = json.dumps(value)
        if which == where:
            raw = raw.replace("{", '{"extra":[{' + assertion + "}],", 1)
        return SimpleNamespace(returncode=0, stdout=raw, stderr="")

    with pytest.raises(AdmissionError, match="malformed JSON"):
        adapter = GitHubActions("org/repo", transport=transport)
        adapter.reconcile(action)
    assert not mutations


@pytest.mark.parametrize("kind", ["label_remove", "review"])
def test_action_adapter_keeps_valid_paginated_historical_reconciliation(monkeypatch, kind):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    action = {
        "repository": "org/repo",
        "pr": 7,
        "author": "alice",
        "head_sha": "a" * 40,
        "kind": kind,
        "key": "fixture-key",
        "label": "crown",
        "body": LITERAL,
    }
    calls = []

    def transport(argv, **kwargs):
        assert argv[5] == "GET" and kwargs["env"]["GH_TOKEN"] == "fixture-credential"
        endpoint = argv[6]
        calls.append(endpoint)
        if endpoint == "user":
            value = {"login": "bot"}
        elif "labels?" in endpoint:
            value = [{"name": f"unrelated-{i}"} for i in range(100)] if endpoint.endswith("page=1") else []
        elif "reviews?" in endpoint:
            review = {
                "id": 101,
                "body": action["body"] + "\n\n" + GitHubActions.marker(action),
                "user": {"login": "bot"},
                "commit_id": action["head_sha"],
                "state": "COMMENTED",
            }
            value = (
                [{**review, "body": "old", "id": i} for i in range(100)] if endpoint.endswith("page=1") else [review]
            )
        else:
            value = {
                "number": 7,
                "base": {"repo": {"full_name": "org/repo"}},
                "head": {"sha": action["head_sha"]},
                "user": {"login": "alice"},
                "state": "closed" if kind == "label_remove" else "open",
                "merged": kind == "label_remove",
                "draft": False,
            }
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    adapter = GitHubActions("org/repo", transport=transport)
    receipt = adapter.reconcile(action)
    assert receipt["key"] == action["key"]
    assert receipt["evidence"] == (
        {"label": "crown", "present": False} if kind == "label_remove" else {"review_id": 101}
    )
    assert any(endpoint.endswith("page=2") for endpoint in calls)
    assert adapter.reconcile(action) == receipt


def test_general_decoder_keeps_arrays_and_does_not_reparse_strings(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    raw = json.dumps([{"body": LITERAL}, None, 3.5])
    source = GitHubSource("org/repo", transport=lambda *a, **k: SimpleNamespace(returncode=0, stdout=raw))
    assert source.get("fixture") == evidence_value(raw) == [{"body": LITERAL}, None, 3.5]
    with pytest.raises(ValueError, match="must be a JSON object"):
        evidence_object(raw)


@pytest.mark.parametrize("raw", ["\u2028", "\v", '{"x":0}\u2028{"x":1}'])
def test_non_json_record_separators_are_not_silently_ignored(raw):
    for parse in (registry_gate.parse_added_registry_lines, rollout_track.added_lines):
        with pytest.raises(ValueError):
            parse(raw, raw + '\n{"miner_id":"alice"}')


def test_training_cli_refuses_malformed_attestation_with_no_tier_or_snapshot(tmp_path, monkeypatch):
    from eval.canonical_dataset import canonical_hf_url, canonical_sft_sha256

    body = tmp_path / "body"
    body.write_text(
        f"- [x] **Training/evaluation improvement**\nCanonical dataset URL: {canonical_hf_url()}\n"
        f"Pinned sft_sha256: `{canonical_sft_sha256()}`\nProof-bundle URL: https://huggingface.co/org/proof\n"
    )
    paths = tmp_path / "paths"
    paths.write_text("runs/r1/attestation.json\n")
    # Controlled local manifest transport; the real canonical precheck still runs.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_url": canonical_hf_url()}))
    mix = tmp_path / "mix_manifest.json"
    mix.write_text(json.dumps({"sft_sha256": canonical_sft_sha256()}))
    scores = tmp_path / "eval_scores.json"
    scores.write_text("{}")

    def download(**kwargs):
        return tmp_path / kwargs["filename"]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_effect)
    # Use the unchanged canonical record as this controlled base-ref response;
    # the test also runs from an installed wheel outside any Git checkout.
    from pathlib import Path

    canonical = Path("datasets/canonical.json").read_text()
    monkeypatch.setattr(
        training_track_gate,
        "_git_show",
        lambda ref, path: '{"passed":false,"passed":true}' if path.endswith("attestation.json") else canonical,
    )
    output = tmp_path / "gate.json"
    assert (
        training_track_gate.main(
            [
                "--head-ref",
                "fixture-head",
                "--pr-body-file",
                str(body),
                "--changed-paths-file",
                str(paths),
                "--out",
                str(output),
                "--skip-hf-pin-check",
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["label"] == "training:REJECT" and report["eval_label"] is None
    assert not report["verified"] and not report["merge_eligible"]
    assert any("attestation.json: invalid JSON" in issue for issue in report["issues"])

"""Tests for eval.canonical_dataset and training_track_gate."""

from pathlib import Path

import yaml

from eval.canonical_dataset import (
    CANONICAL_PREFERENCE_DATASET_PATH,
    CANONICAL_TRAINING_DATASET_PATH,
    assert_recipe_uses_canonical_dataset,
    canonical_dataset_path_for_track,
    load_canonical,
    recipe_training_track,
)
from eval.training_track_gate import (
    gate_training_pr,
    is_training_track_pr,
    should_enforce_training_gate,
    validate_changed_paths,
    validate_pr_body_canonical_pin,
    validate_pr_body_proof_bundle,
    validate_recipe_paths_in_ref,
)

_VALID_PROOF_BUNDLE_URL = "https://huggingface.co/gittensor-model-hub/sparkdistill-2026-07-11-qwen3.5-4b-mining-001"


def _training_pr_body(*, proof_bundle_url: str | None = _VALID_PROOF_BUNDLE_URL) -> str:
    pin = load_canonical()
    proof_line = proof_bundle_url or "pending"
    return (
        "- [x] **Training/evaluation improvement**\n"
        f"- Canonical dataset URL: {pin['hf_url']}\n"
        f"- Pinned sft_sha256: `{pin['mix_manifest']['sft_sha256']}`\n"
        f"- Proof-bundle URL: {proof_line}\n"
    )


def test_load_canonical_pin():
    pin = load_canonical(Path("datasets/canonical.json"))
    assert pin["repo_id"] == "gittensor-model-hub/sparkproof-mining"
    assert pin["mix_manifest"]["sft_sha256"]


def test_recipe_rejects_non_canonical_paths():
    recipe = {
        "datasets": [{"path": "data/processed/triton_sft.jsonl"}],
    }
    issues = assert_recipe_uses_canonical_dataset(recipe)
    assert any(CANONICAL_TRAINING_DATASET_PATH in issue for issue in issues)


def test_recipe_training_track_detection():
    assert recipe_training_track({}) == "sft"
    assert recipe_training_track({"rl": "dpo"}) == "dpo"
    assert recipe_training_track({"rl": "DPO"}) == "dpo"
    # Unknown rl values are not the DPO track (fail safe to SFT gating).
    assert recipe_training_track({"rl": "ppo"}) == "sft"
    assert canonical_dataset_path_for_track("dpo") == CANONICAL_PREFERENCE_DATASET_PATH
    assert canonical_dataset_path_for_track("sft") == CANONICAL_TRAINING_DATASET_PATH


def test_sft_recipe_gating_unchanged_by_dpo_support():
    # An SFT recipe (no rl key) still requires the SFT path — verbatim prior behavior.
    assert assert_recipe_uses_canonical_dataset({"datasets": [{"path": CANONICAL_TRAINING_DATASET_PATH}]}) == []
    issues = assert_recipe_uses_canonical_dataset({"datasets": [{"path": CANONICAL_PREFERENCE_DATASET_PATH}]})
    assert any(CANONICAL_TRAINING_DATASET_PATH in issue for issue in issues)


def test_dpo_recipe_requires_preference_dataset():
    # A DPO recipe pointed at the preference path is accepted...
    assert (
        assert_recipe_uses_canonical_dataset({"rl": "dpo", "datasets": [{"path": CANONICAL_PREFERENCE_DATASET_PATH}]})
        == []
    )
    # ...but a DPO recipe pointed at the SFT mix is rejected (tracks never cross).
    issues = assert_recipe_uses_canonical_dataset(
        {"rl": "dpo", "datasets": [{"path": CANONICAL_TRAINING_DATASET_PATH}]}
    )
    assert any(CANONICAL_PREFERENCE_DATASET_PATH in issue for issue in issues)


def test_training_track_checkbox():
    assert is_training_track_pr("- [x] **Training/evaluation improvement**")
    assert is_training_track_pr("- [x] Training/evaluation improvement")
    assert not is_training_track_pr("- [x] **Dataset track submission**")
    assert not is_training_track_pr("- [x] Dataset track submission")


def test_forbidden_training_paths():
    issues = validate_changed_paths(["eval/gen_triton_kernels.py"])
    assert any("forbidden pattern" in issue for issue in issues)
    issues = validate_changed_paths(["scripts/prepare_triton_kernels.sh"])
    assert issues
    assert validate_changed_paths(["datasets/canonical.json"]) == []


def test_validate_pr_body_requires_canonical_citation():
    pin = load_canonical()
    body = (
        f"Dataset URL: {pin['hf_url']}\n"
        f"sha `{pin['mix_manifest']['sft_sha256']}`\n"
        f"Proof-bundle URL: {_VALID_PROOF_BUNDLE_URL}\n"
    )
    assert validate_pr_body_canonical_pin(body) == []


def test_validate_pr_body_rejects_missing_proof_bundle():
    pin = load_canonical()
    body = (
        f"Dataset URL: {pin['hf_url']}\n"
        f"sha `{pin['mix_manifest']['sft_sha256']}`\n"
        "Proof-bundle URL: pending after local train + eval\n"
    )
    issues = validate_pr_body_proof_bundle(body)
    assert issues
    assert any("pending" in issue.lower() or "published" in issue.lower() for issue in issues)


def test_gate_training_pr_rejects_missing_proof_bundle(tmp_path: Path):
    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/qwen3.5-4b-phase1/sft-mining.yaml"],
        pr_body=_training_pr_body(proof_bundle_url="pending after local train + eval"),
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["label"] == "training:REJECT"
    assert any("Proof-bundle" in issue for issue in report["issues"])


def test_gate_training_pr_rejects_local_generator(tmp_path: Path):
    recipe = tmp_path / "recipes/qwen3.5-4b-phase1/sft-triton.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/triton_sft.jsonl"}],
            }
        ),
        encoding="utf-8",
    )
    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["eval/gen_triton_kernels.py", recipe.as_posix()],
        pr_body=_training_pr_body(),
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["label"] == "training:REJECT"
    assert not report["verified"]
    assert report["issues"]


def test_validate_recipe_paths_in_worktree(tmp_path: Path, monkeypatch):
    recipe = tmp_path / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump({"datasets": [{"path": CANONICAL_TRAINING_DATASET_PATH}]}),
        encoding="utf-8",
    )

    def _fake_show(ref, path):
        if path.endswith("recipes/demo/sft.yaml"):
            return recipe.read_text(encoding="utf-8")
        return None

    monkeypatch.setattr("eval.training_track_gate._git_show", _fake_show)
    assert validate_recipe_paths_in_ref("HEAD", ["recipes/demo/sft.yaml"]) == []


def test_pr_training_track_detects_dpo(monkeypatch):
    import eval.training_track_gate as gate
    from eval.training_track_gate import pr_training_track

    monkeypatch.setattr(gate, "_git_show", lambda ref, path: "rl: dpo\ndatasets:\n  - path: x\n")
    assert pr_training_track("HEAD", ["recipes/demo/dpo.yaml"]) == "dpo"
    monkeypatch.setattr(gate, "_git_show", lambda ref, path: "datasets:\n  - path: x\n")
    assert pr_training_track("HEAD", ["recipes/demo/sft.yaml"]) == "sft"
    assert pr_training_track("HEAD", []) == "sft"


def test_validate_pr_body_canonical_pin_dpo_track(monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "canonical_pref_hf_url", lambda: "https://huggingface.co/datasets/org/dpo")
    monkeypatch.setattr(gate, "canonical_pref_sha256", lambda: "a" * 64)
    good = f"Preference dataset: https://huggingface.co/datasets/org/dpo\npref `{'a' * 64}`\n"
    assert gate.validate_pr_body_canonical_pin(good, track="dpo") == []
    # Wrong (valid-hex) sha is rejected.
    bad = f"https://huggingface.co/datasets/org/dpo\n`{'b' * 64}`\n"
    assert gate.validate_pr_body_canonical_pin(bad, track="dpo")


def test_validate_pr_body_canonical_pin_dpo_fails_closed_without_pin(monkeypatch):
    import eval.training_track_gate as gate

    def _raise():
        raise ValueError("no preference pin")

    monkeypatch.setattr(gate, "canonical_pref_hf_url", _raise)
    issues = gate.validate_pr_body_canonical_pin("some body", track="dpo")
    assert any("DPO track is unavailable" in issue for issue in issues)


def test_should_enforce_gate_ignores_docs_under_recipes():
    # A README (or any non-YAML file) under recipes/ is documentation, not a training
    # submission — regression guard for PR #284 being auto-closed by --close-on-reject.
    assert should_enforce_training_gate(None, ["recipes/qwen3.5-4b-phase1/README.md"]) is False
    assert (
        should_enforce_training_gate(
            None,
            ["recipes/qwen3.5-4b-phase1/README.md", "eval/train_prep.py", "scripts/install_train.sh"],
        )
        is False
    )


def test_should_enforce_gate_on_recipe_yaml_and_generators():
    # Actual recipe files and local generators still trigger the gate.
    assert should_enforce_training_gate(None, ["recipes/qwen3.5-4b-phase1/sft.yaml"]) is True
    assert should_enforce_training_gate(None, ["recipes/demo/dpo.yml"]) is True
    assert should_enforce_training_gate(None, ["eval/gen_triton_kernels.py"]) is True
    assert should_enforce_training_gate(None, ["scripts/prepare_triton_kernels.sh"]) is True


def test_should_enforce_gate_respects_body_checkboxes():
    # An explicit training checkbox forces enforcement regardless of paths...
    assert should_enforce_training_gate("- [x] **Training/evaluation improvement**", ["docs/x.md"]) is True
    # ...and a dataset-track checkbox opts out even when a recipe yaml changed.
    assert should_enforce_training_gate("- [x] **Dataset track submission**", ["recipes/x/sft.yaml"]) is False


def test_gate_skips_docs_only_recipes_change():
    # End-to-end: a docs-only PR touching recipes/ is training:skipped, never REJECT.
    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/qwen3.5-4b-phase1/README.md"],
        pr_body=None,
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["label"] == "training:skipped"
    assert report["verified"] is True

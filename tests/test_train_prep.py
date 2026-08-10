"""Tests for eval.train_prep."""

import json
from pathlib import Path

import pytest
import yaml

from eval.train_prep import (
    MIN_SAMPLE_PACKING_ROWS,
    _is_qwen3_5_recipe,
    count_jsonl_rows,
    prepare_train_recipe,
)


def _write_jsonl(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"p{i}"},
                            {"role": "assistant", "content": f"a{i}"},
                        ]
                    }
                )
                + "\n"
            )


def test_count_jsonl_rows(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, 3)
    assert count_jsonl_rows(path) == 3


def test_prepare_train_recipe_accepts_dpo_preference_dataset(tmp_path: Path, monkeypatch):
    # A `rl: dpo` recipe pointing at the canonical preference dataset is accepted and
    # resolved the same way as SFT — no method-specific handling in train_prep.
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_dpo.jsonl"
    data.parent.mkdir(parents=True, exist_ok=True)
    with data.open("w", encoding="utf-8") as handle:
        for i in range(3):
            handle.write(json.dumps({"prompt": f"p{i}", "chosen": "good", "rejected": "bad"}) + "\n")
    recipe = root / "recipes/demo/dpo.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "rl": "dpo",
                "datasets": [{"path": "data/processed/sparkproof-mining_dpo.jsonl", "type": "chat_template.argilla"}],
                "dataset_prepared_path": "data/prepared/demo-dpo",
                "output_dir": "outputs/demo-dpo",
                "attn_implementation": "flash_attention_2",
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))
    assert prepared["datasets"][0]["path"] == str(data.resolve())
    assert prepared["rl"] == "dpo"
    assert result["row_count"] == 3


def test_prepare_train_recipe_resolves_paths_and_disables_packing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS - 1)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "dataset_prepared_path": "data/prepared/demo",
                "output_dir": "outputs/demo",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "attn_implementation": "flash_attention_2",
                "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert prepared["datasets"][0]["path"] == str(data.resolve())
    assert prepared["dataset_prepared_path"] == str((root / "data/prepared/demo").resolve())
    assert prepared["output_dir"] == str((root / "outputs/demo").resolve())
    assert prepared["sample_packing"] is False
    assert prepared["pad_to_sequence_len"] is False
    assert prepared["attn_implementation"] == "sdpa"
    assert "plugins" not in prepared
    assert result["row_count"] == MIN_SAMPLE_PACKING_ROWS - 1
    assert any("sample_packing disabled" in note for note in result["notes"])


def test_multipack_guard_keeps_packing_above_threshold(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "sample_packing": True,
                "pad_to_sequence_len": True,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert result["row_count"] == MIN_SAMPLE_PACKING_ROWS
    assert prepared["sample_packing"] is True
    assert prepared["pad_to_sequence_len"] is True
    assert not any("sample_packing disabled" in note for note in result["notes"])


def test_multipack_guard_disables_when_total_below_threshold(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, 10)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "sample_packing": True,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert result["row_count"] == 10
    assert prepared["sample_packing"] is False


def test_prepare_train_recipe_rejects_multiple_datasets(tmp_path: Path):
    root = tmp_path / "distill"
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {"path": "data/processed/sparkproof-mining_sft.jsonl"},
                    {"path": "data/processed/other.jsonl"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical mining dataset"):
        prepare_train_recipe(recipe_path=recipe, distill_root=root)


def test_prepare_train_recipe_upgrades_to_flash_attention_3(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: True)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: True)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "attn_implementation": "flash_attention_2",
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert prepared["attn_implementation"] == "flash_attention_3"
    assert any("flash_attention_3" in note for note in result["notes"])


def test_is_qwen3_5_recipe_detection():
    assert _is_qwen3_5_recipe({"base_model": "Qwen/Qwen3.5-4B"})
    assert _is_qwen3_5_recipe({"chat_template": "qwen3_5"})
    assert _is_qwen3_5_recipe({"model_config_type": "qwen3_5"})
    assert not _is_qwen3_5_recipe({"base_model": "Qwen/Qwen2.5-7B", "chat_template": "chatml"})
    assert not _is_qwen3_5_recipe({})


def _qwen3_5_packing_recipe(root: Path) -> Path:
    """A Qwen3.5 recipe with enough rows that only the Qwen3.5 guard can disable packing."""
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS)  # above the small-mix threshold
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "base_model": "Qwen/Qwen3.5-4B",
                "chat_template": "qwen3_5",
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "sample_packing": True,
                "pad_to_sequence_len": True,
            }
        ),
        encoding="utf-8",
    )
    return recipe


def test_qwen3_5_recipe_disables_sample_packing(tmp_path: Path, monkeypatch):
    # Above the row threshold, packing is only disabled by the Qwen3.5 Axolotl-skew guard.
    monkeypatch.delenv("SPARKDISTILL_ALLOW_QWEN3_5_PACKING", raising=False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    monkeypatch.setattr("eval.train_prep._python_headers_available", lambda: True)
    root = tmp_path / "distill"
    recipe = _qwen3_5_packing_recipe(root)

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert result["row_count"] == MIN_SAMPLE_PACKING_ROWS  # not the small-mix path
    assert prepared["sample_packing"] is False
    assert any("Qwen3.5" in note and "block_type" in note for note in result["notes"])


def test_qwen3_5_packing_kept_with_override_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_ALLOW_QWEN3_5_PACKING", "1")
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    monkeypatch.setattr("eval.train_prep._python_headers_available", lambda: True)
    root = tmp_path / "distill"
    recipe = _qwen3_5_packing_recipe(root)

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert prepared["sample_packing"] is True
    assert any("kept for Qwen3.5" in note for note in result["notes"])


def test_qwen3_5_missing_python_headers_warns(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPARKDISTILL_ALLOW_QWEN3_5_PACKING", raising=False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    monkeypatch.setattr("eval.train_prep._python_headers_available", lambda: False)
    root = tmp_path / "distill"
    recipe = _qwen3_5_packing_recipe(root)

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    assert any("Python.h" in note and "python3-dev" in note for note in result["notes"])


def test_qwen3_5_present_python_headers_no_warn(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPARKDISTILL_ALLOW_QWEN3_5_PACKING", raising=False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    monkeypatch.setattr("eval.train_prep._python_headers_available", lambda: True)
    root = tmp_path / "distill"
    recipe = _qwen3_5_packing_recipe(root)

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    assert not any("Python.h" in note for note in result["notes"])


def test_prepare_train_recipe_strips_cce_for_qwen3_5(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: True)
    monkeypatch.setattr("eval.train_prep._python_headers_available", lambda: True)
    root = tmp_path / "distill"
    data = root / "data/processed/sparkproof-mining_sft.jsonl"
    _write_jsonl(data, 2)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "chat_template": "qwen3_5",
                "datasets": [{"path": "data/processed/sparkproof-mining_sft.jsonl"}],
                "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert "plugins" not in prepared
    assert any("unsupported for qwen3_5" in note for note in result["notes"])

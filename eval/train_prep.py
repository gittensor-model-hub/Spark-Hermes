"""Prepare Axolotl recipe YAML for reliable local training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from eval.canonical_dataset import assert_recipe_uses_canonical_dataset

# Axolotl multipack sampler fails on very small mixes (observed at 17 rows).
MIN_SAMPLE_PACKING_ROWS = 32

_PATH_KEYS = ("path", "dataset_prepared_path", "output_dir")

# Qwen3.5 recipe detection (base_model / chat_template / config type all encode it).
_QWEN3_5_MARKERS = ("qwen3.5", "qwen3_5")
# Axolotl 0.18's Qwen3.5 sample-packing monkeypatch reads ``self.layer_type``, but
# current transformers ``Qwen3_5DecoderLayer`` stores ``self.block_type`` and never
# sets ``layer_type`` — so packing crashes with AttributeError on a fresh install
# (issue #283). Disable packing for Qwen3.5 until the upstream fix lands; this env
# var re-enables it for a patched/compatible Axolotl.
_ALLOW_QWEN3_5_PACKING_ENV = "SPARKDISTILL_ALLOW_QWEN3_5_PACKING"


def _is_qwen3_5_recipe(cfg: dict[str, Any]) -> bool:
    for key in ("base_model", "model_type", "model_config_type", "chat_template", "tokenizer_type"):
        value = str(cfg.get(key) or "").lower()
        if any(marker in value for marker in _QWEN3_5_MARKERS):
            return True
    return False


def _qwen3_5_packing_allowed() -> bool:
    return os.environ.get(_ALLOW_QWEN3_5_PACKING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _python_headers_available() -> bool:
    """Whether the active interpreter's CPython dev headers (Python.h) are present.

    Qwen3.5's hybrid linear-attention path binds flash-linear-attention, whose Triton
    CUDA extensions compile against Python.h. Without the headers (no python3-dev) the
    build fails and FLA rolls back to a CPU path that crashes mid-train (issue #283).
    """
    import sysconfig

    include = sysconfig.get_paths().get("include")
    return bool(include) and Path(include, "Python.h").exists()


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _has_flash_attn_3() -> bool:
    try:
        import torch
        from transformers.utils import is_flash_attn_3_available

        # Official FA3 wheels currently contain Hopper kernels; they import on
        # Blackwell but fail at launch with "no kernel image".
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            return False
        return bool(is_flash_attn_3_available())
    except ImportError:
        return False


def _has_cut_cross_entropy() -> bool:
    try:
        import cut_cross_entropy  # noqa: F401
        from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin

        CutCrossEntropyPlugin()._check_requirements()
        return True
    except Exception:
        return False


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def prepare_train_recipe(
    *,
    recipe_path: Path,
    distill_root: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Return a training-safe recipe with absolute paths and runtime fallbacks."""
    root = distill_root.resolve()
    cfg: dict[str, Any] = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{recipe_path} must contain a YAML mapping")

    canonical_issues = assert_recipe_uses_canonical_dataset(cfg)
    if canonical_issues:
        raise ValueError(
            "training recipes must use the pinned canonical mining dataset only: " + "; ".join(canonical_issues)
        )

    notes: list[str] = []
    row_count: int | None = None

    for key in _PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            cfg[key] = _resolve_path(value, root)

    datasets = cfg.get("datasets")
    if isinstance(datasets, list):
        total_rows = 0
        counted_sources = 0
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            data_path = entry.get("path")
            if isinstance(data_path, str) and data_path.strip():
                entry["path"] = _resolve_path(data_path, root)
            data_path = entry.get("path")
            if isinstance(data_path, str) and data_path.endswith(".jsonl"):
                rows = count_jsonl_rows(Path(data_path))
                total_rows += rows
                counted_sources += 1
                notes.append(f"dataset rows: {rows}")
        if counted_sources:
            # Axolotl concatenates all datasets, so the multipack guard below must
            # look at the combined row count, not just the last shard's.
            row_count = total_rows
            if counted_sources > 1:
                notes.append(f"dataset rows (total): {total_rows}")

    if row_count is not None and row_count < MIN_SAMPLE_PACKING_ROWS:
        if cfg.get("sample_packing"):
            cfg["sample_packing"] = False
            notes.append(f"sample_packing disabled (<{MIN_SAMPLE_PACKING_ROWS} rows)")
        if cfg.get("pad_to_sequence_len"):
            cfg["pad_to_sequence_len"] = False
            notes.append("pad_to_sequence_len disabled for small dataset")

    # Axolotl's Qwen3.5 packing monkeypatch crashes against current transformers
    # (self.layer_type vs self.block_type); disable packing until upstream fixes it.
    if cfg.get("sample_packing") and _is_qwen3_5_recipe(cfg):
        if _qwen3_5_packing_allowed():
            notes.append(f"sample_packing kept for Qwen3.5 ({_ALLOW_QWEN3_5_PACKING_ENV} set)")
        else:
            cfg["sample_packing"] = False
            notes.append(
                "sample_packing disabled for Qwen3.5: Axolotl's packing monkeypatch reads "
                "self.layer_type but current transformers uses self.block_type (issue #283); "
                f"set {_ALLOW_QWEN3_5_PACKING_ENV}=1 to override"
            )

    # Qwen3.5 binds flash-linear-attention; its Triton kernels need Python.h to build.
    if _is_qwen3_5_recipe(cfg) and not _python_headers_available():
        notes.append(
            "WARNING: Python.h (python3-dev) not found — Qwen3.5 binds flash-linear-attention, "
            "whose Triton CUDA extensions need CPython headers to compile; without them FLA rolls "
            "back to CPU and crashes mid-train (issue #283). Install python3-dev before training."
        )

    attn = cfg.get("attn_implementation")
    if attn == "flash_attention_2" and _has_flash_attn_3():
        cfg["attn_implementation"] = "flash_attention_3"
        notes.append("attn_implementation: flash_attention_3 (flash_attn_3 wheel)")
    elif attn == "flash_attention_2" and not _has_flash_attn():
        cfg["attn_implementation"] = "sdpa"
        notes.append("attn_implementation: sdpa (flash_attn not installed)")

    plugins = cfg.get("plugins")
    if isinstance(plugins, list) and plugins:
        cce_reason: str | None = None
        if not _has_cut_cross_entropy():
            cce_reason = "not installed"
        elif cfg.get("chat_template") == "qwen3_5":
            # CCE's qwen3_5 patch currently fails against transformers' remote-code
            # module layout (FileNotFoundError on modeling_qwen3_5).
            cce_reason = "unsupported for qwen3_5 chat_template"
        if cce_reason:
            cfg.pop("plugins", None)
            notes.append(f"removed CutCrossEntropyPlugin ({cce_reason})")

    destination = out_path or (root / "data" / "prepared" / f"{recipe_path.stem}.prepared.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    return {
        "source_recipe": str(recipe_path.resolve()),
        "prepared_recipe": str(destination.resolve()),
        "row_count": row_count,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True, help="source Axolotl yaml")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="SparkDistill repo root for resolving relative paths",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write prepared yaml here (default: data/prepared/<recipe-stem>.prepared.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        result = prepare_train_recipe(
            recipe_path=args.recipe,
            distill_root=args.root,
            out_path=args.out,
        )
    except Exception as exc:
        print(f"train prep failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

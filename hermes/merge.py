"""Folding a LoRA adapter back into the base, and refusing the merges that look like they worked.

    python -m hermes.merge --recipe hermes/recipes/<line>/stage-c-tools.yaml
    -> merge:OK    the plan, and the command that performs it
    -> merge:STOP  every reason, not the first

A merge is the step where the training run stops being a directory of adapter weights and
becomes the model that gets served and benchmarked. Almost nothing about it fails loudly.

**PEFT applies an adapter to whatever base you hand it.** The shapes match across the whole
family, so merging stage C's adapter into the wrong checkpoint -- a different revision, a
different member of the line, last week's merge -- produces a model, not an error. It loads, it
serves, it benchmarks. Nothing downstream can tell you the weights are not what the recipe says
they are, which is why `adapter_config.json` recording `base_model_name_or_path` is the single
most useful fact in a LoRA output directory.

**An adapter directory with no weights merges to a no-op.** `adapter_config.json` alone yields
the base model back, and the base model benchmarks exactly like the base model -- which is the
honest-looking version of the worst outcome: a promotion decision made between a model and
itself, with every number in it correct.

**A 4-bit-trained adapter merges into BF16 without complaint.** That is what the previous commit
took the recipes off, and the merge is where it would have been laundered: adapters fitted to a
quantized copy, folded into weights that were never quantized, served as an improvement.

So this module reads the recipe, reads the adapter, and compares them before anything is
written. It does not perform the merge; `axolotl merge-lora` does, and `scripts/merge_lora.sh`
runs this first and refuses to call it on a non-empty verdict.

## Where the merged model lands

Axolotl writes to `{output_dir}/merged` and does not take a destination. Rather than move the
result afterwards -- a rename is one more step that can half-succeed and leave a directory whose
name no longer describes what is inside it -- the recipe that consumes a merge names that path
directly. `tests/test_base_model.py` checks the two agree, so a stage starting from a merge is a
stage starting from a merge some other stage in the same line actually produces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OK = "merge:OK"
STOP = "merge:STOP"

# What Axolotl appends to `output_dir`. Not configurable there, so not configurable here.
MERGED_DIRNAME = "merged"

# One of these must exist beside `adapter_config.json`. Both names are in use depending on the
# version that trained the adapter, and neither being present is the no-op merge above.
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


class MergeError(ValueError):
    """A merge cannot be planned from what was supplied."""


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise MergeError(f"recipe not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise MergeError(f"{path} does not parse to a mapping")
    return loaded


@dataclass(frozen=True)
class Recipe:
    """The fields of a training recipe that decide what a merge produces."""

    path: Path
    base_model: str
    output_dir: str
    adapter: str
    lora_r: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    load_in_4bit: bool
    load_in_8bit: bool

    @classmethod
    def load(cls, path: Path) -> Recipe:
        config = _load_yaml(path)
        missing = [k for k in ("base_model", "output_dir") if not config.get(k)]
        if missing:
            raise MergeError(f"{path} is missing {', '.join(missing)}")
        return cls(
            path=path,
            base_model=str(config["base_model"]),
            output_dir=str(config["output_dir"]),
            adapter=str(config.get("adapter") or ""),
            lora_r=int(config.get("lora_r") or 0),
            lora_alpha=int(config.get("lora_alpha") or 0),
            target_modules=tuple(str(m) for m in config.get("lora_target_modules") or ()),
            load_in_4bit=bool(config.get("load_in_4bit")),
            load_in_8bit=bool(config.get("load_in_8bit")),
        )

    @property
    def adapter_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def merged_dir(self) -> Path:
        return Path(self.output_dir) / MERGED_DIRNAME


@dataclass(frozen=True)
class Adapter:
    """What a trained LoRA directory says about itself."""

    path: Path
    base_model: str
    peft_type: str
    r: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    weight_files: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Adapter:
        config_path = path / "adapter_config.json"
        if not config_path.is_file():
            raise MergeError(
                f"{path} holds no adapter_config.json, so there is nothing to merge. A training run "
                "that was interrupted before saving leaves the directory behind."
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MergeError(f"{config_path} is not readable JSON: {exc}") from exc
        modules = config.get("target_modules") or ()
        return cls(
            path=path,
            # `base_model_name_or_path` is the only record of what these weights were fitted to.
            base_model=str(config.get("base_model_name_or_path") or ""),
            peft_type=str(config.get("peft_type") or ""),
            r=int(config.get("r") or 0),
            lora_alpha=int(config.get("lora_alpha") or 0),
            # A set in the JSON on some versions, a list on others; ordering is not meaningful.
            target_modules=tuple(sorted(str(m) for m in modules)),
            weight_files=tuple(sorted(n for n in ADAPTER_WEIGHT_NAMES if (path / n).is_file())),
        )


def check_weights_exist(adapter: Adapter) -> list[str]:
    """The no-op merge.

    An adapter directory holding only its config merges to the base model. Nothing raises: the
    result loads, serves, and scores exactly like the model it was supposed to improve on, so the
    promotion comparison is run between a model and itself and every number in it is correct.
    """
    if not adapter.weight_files:
        return [
            f"{adapter.path} holds adapter_config.json but none of {', '.join(ADAPTER_WEIGHT_NAMES)}. "
            "Merging that yields the base model unchanged -- which serves, benchmarks, and reads as a "
            "model that learned nothing rather than as a failed merge."
        ]
    return []


def check_base_agrees(adapter: Adapter, recipe: Recipe) -> list[str]:
    """Whether these weights were fitted to the model they are about to be folded into.

    The shapes match across the whole family, so a mismatch produces a model rather than an
    error. This is the check with the least to go on and the most to prevent: a bare string
    comparison, because the alternative is nothing.
    """
    if not adapter.base_model:
        return [
            f"{adapter.path}/adapter_config.json records no base_model_name_or_path, so there is no "
            "way to tell what these weights were fitted to. PEFT will merge them into anything with "
            "matching shapes and report success."
        ]
    if adapter.base_model != recipe.base_model:
        return [
            f"the adapter was trained on {adapter.base_model!r} and {recipe.path.name} merges into "
            f"{recipe.base_model!r}. Merging across bases succeeds silently; the result is a model "
            "whose weights are not what any recipe describes."
        ]
    return []


def check_shape_agrees(adapter: Adapter, recipe: Recipe) -> list[str]:
    """Whether this directory holds the adapter this recipe produced.

    Output directories get reused between runs, and a stale adapter under the name of a current
    recipe is the merge that silently ships an older experiment. Rank, alpha and target modules
    are what the recipe sets and the adapter records, so disagreement means the two were not
    produced together.
    """
    issues: list[str] = []
    if recipe.lora_r and adapter.r != recipe.lora_r:
        issues.append(f"adapter rank {adapter.r} is not the recipe's lora_r {recipe.lora_r}")
    if recipe.lora_alpha and adapter.lora_alpha != recipe.lora_alpha:
        issues.append(f"adapter alpha {adapter.lora_alpha} is not the recipe's lora_alpha {recipe.lora_alpha}")
    if recipe.target_modules and adapter.target_modules != tuple(sorted(recipe.target_modules)):
        issues.append(
            f"adapter targets {list(adapter.target_modules)} and the recipe targets {sorted(recipe.target_modules)}"
        )
    if issues:
        return [
            f"{adapter.path} does not look like the output of {recipe.path.name}: "
            + "; ".join(issues)
            + ". An output directory reused between runs is how an older experiment gets merged and served."
        ]
    return []


def check_precision(recipe: Recipe) -> list[str]:
    """A quantized-training recipe merging into an unquantized base.

    The adapter records nothing about this -- `adapter_config.json` describes rank and targets,
    not what the base was loaded as -- so the recipe is the only witness. Folding adapters fitted
    to a 4-bit copy into BF16 weights produces a model that serves and scores; what it does not
    produce is the model the adapters were trained for.
    """
    if recipe.load_in_4bit or recipe.load_in_8bit or recipe.adapter == "qlora":
        which = "load_in_4bit" if recipe.load_in_4bit else ("load_in_8bit" if recipe.load_in_8bit else "adapter: qlora")
        return [
            f"{recipe.path.name} sets {which}, so the adapter was fitted to a quantized copy of "
            f"{recipe.base_model!r}. Merging it into the unquantized base succeeds and yields adapters "
            "shaped around quantization noise the served model does not have."
        ]
    return []


def check_destination(recipe: Recipe, *, force: bool) -> list[str]:
    """Refuse to write over an existing merge unless told to.

    A merge that half-completes over a previous one leaves a directory that loads -- shard files
    are written independently -- and serves a mixture of two models under one name.
    """
    merged = recipe.merged_dir
    if force or not merged.exists():
        return []
    if any(merged.iterdir()):
        return [
            f"{merged} already exists and is not empty. Merging over it can leave shards from two "
            "different models in one directory, which loads and serves without complaint. Move it "
            "aside, or pass --force if it is known to be disposable."
        ]
    return []


@dataclass(frozen=True)
class Plan:
    """A merge that may proceed, and the command that performs it."""

    recipe: Recipe
    adapter: Adapter
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def command(self) -> tuple[str, ...]:
        return ("axolotl", "merge-lora", str(self.recipe.path), f"--lora-model-dir={self.adapter.path}")

    def to_record(self) -> dict[str, Any]:
        return {
            "verdict": OK if self.ok else STOP,
            "recipe": str(self.recipe.path),
            "adapter": str(self.adapter.path),
            "base_model": self.recipe.base_model,
            "merged_dir": str(self.recipe.merged_dir),
            "command": list(self.command),
            "issues": list(self.issues),
        }


def plan(recipe_path: Path, *, adapter_dir: Path | None = None, force: bool = False) -> Plan:
    """Every reason this merge would produce something other than what the recipe describes.

    All of them, not the first: a merge is expensive enough that learning one problem per attempt
    means learning them an hour apart.
    """
    recipe = Recipe.load(recipe_path)
    adapter = Adapter.load(adapter_dir if adapter_dir is not None else recipe.adapter_dir)
    issues = [
        *check_weights_exist(adapter),
        *check_base_agrees(adapter, recipe),
        *check_shape_agrees(adapter, recipe),
        *check_precision(recipe),
        *check_destination(recipe, force=force),
    ]
    return Plan(recipe=recipe, adapter=adapter, issues=tuple(issues))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, type=Path, help="the training recipe whose adapter to merge")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="the adapter directory; defaults to the recipe's own output_dir",
    )
    parser.add_argument("--force", action="store_true", help="allow writing over an existing merged directory")
    parser.add_argument("--json", action="store_true", help="emit the plan as a record")
    args = parser.parse_args(argv)

    try:
        result = plan(args.recipe, adapter_dir=args.adapter, force=args.force)
    except MergeError as exc:
        print(f"{STOP} {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_record(), indent=2))
    elif result.ok:
        print(f"{OK} {result.recipe.path.name}: {result.adapter.path} -> {result.recipe.merged_dir}")
        print(" ".join(result.command))
    else:
        print(f"{STOP} {result.recipe.path.name}")
        for issue in result.issues:
            print(f"  - {issue}")
    return 0 if result.ok else 1


__all__ = [
    "ADAPTER_WEIGHT_NAMES",
    "MERGED_DIRNAME",
    "OK",
    "STOP",
    "Adapter",
    "MergeError",
    "Plan",
    "Recipe",
    "check_base_agrees",
    "check_destination",
    "check_precision",
    "check_shape_agrees",
    "check_weights_exist",
    "main",
    "plan",
]


if __name__ == "__main__":
    raise SystemExit(main())

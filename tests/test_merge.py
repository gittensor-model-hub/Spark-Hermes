"""The merges that succeed and produce the wrong model.

Every case here loads, serves and benchmarks. None of them raises in PEFT, and none is visible
in the result: a merged directory looks the same whether the adapter belonged to it or not. That
is the whole reason this preflight exists, so the tests are written as the specific way each
merge lies rather than as "returns a non-empty list".

The sharpest one is `test_an_adapter_with_no_weights_would_merge_to_the_base_model`. That merge
produces the base back, the base benchmarks exactly like the base, and the M0/M1 comparison is
then run between a model and itself with every number in it correct.
"""

import json

import pytest
import yaml

from hermes.merge import (
    MERGED_DIRNAME,
    OK,
    STOP,
    Adapter,
    MergeError,
    Recipe,
    check_base_agrees,
    check_precision,
    check_shape_agrees,
    main,
    plan,
)

BASE = "Qwen/Qwen3.6-27B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _recipe(tmp_path, **overrides):
    config = {
        "base_model": BASE,
        "base_model_revision": "a" * 40,
        "output_dir": str(tmp_path / "outputs" / "stage-c"),
        "adapter": "lora",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_target_modules": list(TARGETS),
    }
    config.update(overrides)
    path = tmp_path / "stage-c-tools.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _adapter(tmp_path, *, weights=True, **overrides):
    config = {
        "base_model_name_or_path": BASE,
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 64,
        # Reversed on purpose: PEFT writes these from a set, so order carries no meaning and a
        # comparison that depended on it would fail on a correct adapter.
        "target_modules": list(reversed(TARGETS)),
    }
    config.update(overrides)
    directory = tmp_path / "outputs" / "stage-c"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (directory / "adapter_model.safetensors").write_bytes(b"\x00")
    return directory


# --- the merge that may proceed -------------------------------------------------------------------


def test_a_matching_adapter_merges(tmp_path):
    recipe = _recipe(tmp_path)
    _adapter(tmp_path)
    result = plan(recipe)
    assert result.ok and result.issues == ()
    assert result.recipe.merged_dir.name == MERGED_DIRNAME
    assert result.command[:2] == ("axolotl", "merge-lora")


def test_target_module_order_is_not_a_mismatch(tmp_path):
    """PEFT serialises `target_modules` from a set. A comparison that kept order would refuse
    every correct adapter, which is the failure mode that gets a check deleted."""
    _adapter(tmp_path, target_modules=["v_proj", "o_proj", "q_proj", "k_proj"])
    assert plan(_recipe(tmp_path)).ok


# --- the no-op merge ----------------------------------------------------------------------------


def test_an_adapter_with_no_weights_would_merge_to_the_base_model(tmp_path):
    """The result is the base model: it loads, serves, and scores exactly like the model that was
    supposed to be improved. Nothing downstream distinguishes it from a model that learned
    nothing, so the M0/M1 comparison would be run between a model and itself."""
    recipe = _recipe(tmp_path)
    _adapter(tmp_path, weights=False)
    issues = plan(recipe).issues
    assert issues and "yields the base model unchanged" in issues[0]


def test_a_bin_adapter_is_accepted_too(tmp_path):
    """Older runs wrote `adapter_model.bin`. Refusing those would refuse a valid merge."""
    directory = _adapter(tmp_path, weights=False)
    (directory / "adapter_model.bin").write_bytes(b"\x00")
    assert plan(_recipe(tmp_path)).ok


def test_a_directory_with_no_adapter_config_is_an_error_not_a_verdict(tmp_path):
    """Nothing to check against. An interrupted run leaves the directory behind, so this is a
    real case and not a hypothetical."""
    recipe = _recipe(tmp_path)
    (tmp_path / "outputs" / "stage-c").mkdir(parents=True)
    with pytest.raises(MergeError, match="nothing to merge"):
        plan(recipe)


# --- merging into the wrong base ------------------------------------------------------------------


def test_an_adapter_from_another_base_is_refused(tmp_path):
    """The shapes match across the family, so this merge succeeds and produces a model whose
    weights correspond to no recipe."""
    _adapter(tmp_path, base_model_name_or_path="Qwen/Qwen3.6-14B")
    issues = plan(_recipe(tmp_path)).issues
    assert any("Merging across bases succeeds silently" in i for i in issues)


def test_an_adapter_that_records_no_base_is_refused(tmp_path):
    """`base_model_name_or_path` is the only record of what the weights were fitted to. Missing,
    there is nothing to compare, and PEFT will merge into anything with matching shapes."""
    _adapter(tmp_path, base_model_name_or_path="")
    assert check_base_agrees(Adapter.load(tmp_path / "outputs" / "stage-c"), Recipe.load(_recipe(tmp_path)))


# --- the stale output directory --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("r", 16, "rank"),
        ("lora_alpha", 32, "alpha"),
        ("target_modules", ["q_proj", "k_proj"], "targets"),
    ],
)
def test_an_adapter_from_a_different_config_is_refused(tmp_path, field, value, expected):
    """Output directories get reused between runs, and a stale adapter under a current recipe's
    name is the merge that ships last week's experiment."""
    _adapter(tmp_path, **{field: value})
    issues = check_shape_agrees(Adapter.load(tmp_path / "outputs" / "stage-c"), Recipe.load(_recipe(tmp_path)))
    assert issues and expected in issues[0]


def test_a_recipe_that_sets_no_rank_does_not_claim_a_mismatch(tmp_path):
    """Absence of a setting is not a disagreement with one. A recipe without `lora_r` takes the
    default, and comparing the adapter's real rank against a zero would refuse every merge."""
    recipe = Recipe.load(_recipe(tmp_path, lora_r=None, lora_alpha=None, lora_target_modules=None))
    assert check_shape_agrees(Adapter.load(_adapter(tmp_path)), recipe) == []


# --- quantized training, unquantized merge -----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{"load_in_4bit": True}, {"load_in_8bit": True}, {"adapter": "qlora"}],
)
def test_a_quantized_recipe_may_not_merge_into_the_unquantized_base(tmp_path, overrides):
    """The adapter records nothing about how the base was loaded, so the recipe is the only
    witness. This is the failure the 27B stages were just taken off, and the merge is where it
    would have been laundered into a model that serves and scores."""
    _adapter(tmp_path)
    issues = check_precision(Recipe.load(_recipe(tmp_path, **overrides)))
    assert issues and "quantization noise the served model does not have" in issues[0]


# --- the destination ------------------------------------------------------------------------------------


def test_an_existing_merge_is_not_written_over(tmp_path):
    """Shards are written independently, so a half-finished merge over a previous one leaves a
    directory holding two models that loads and serves without complaint."""
    recipe = _recipe(tmp_path)
    _adapter(tmp_path)
    merged = Recipe.load(recipe).merged_dir
    merged.mkdir(parents=True)
    (merged / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
    assert any("already exists and is not empty" in i for i in plan(recipe).issues)
    assert plan(recipe, force=True).ok


def test_an_empty_destination_directory_is_not_an_obstacle(tmp_path):
    recipe = _recipe(tmp_path)
    _adapter(tmp_path)
    Recipe.load(recipe).merged_dir.mkdir(parents=True)
    assert plan(recipe).ok


# --- reporting -------------------------------------------------------------------------------------------


def test_every_reason_is_reported_not_just_the_first(tmp_path):
    """A merge takes long enough that one problem per attempt means learning them an hour apart."""
    recipe = _recipe(tmp_path, load_in_4bit=True, lora_r=8)
    _adapter(tmp_path, weights=False, base_model_name_or_path="mistralai/Mistral-7B-v0.3")
    assert len(plan(recipe).issues) >= 4


def test_the_cli_exits_non_zero_on_a_refusal(tmp_path, capsys):
    """The shell script decides whether to call axolotl on this status, so a refusal that exits 0
    is a refusal that merges anyway."""
    recipe = _recipe(tmp_path)
    _adapter(tmp_path, weights=False)
    assert main(["--recipe", str(recipe)]) == 1
    assert STOP in capsys.readouterr().out


def test_the_cli_prints_the_command_it_would_run(tmp_path, capsys):
    recipe = _recipe(tmp_path)
    adapter = _adapter(tmp_path)
    assert main(["--recipe", str(recipe)]) == 0
    out = capsys.readouterr().out
    assert OK in out and "axolotl merge-lora" in out and str(adapter) in out


def test_the_json_record_carries_the_paths_the_script_uses(tmp_path, capsys):
    """`scripts/merge_lora.sh` reads `adapter` and `merged_dir` from here rather than parsing the
    human output, so both have to be present in the record."""
    recipe = _recipe(tmp_path)
    _adapter(tmp_path)
    main(["--recipe", str(recipe), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert record["verdict"] == OK
    assert record["merged_dir"].endswith(MERGED_DIRNAME)
    assert record["adapter"] and record["base_model"] == BASE


def test_a_missing_recipe_is_an_error_with_the_path(tmp_path, capsys):
    assert main(["--recipe", str(tmp_path / "nope.yaml")]) == 1
    assert "recipe not found" in capsys.readouterr().out


# --- the real recipes ---------------------------------------------------------------------------------------


def test_the_shipped_recipes_would_pass_the_precision_check():
    """The check that would have caught the QLoRA-into-BF16 merge, run against what is committed."""
    from pathlib import Path

    for path in sorted(Path("hermes/recipes/spark-hermes-agent-3.8-27b").glob("stage-*.yaml")):
        assert check_precision(Recipe.load(path)) == [], path.name

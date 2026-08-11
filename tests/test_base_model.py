"""The base model is pinned to a revision, and the recipes agree with the pin."""

import json
from pathlib import Path

import pytest
import yaml

from hermes.base_model import PIN_PATH, BaseModelError, load

RECIPE_DIR = Path("hermes/recipes/spark-hermes-agent-3.8-27b")


def _recipes():
    return sorted(RECIPE_DIR.glob("stage-*.yaml"))


def test_the_pin_loads_and_names_a_real_commit():
    pin = load()
    assert pin.repository == "Qwen/Qwen3.6-27B"
    assert len(pin.revision) == 40


def test_a_movable_ref_is_refused(tmp_path):
    """`main` resolves to whatever the repository holds when someone runs it. Two runs
    could then agree on every other digest this project computes and still have trained on
    different weights -- the same refusal eval.hf_pin already makes on the mining side."""
    bad = tmp_path / "pin.json"
    bad.write_text(json.dumps({**json.loads(PIN_PATH.read_text()), "revision": "main"}))
    with pytest.raises(BaseModelError):
        load(bad)


def test_a_short_sha_is_refused(tmp_path):
    bad = tmp_path / "pin.json"
    bad.write_text(json.dumps({**json.loads(PIN_PATH.read_text()), "revision": "6a9e13bd6fc8"}))
    with pytest.raises(BaseModelError):
        load(bad)


# A stage may start from an earlier stage's merged output instead of from the hub -- DPO needs the
# policy it is improving as its reference, and after the SFT stages that policy is the merged model,
# not the raw base. Such a recipe names a local path, so there is nothing to pin it against
# directly; what keeps it honest is that the path lies inside this model line's own output tree,
# every hub-rooted recipe in which is pinned by the test below. The chain therefore roots at the pin.
DERIVED_PREFIX = f"outputs/{RECIPE_DIR.name}/"


def _is_derived(base_model: str) -> bool:
    return base_model.startswith("outputs/")


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_every_recipe_matches_the_pin(recipe):
    """A recipe drifting from the pin is the failure the pin exists to prevent, and it
    would be invisible: both files would still name a real model."""
    pin = load()
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    if _is_derived(base):
        pytest.skip("derived base; covered by test_a_derived_base_stays_inside_this_model_line")
    assert base == pin.repository
    assert config["base_model_revision"] == pin.revision


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_a_derived_base_stays_inside_this_model_line(recipe):
    """The escape hatch above, held shut.

    `outputs/` as a prefix is not itself an assurance -- it would admit any other line's checkpoint,
    or one built from an unpinned base, and the resulting model would still train and still serve.
    So a derived base must sit under this recipe directory's own output tree, whose every other
    stage is pinned.

    It must also carry no `base_model_revision`: a local directory has no hub revision, and a
    stamped one would agree with the pin while describing something the pin never produced."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    if not _is_derived(base):
        pytest.skip("hub base; covered by test_every_recipe_matches_the_pin")
    assert base.startswith(DERIVED_PREFIX), f"{base!r} is outside {DERIVED_PREFIX!r}"
    assert "base_model_revision" not in config, "a local path has no hub revision to stamp"


def test_at_most_one_stage_starts_from_a_derived_base():
    """Named rather than counted loosely, so a second recipe going off-pin fails here instead of
    silently joining the exception."""
    derived = {
        r.name for r in _recipes() if _is_derived(str(yaml.safe_load(r.read_text(encoding="utf-8"))["base_model"]))
    }
    assert derived == {"stage-d-preference.yaml"}


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_no_recipe_trains_against_the_unpublished_model(recipe):
    """Qwen3.8-27B is the Phase 1 target and is not published. The only repositories under
    that name are third-party derivatives with no official base, so a recipe pointing at it
    cannot be pinned and would 404 for anyone who ran it.

    Checked on the parsed value, not the file text: the comments explain why 3.8 is not
    used yet, and a test that forbade naming it would forbid saying so.

    Matched on the vendor name rather than on the bare version. This project's own model line is
    called spark-hermes-agent-3.8-27b -- the target it is aimed at, not the base it trains from --
    so every output path under it contains "3.8", and a substring check on the version alone
    failed the first recipe to start from one of those paths, reporting our own directory name as
    an unpublished Qwen release."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    assert "qwen3.8" not in base.lower()


def test_the_dialect_is_recorded_with_its_evidence():
    """Hermes is upstream; the model is what adapts. So the dialect is established by
    reading the model's own template, not chosen for it."""
    pin = load()
    assert pin.hermes_dialect == "hermes-4"
    evidence = pin.raw["hermes_dialect_evidence"]
    assert "<tool_call>" in evidence and "<think>" in evidence
    assert "<scratch_pad>" in evidence  # named as the thing it does NOT emit


def test_the_dialect_names_one_this_repo_implements():
    from hermes.protocol import DIALECTS

    assert load().hermes_dialect in DIALECTS


def test_multimodal_is_recorded_because_it_is_easy_to_miss():
    """A Qwen3_5ForConditionalGeneration keeps its text hyperparameters under
    `text_config`; anything reading the top level of config.json gets None for every
    field and computes a memory budget out of nothing."""
    pin = load()
    assert pin.multimodal is True
    assert pin.raw["text_config"]["num_hidden_layers"] == 64
    assert pin.raw["text_config"]["num_key_value_heads"] == 4
    assert pin.raw["text_config"]["head_dim"] == 256


def test_kv_bytes_count_only_the_full_attention_layers():
    """This model is hybrid, so its KV cache lives in 16 of 64 layers, not all of them.

    The previous version of this test asserted
    `2 * num_hidden_layers * kv_heads * head_dim` and passed -- which is exactly how the
    wrong figure shipped. The formula and the pin agreed with each other, and neither
    described the model. `layer_types` in the published config is 48 `linear_attention` +
    16 `full_attention`; the linear layers hold a fixed per-sequence recurrent state rather
    than a per-token cache.

    Getting it wrong is not cosmetic. `kv_bytes()` sizes a VRAM budget against a card, and
    counting all 64 layers overstates the per-token cost by 4x, which rules out hardware that
    in fact runs this model comfortably. Measured against a live vLLM server on an RTX PRO
    6000 Blackwell: 33.36 GiB of KV pool held 507,539 tokens.
    """
    pin = load()
    t = pin.raw["text_config"]
    full = t["layer_types_counts"]["full_attention"]
    linear = t["layer_types_counts"]["linear_attention"]

    assert full + linear == t["num_hidden_layers"]
    assert full == t["num_hidden_layers"] // t["full_attention_interval"]

    expected = 2 * full * t["num_key_value_heads"] * t["head_dim"]
    assert pin.kv_bytes_per_token["bf16"] == expected * 2
    assert pin.kv_bytes_per_token["fp8"] == expected

    # Name the figure the all-layers formula would produce, so an edit that reintroduces it
    # fails here rather than silently quadrupling every budget again.
    all_layers = 2 * t["num_hidden_layers"] * t["num_key_value_heads"] * t["head_dim"] * 2
    assert pin.kv_bytes_per_token["bf16"] != all_layers


def test_the_measured_kv_figure_is_recorded_but_is_not_what_budgets_multiply():
    """The measurement includes per-sequence recurrent state, which does not scale with
    context -- so multiplying it by a context length would be the wrong kind of wrong."""
    pin = load()
    measured = pin.raw["kv_bytes_measured"]
    assert measured["gpu_kv_cache_tokens"] > 0
    assert pin.kv_bytes_per_token["bf16"] < measured["implied_bytes_per_token"]
    assert "does not scale with context length" in measured["note"]


def test_kv_bytes_for_a_peak_context():
    pin = load()
    # 200K tokens of fp8 KV over the 16 full-attention layers is ~6.6 GB, so a 200K peak
    # context is comfortable on one 96 GB card alongside 51 GB of bf16 weights. Under the
    # old all-layers figure this read 26 GB and looked marginal.
    assert round(pin.kv_bytes(200_000, dtype="fp8") / 1e9, 1) == 6.6


def test_an_unknown_dtype_is_refused_rather_than_assumed():
    with pytest.raises(BaseModelError, match="no KV size recorded"):
        load().kv_bytes(1000, dtype="int2")

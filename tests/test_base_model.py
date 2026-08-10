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


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_every_recipe_matches_the_pin(recipe):
    """A recipe drifting from the pin is the failure the pin exists to prevent, and it
    would be invisible: both files would still name a real model."""
    pin = load()
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    assert config["base_model"] == pin.repository
    assert config["base_model_revision"] == pin.revision


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_no_recipe_trains_against_the_unpublished_model(recipe):
    """Qwen3.8-27B is the Phase 1 target and is not published. The only repositories under
    that name are third-party derivatives with no official base, so a recipe pointing at it
    cannot be pinned and would 404 for anyone who ran it.

    Checked on the parsed value, not the file text: the comments explain why 3.8 is not
    used yet, and a test that forbade naming it would forbid saying so."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    assert "3.8" not in str(config["base_model"])


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


def test_kv_bytes_match_the_recorded_architecture():
    """2 x layers x kv_heads x head_dim, so a budget can be checked without a network call."""
    pin = load()
    t = pin.raw["text_config"]
    expected = 2 * t["num_hidden_layers"] * t["num_key_value_heads"] * t["head_dim"]
    assert pin.kv_bytes_per_token["bf16"] == expected * 2
    assert pin.kv_bytes_per_token["fp8"] == expected


def test_kv_bytes_for_a_peak_context():
    pin = load()
    # 200K tokens of fp8 KV is ~26 GB, which is what makes a 200K peak context feasible
    # at Q4 and impossible at BF16 on one 96 GB card.
    assert round(pin.kv_bytes(200_000, dtype="fp8") / 1e9) == 26


def test_an_unknown_dtype_is_refused_rather_than_assumed():
    with pytest.raises(BaseModelError, match="no KV size recorded"):
        load().kv_bytes(1000, dtype="int2")

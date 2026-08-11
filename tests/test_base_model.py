"""The base model is pinned to a revision, and the recipes agree with the pin."""

import json
from pathlib import Path

import pytest
import yaml

from hermes.base_model import PIN_PATH, BaseModelError, load
from hermes.merge import MERGED_DIRNAME

RECIPE_DIR = Path("hermes/recipes/spark-hermes-glimmer-30b")


def _recipes():
    return sorted(RECIPE_DIR.glob("stage-*.yaml"))


def test_the_pin_loads_and_names_a_real_commit():
    pin = load()
    assert pin.repository == "meta-models/Muse-Glimmer-30B"
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
# directly; what keeps it honest is that the path is one some other stage in this line actually
# produces, and every one of those is pinned. The chain therefore roots at the pin.
def _is_derived(base_model: str) -> bool:
    return base_model.startswith("outputs/")


def _merge_outputs() -> set[str]:
    """Every path a merge of a stage in this line would write to."""
    out = set()
    for recipe in _recipes():
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        if config.get("output_dir") and not _is_derived(str(config["base_model"])):
            out.add(f"{config['output_dir'].rstrip('/')}/{MERGED_DIRNAME}")
    return out


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
    So the path must be one that merging a pinned stage of this same line actually produces:
    that stage's `output_dir` with Axolotl's `merged` appended. A path nothing produces is a
    recipe that cannot run, and one produced by an unpinned stage is off the pin by a hop.

    It must also carry no `base_model_revision`: a local directory has no hub revision, and a
    stamped one would agree with the pin while describing something the pin never produced."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    if not _is_derived(base):
        pytest.skip("hub base; covered by test_every_recipe_matches_the_pin")
    produced = _merge_outputs()
    assert base in produced, f"{base!r} is not produced by any pinned stage here; those write {sorted(produced)}"
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

    Matched on the vendor name rather than on the bare version, which mattered while the model
    line was called spark-hermes-agent-3.8-27b: every output path under it contained "3.8", and a
    substring check on the version alone reported our own directory name as an unpublished Qwen
    release. The line is spark-hermes-glimmer-30b now and the check is kept anyway -- it costs
    nothing and the pin is one edit away from naming an unpublished model again."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    assert "qwen3.8" not in base.lower()


def test_the_dialect_is_recorded_with_its_evidence():
    """Hermes is upstream; the model is what adapts. So the dialect is established by
    reading the model's own template, not chosen for it -- and this base does not speak Hermes.

    The evidence has to name the markup the parser depends on AND the Hermes markup the model
    does not emit. Recording only the first would let a pin claim `atem` for a model whose
    template happens to mention an atem tag in a comment, and recording only the second would
    say what it is not."""
    pin = load()
    assert pin.hermes_dialect == "atem"
    evidence = pin.raw["hermes_dialect_evidence"]
    for marker in ("<atem:function_calls>", "<atem:invoke", "<atem:parameter", "<tool_output"):
        assert marker in evidence, marker
    # Named as the things it does NOT emit, which is why hermes/atem.py exists at all.
    assert "<tool_call>" in evidence and "<think>" in evidence
    assert "does not speak Hermes" in evidence


def test_the_dialect_names_one_this_repo_implements():
    from hermes.protocol import DIALECTS

    assert load().hermes_dialect in DIALECTS


def test_multimodal_is_recorded_because_it_is_easy_to_miss():
    """A MuseGlimmerForConditionalGeneration keeps its text hyperparameters under `text_config`;
    the top level of config.json carries only vision and projector keys, so anything reading it
    for hidden_size or num_hidden_layers gets None and computes a memory budget out of nothing.

    Two base models in a row have had this shape, which is why it is asserted rather than noted.
    `AutoModelForCausalLM` also refuses this config outright -- the class is
    `AutoModelForImageTextToText`, and finding that out cost a load attempt."""
    pin = load()
    assert pin.multimodal is True
    assert pin.raw["text_config"]["num_hidden_layers"] == 52
    assert pin.raw["text_config"]["num_key_value_heads"] == 2
    assert pin.raw["text_config"]["head_dim"] == 128
    assert "hidden_size" not in {k for k in pin.raw if k != "text_config"}


def test_kv_bytes_count_only_the_full_attention_layers():
    """This model is hybrid, so its KV cache lives in 13 of 52 layers, not all of them.

    The previous version of this test asserted
    `2 * num_hidden_layers * kv_heads * head_dim` and passed -- which is exactly how the
    wrong figure shipped. The formula and the pin agreed with each other, and neither
    described the model. `layer_types` in the published config is 39 `sliding_attention` +
    13 `full_attention`; a sliding layer holds at most `sliding_window` tokens rather than
    growing with the context, so it belongs in a bounded per-sequence budget and not in a
    figure that gets multiplied by a context length.

    The previous base was hybrid too, with 48 linear-attention layers instead of sliding ones.
    The mechanism differs and the arithmetic lesson is identical, which is why this test survived
    the base model changing underneath it.

    Getting it wrong is not cosmetic: `kv_bytes()` sizes a VRAM budget against a card, and
    counting all 52 layers overstates the per-token cost by 4x, which rules out hardware that in
    fact runs this model comfortably.
    """
    pin = load()
    t = pin.raw["text_config"]
    counts = t["layer_types_counts"]
    full = counts["full_attention"]

    assert sum(counts.values()) == t["num_hidden_layers"]
    assert t["sliding_window"] > 0, "a bounded layer needs its bound recorded"

    expected = 2 * full * t["num_key_value_heads"] * t["head_dim"]
    assert pin.kv_bytes_per_token["bf16"] == expected * 2
    assert pin.kv_bytes_per_token["fp8"] == expected

    # Name the figure the all-layers formula would produce, so an edit that reintroduces it
    # fails here rather than silently quadrupling every budget again.
    all_layers = 2 * t["num_hidden_layers"] * t["num_key_value_heads"] * t["head_dim"] * 2
    assert pin.kv_bytes_per_token["bf16"] != all_layers


def test_an_unmeasured_kv_figure_is_null_rather_than_inherited():
    """The previous pin carried a figure measured on a live vLLM server -- 33.36 GiB of KV pool
    holding 507,539 tokens. That described Qwen3.6-27B. Carrying it forward would have made a
    measurement of one model read as a measurement of another, which is worse than having none:
    a stale number is indistinguishable from a fresh one and nothing downstream can tell."""
    pin = load()
    assert pin.raw["kv_bytes_measured"] is None
    assert "not transferable" in pin.raw["kv_bytes_measured_note"]


def test_kv_bytes_for_a_peak_context():
    pin = load()
    # 131K tokens -- this model's full context -- of fp8 KV over the 13 full-attention layers is
    # ~0.9 GB. Two KV heads is aggressive GQA and is most of why that is so small for a 30B: the
    # whole context costs less than 1 GB beside ~60 GB of bf16 weights on a 96 GB card. Under an
    # all-52-layers figure it would read 3.5 GB, which is still comfortable and still wrong.
    assert round(pin.kv_bytes(131_072, dtype="fp8") / 1e9, 1) == 0.9
    assert pin.kv_bytes(131_072, dtype="bf16") == 2 * pin.kv_bytes(131_072, dtype="fp8")


def test_an_unknown_dtype_is_refused_rather_than_assumed():
    with pytest.raises(BaseModelError, match="no KV size recorded"):
        load().kv_bytes(1000, dtype="int2")

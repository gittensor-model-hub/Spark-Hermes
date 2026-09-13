"""Labels that feed a payout, and the gap this module refuses to hide.

`.gittensor/weights.json` is the only declaration of what a label is worth. Every test here is
about keeping that true: no second copy of the multipliers, no invented tier, and no silent
zero where nobody has actually decided.
"""

from __future__ import annotations

import json

import pytest

from eval.pr_labels import (
    AREA_PATHS,
    LabelError,
    area_labels,
    check_areas_match_weights,
    emission_multiplier,
    report,
    unknown_emission_labels,
)

# --- area labels ----------------------------------------------------------------------


def test_area_labels_are_a_pure_function_of_paths():
    assert area_labels(["teacher/providers.py"]) == ["area:teacher"]
    assert area_labels(["proof/bundle.py", "proof/publish.py"]) == ["area:proof"]


def test_the_same_diff_always_earns_the_same_labels():
    """Deterministic, because a number feeding a payout must not depend on iteration order."""
    diff = ["eval/score.py", "teacher/providers.py", "hermes/recipes/a.yaml", "eval/verify.py"]
    assert area_labels(diff) == area_labels(list(reversed(diff))) == ["area:eval", "area:recipes", "area:teacher"]


def test_a_path_under_no_declared_area_earns_nothing():
    """Not a gap. weights.json says tooling, benchmarks, docs and refactors carry no
    model-quality credit, so labelling them would assert a category the reward model lacks."""
    assert area_labels(["README.md", "docs/zenith.md", ".github/workflows/ci.yml", "scripts/install.sh"]) == []


def test_both_recipe_roots_count_as_recipes():
    """`recipes/` is the 4B student line and `hermes/recipes/` is the agent line. Both are
    recipes to the weights file, which names the area in prose and cannot distinguish them."""
    assert area_labels(["recipes/qwen3.5-4b-phase1/sft.yaml"]) == ["area:recipes"]
    assert area_labels(["hermes/recipes/spark-hermes-3.8-27b/stage-a-reasoning.yaml"]) == ["area:recipes"]


def test_leading_dot_slash_and_blank_lines_survive_a_git_diff_pipe():
    assert area_labels(["./teacher/x.py", "", "   ", "proof/y.py"]) == ["area:proof", "area:teacher"]


# --- the multiplier table is read, never copied -----------------------------------------


def test_multipliers_come_from_the_weights_file():
    """A second copy would drift silently: labels keep being applied, multipliers keep being
    read from the file, and the disagreement only shows up as someone paid the wrong amount."""
    declared = json.load(open(".gittensor/weights.json"))["label_multipliers"]
    for label, value in declared.items():
        if label.startswith("_"):
            continue
        assert emission_multiplier(label) == pytest.approx(float(value))


def test_an_unpriced_label_is_none_not_zero():
    """`eval:none` is priced AT zero -- judged, earned nothing. An unpriced label means nobody
    has decided. Collapsing the two would pay a pending decision as a settled zero."""
    assert emission_multiplier("eval:none") == 0.0
    assert emission_multiplier("strategy:ACCEPT") is None


def test_a_missing_weights_file_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(LabelError, match="declared there and nowhere else"):
        emission_multiplier("eval:XL", tmp_path / "absent.json")


# --- the gap this module exists to keep loud --------------------------------------------


def test_the_strategy_track_emits_labels_the_table_does_not_price():
    """The live gap, asserted so it cannot be forgotten.

    `eval/strategy_track.py` emits `strategy:ACCEPT` / `strategy:REJECT`, and `strategy:*` is
    absent from `label_multipliers`. A miner whose surface wins a round and merges with
    `strategy:ACCEPT` scores against no multiplier at all. What a won round is worth is a policy
    decision with payout consequences; this test fails the day someone prices it, which is the
    right moment to revisit this file rather than discover the change later.
    """
    multipliers = json.load(open(".gittensor/weights.json"))["label_multipliers"]
    assert not any(key.startswith("strategy:") for key in multipliers), (
        "strategy:* is now priced -- update eval/pr_labels.py and the strategy track's labelling"
    )


def test_emission_shaped_labels_outside_the_table_are_reported():
    assert unknown_emission_labels(["eval:XL", "eval:INVENTED", "dataset:xl", "dataset:huge"]) == [
        "dataset:huge",
        "eval:INVENTED",
    ]


def test_area_labels_are_never_reported_as_unpriced():
    """They are declared `Categorization only ... NOT emission weights`, so an absent multiplier
    is correct for them and must not read as a missing decision."""
    result = report(["area:eval", "area:proof"])
    assert result["unpriced"] == []
    assert result["area_only"] == ["area:eval", "area:proof"]


def test_report_separates_what_is_priced_from_what_is_pending():
    result = report(["eval:L", "area:teacher", "eval:INVENTED"])
    assert result["priced"] == {"eval:L": 5.0}
    assert result["area_only"] == ["area:teacher"]
    assert result["unpriced"] == ["eval:INVENTED"]


# --- the two files cannot drift apart ---------------------------------------------------


def test_area_paths_matches_the_areas_weights_declares():
    """The halves live in different files by necessity -- weights describes an area in prose, and
    a diff can only be matched on paths. An area declared there and unmapped here is never
    applied; one mapped here and absent there applies a label nothing recognises."""
    assert check_areas_match_weights() == []
    declared = {k for k in json.load(open(".gittensor/weights.json"))["areas"] if not k.startswith("_")}
    assert set(AREA_PATHS) == declared

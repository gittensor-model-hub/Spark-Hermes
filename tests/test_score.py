from eval.benchmarks import BENCHMARKS
from eval.score import score

# The regression labels documented in docs/miner-guide.md. Kept here as the source of
# truth so a benchmark key/label drift (issue #65) fails CI instead of shipping a label
# the docs never declared.
DOCUMENTED_REGRESSION_LABELS = {
    "regression-bfcl",
    "regression-gsm8k",
    "regression-humaneval",
    "regression-ifeval",
    "regression-mmlu-pro",
    "regression-aime24",
    "regression-gpqa-diamond",
    "regression-triton",
}


def test_every_benchmark_regression_label_is_documented():
    emitted = {f"regression-{b.label_slug}" for b in BENCHMARKS.values()}
    assert emitted == DOCUMENTED_REGRESSION_LABELS


def test_regression_labels_use_hyphenated_slugs_not_lm_eval_keys():
    for b in BENCHMARKS.values():
        assert "_" not in b.label_slug, f"{b.key} label_slug leaks an lm-eval key: {b.label_slug}"


def test_score_emits_documented_slug_for_underscored_keys():
    # mmlu_pro and gpqa_diamond_cot_zeroshot keys must surface as the hyphenated labels.
    candidate = {"mmlu_pro": 0.70, "gpqa_diamond_cot_zeroshot": 0.40}
    frontier = {"mmlu_pro": 0.80, "gpqa_diamond_cot_zeroshot": 0.50}
    report = score(candidate, frontier)
    assert "regression-mmlu-pro" in report["regressions"]
    assert "regression-gpqa-diamond" in report["regressions"]
    assert set(report["regressions"]) <= DOCUMENTED_REGRESSION_LABELS


def test_score_improvement_gets_expected_tier_from_triton():
    candidate = {"triton": 0.71, "gsm8k": 0.80}
    frontier = {"triton": 0.60, "gsm8k": 0.80}
    report = score(candidate, frontier)
    assert report["label"] == "eval:XL"  # (0.71-0.60)/0.60 = 18.3% -> XL band
    assert report["best_benchmark"] == "triton"
    assert report["regressions"] == []


def test_score_gsm8k_improvement_alone_does_not_tier():
    candidate = {"gsm8k": 0.90, "triton": 0.60}
    frontier = {"gsm8k": 0.80, "triton": 0.60}
    report = score(candidate, frontier)
    assert report["label"] == "eval:none"
    assert report["best_benchmark"] == "triton"
    assert report["best_pct_delta"] == 0.0


def test_score_rejects_on_gsm8k_regression_even_when_triton_improves():
    candidate = {"triton": 0.71, "gsm8k": 0.50}
    frontier = {"triton": 0.60, "gsm8k": 0.60}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-gsm8k" in report["regressions"]


def test_score_triton_tiers_at_l_and_m():
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    # L band: >= 10% relative improvement on triton
    l_report = score({"triton": 0.471, "gsm8k": 0.6}, frontier)
    assert l_report["label"] == "eval:L"
    # M band: >= 6%
    m_report = score({"triton": 0.454, "gsm8k": 0.6}, frontier)
    assert m_report["label"] == "eval:M"


def test_score_rejects_on_regression_beyond_floor():
    candidate = {"gsm8k": 0.88, "humaneval": 0.70, "triton": 0.60}
    frontier = {"gsm8k": 0.80, "humaneval": 0.80, "triton": 0.60}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-humaneval" in report["regressions"]


def test_score_none_below_minimum_tier():
    candidate = {"triton": 0.61, "gsm8k": 0.80}
    frontier = {"triton": 0.60, "gsm8k": 0.80}
    report = score(candidate, frontier)
    assert report["label"] == "eval:none"


def test_score_rejects_percentage_unit_scores():
    # Guarding the convention: 0-100 percentages must fail loudly, not silently mis-tier.
    import pytest

    with pytest.raises(ValueError, match=r"fractions in \[0, 1\]"):
        score({"gsm8k": 88.0}, {"gsm8k": 80.0})

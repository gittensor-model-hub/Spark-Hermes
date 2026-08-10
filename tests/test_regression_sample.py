import pytest

from eval.gsm8k_eval import normalize_gsm8k_answer
from eval.regression_sample import (
    REGRESSION_BENCHMARK_KEY,
    REGRESSION_PROBLEM_COUNT,
    build_regression_sample,
    check_gsm8k_no_regression,
    load_regression_problems,
    verify_regression_sample,
)


def _all_correct_responses():
    return [
        {
            "problem_id": int(row["problem_id"]),
            "model_response": f"work\n#### {row['answer'].split('####')[-1].strip()}",
        }
        for row in load_regression_problems()
    ]


def test_normalize_gsm8k_answer_strips_markers_and_currency():
    assert normalize_gsm8k_answer("reasoning\n#### 18") == "18"
    assert normalize_gsm8k_answer("$70,000.") == "70000"


def test_build_regression_sample_exact_match():
    sample = build_regression_sample(_all_correct_responses())
    assert sample["benchmark"] == REGRESSION_BENCHMARK_KEY
    assert sample["rows_total"] == REGRESSION_PROBLEM_COUNT
    assert sample["exact_match"] == 1.0


def test_verify_regression_sample_passes_valid_sample():
    sample = build_regression_sample(_all_correct_responses())
    assert verify_regression_sample(sample, claimed_gsm8k=1.0) == []


def test_verify_regression_sample_catches_tampered_score():
    sample = build_regression_sample(_all_correct_responses())
    sample["exact_match"] = 0.5
    issues = verify_regression_sample(sample, claimed_gsm8k=0.5)
    assert any("does not match recomputed" in issue for issue in issues)


def test_verify_regression_sample_catches_claim_divergence():
    sample = build_regression_sample(_all_correct_responses())
    issues = verify_regression_sample(sample, claimed_gsm8k=0.5)
    assert any("claimed gsm8k" in issue for issue in issues)


def test_check_gsm8k_no_regression_within_floor():
    assert check_gsm8k_no_regression(0.595, 0.60) == []
    # 2% relaxed floor when triton up >= 2%
    assert check_gsm8k_no_regression(0.591, 0.60, triton_pct=12.0) == []


def test_check_gsm8k_no_regression_flags_large_drop():
    issues = check_gsm8k_no_regression(0.50, 0.60)
    assert any("gsm8k regression" in issue for issue in issues)


def test_build_regression_sample_rejects_incomplete_ids():
    responses = _all_correct_responses()[:-1]
    with pytest.raises(ValueError, match=f"expected {REGRESSION_PROBLEM_COUNT} responses"):
        build_regression_sample(responses)


def test_verify_regression_sample_rejects_duplicated_problem_ids():
    # A forged sample that duplicates one correct answer to reach exact_match 1.0
    # must be rejected by the validator — build refuses to produce it, but the
    # bundle content is miner-controlled, so verify must guard coverage itself.
    correct = _all_correct_responses()
    forged = [dict(correct[0]) for _ in range(REGRESSION_PROBLEM_COUNT)]
    sample = {
        "version": build_regression_sample(correct)["version"],
        "benchmark": REGRESSION_BENCHMARK_KEY,
        "problem_set_path": build_regression_sample(correct)["problem_set_path"],
        "problem_set_sha256": build_regression_sample(correct)["problem_set_sha256"],
        "rows_total": REGRESSION_PROBLEM_COUNT,
        "exact_match": 1.0,
        "responses": forged,
    }
    issues = verify_regression_sample(sample, claimed_gsm8k=1.0)
    assert any("exactly once" in issue for issue in issues)


def test_verify_regression_sample_rejects_incomplete_ids():
    sample = build_regression_sample(_all_correct_responses())
    sample["responses"] = sample["responses"][:-1]
    issues = verify_regression_sample(sample, claimed_gsm8k=1.0)
    assert any("exactly once" in issue for issue in issues)

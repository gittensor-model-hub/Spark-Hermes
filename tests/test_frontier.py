from eval.frontier import (
    GSM8K_RELAXED_REGRESSION_FLOOR_PCT,
    merge_frontier_scores,
    regression_floor_pct,
    triton_pct_delta,
)
from eval.score import score


def test_merge_frontier_updates_gsm8k_when_improved():
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    candidate = {"triton": 0.42, "gsm8k": 0.65}
    merged, updates = merge_frontier_scores(frontier, candidate)
    assert merged["gsm8k"] == 0.65
    assert merged["triton"] == 0.428
    assert updates == ["gsm8k"]


def test_gsm8k_floor_relaxes_when_triton_improves_at_least_2pct():
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    candidate = {"triton": 0.48, "gsm8k": 0.6}
    assert triton_pct_delta(candidate, frontier) >= 2.0
    assert (
        regression_floor_pct("gsm8k", triton_pct=triton_pct_delta(candidate, frontier))
        == GSM8K_RELAXED_REGRESSION_FLOOR_PCT
    )


def test_score_allows_gsm8k_regression_up_to_2pct_when_triton_improves():
    candidate = {"triton": 0.48, "gsm8k": 0.591}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:L"
    assert report["regressions"] == []
    assert report["gsm8k_regression_floor_pct"] == 2.0


def test_score_rejects_gsm8k_regression_beyond_relaxed_floor():
    candidate = {"triton": 0.48, "gsm8k": 0.58}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-gsm8k" in report["regressions"]


def test_score_keeps_1pct_gsm8k_floor_when_triton_gain_is_small():
    candidate = {"triton": 0.433, "gsm8k": 0.591}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-gsm8k" in report["regressions"]
    assert report["gsm8k_regression_floor_pct"] == 1.0


def test_score_reports_frontier_updates_on_verified_run():
    candidate = {"triton": 0.48, "gsm8k": 0.65}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert set(report["frontier_updates"]) == {"triton", "gsm8k"}
    assert report["frontier_scores"]["gsm8k"] == 0.65
    assert report["frontier_scores"]["triton"] == 0.48

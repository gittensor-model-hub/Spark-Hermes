"""Frontier merge + per-benchmark regression floors for mining-track scoring."""

from __future__ import annotations

from eval.benchmarks import BENCHMARKS

TIER_BENCHMARK = "triton"
GSM8K_BENCHMARK = "gsm8k"

# When triton improves at least this much vs frontier, GSM8K may regress up to 2%.
GSM8K_RELAX_TRITON_IMPROVEMENT_PCT = 2.0
GSM8K_RELAXED_REGRESSION_FLOOR_PCT = 2.0


def pct_delta(candidate: float, frontier: float) -> float:
    if frontier == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (candidate - frontier) / frontier * 100.0


def triton_pct_delta(candidate: dict[str, float], frontier: dict[str, float]) -> float | None:
    if TIER_BENCHMARK not in candidate or TIER_BENCHMARK not in frontier:
        return None
    return pct_delta(float(candidate[TIER_BENCHMARK]), float(frontier[TIER_BENCHMARK]))


def regression_floor_pct(benchmark_key: str, *, triton_pct: float | None) -> float:
    """Per-benchmark regression floor; GSM8K relaxes when triton improves enough."""
    benchmark = BENCHMARKS[benchmark_key]
    if benchmark_key == GSM8K_BENCHMARK and triton_pct is not None:
        if triton_pct >= GSM8K_RELAX_TRITON_IMPROVEMENT_PCT:
            return GSM8K_RELAXED_REGRESSION_FLOOR_PCT
    return benchmark.regression_floor_pct


def is_regression(
    benchmark_key: str,
    candidate_score: float,
    frontier_score: float,
    *,
    triton_pct: float | None,
) -> bool:
    delta = pct_delta(candidate_score, frontier_score)
    if delta >= 0:
        return False
    return abs(delta) > regression_floor_pct(benchmark_key, triton_pct=triton_pct)


def merge_frontier_scores(
    current: dict[str, float],
    candidate: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Raise per-benchmark frontier highs from a verified candidate.

    Any benchmark that beats the current frontier is updated — including GSM8K
    when a miner improves math reasoning even if Triton is flat.
    """
    merged = dict(current)
    updates: list[str] = []
    for key in BENCHMARKS:
        if key not in candidate:
            continue
        value = float(candidate[key])
        if key not in merged or value > float(merged[key]):
            merged[key] = value
            updates.append(key)
    return merged, updates

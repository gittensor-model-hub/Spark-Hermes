"""Frozen GSM8K regression sample for attested no-GPU verification.

Miners run `eval.export_gsm8k_regression_sample` on their checkpoint, include the
output in a proof bundle, and bind it with GPU attestation (`claim_sha256` nonce).
Validators re-grade the bundled responses against the frozen gold answers without
re-running the model when attestation + binding pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dataset_verify import _sha256_file
from eval.frontier import is_regression, regression_floor_pct
from eval.gsm8k_eval import gold_answer, grade_gsm8k_response

REGRESSION_VERSION = "sparkdistill-gsm8k-regression-v3"
REGRESSION_PROBLEM_COUNT = 50
REGRESSION_PROBLEMS_PATH = (
    Path(__file__).resolve().parent / "data" / f"gsm8k_regression_{REGRESSION_PROBLEM_COUNT}.jsonl"
)
REGRESSION_SAMPLE_FILENAME = "gsm8k_regression_sample.json"
REGRESSION_BENCHMARK_KEY = "gsm8k"


def load_regression_problems(path: Path = REGRESSION_PROBLEMS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "problem_id" not in row or "question" not in row or "answer" not in row:
                raise ValueError(f"{path}:{line_no}: each row needs problem_id, question, answer")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: regression problem set is empty")
    return rows


def regression_problem_set_sha256(path: Path = REGRESSION_PROBLEMS_PATH) -> str:
    return _sha256_file(path)


def grade_response(gold: str, model_response: str) -> bool:
    return grade_gsm8k_response(gold, model_response)


def compute_exact_match(responses: list[dict[str, Any]], problems: list[dict[str, Any]]) -> float:
    by_id = {int(row["problem_id"]): row for row in problems}
    correct = 0
    for item in responses:
        problem_id = int(item["problem_id"])
        if problem_id not in by_id:
            raise ValueError(f"unknown problem_id {problem_id}")
        if grade_response(gold_answer(by_id[problem_id]), str(item["model_response"])):
            correct += 1
    return correct / len(problems)


def build_regression_sample(
    responses: list[dict[str, Any]],
    *,
    problems_path: Path = REGRESSION_PROBLEMS_PATH,
) -> dict[str, Any]:
    problems = load_regression_problems(problems_path)
    if len(responses) != len(problems):
        raise ValueError(f"expected {len(problems)} responses, got {len(responses)}")
    seen = {int(item["problem_id"]) for item in responses}
    expected = {int(row["problem_id"]) for row in problems}
    if seen != expected:
        raise ValueError(f"response problem_id set {sorted(seen)} != expected {sorted(expected)}")
    exact_match = compute_exact_match(responses, problems)
    return {
        "version": REGRESSION_VERSION,
        "benchmark": REGRESSION_BENCHMARK_KEY,
        "problem_set_path": problems_path.name,
        "problem_set_sha256": regression_problem_set_sha256(problems_path),
        "rows_total": len(problems),
        "exact_match": exact_match,
        "responses": sorted(responses, key=lambda row: int(row["problem_id"])),
    }


def verify_regression_sample(
    sample: dict[str, Any],
    *,
    claimed_gsm8k: float | None = None,
    problems_path: Path = REGRESSION_PROBLEMS_PATH,
    score_tolerance_pct: float = 0.5,
) -> list[str]:
    """CPU-only verification of a bundled GSM8K regression sample."""
    issues: list[str] = []
    if sample.get("version") != REGRESSION_VERSION:
        issues.append(f"regression sample version must be {REGRESSION_VERSION!r}")
    if sample.get("benchmark") != REGRESSION_BENCHMARK_KEY:
        issues.append(f"regression sample benchmark must be {REGRESSION_BENCHMARK_KEY!r}")

    expected_sha = regression_problem_set_sha256(problems_path)
    if sample.get("problem_set_sha256") != expected_sha:
        issues.append("regression sample problem_set_sha256 does not match frozen problem set")

    problems = load_regression_problems(problems_path)
    responses = sample.get("responses")
    if not isinstance(responses, list):
        return issues + ["regression sample missing responses list"]

    # Coverage guard (mirror build_regression_sample): the responses must answer
    # each frozen problem exactly once. compute_exact_match divides the correct
    # count by len(problems) while iterating over `responses`, so without this a
    # miner could forge an arbitrary score by duplicating or cherry-picking
    # problem_ids — e.g. 50 copies of one correct answer recomputes to exact_match
    # 1.0 — and slip a regressed gsm8k past the attested no-GPU floor.
    expected_ids = {int(row["problem_id"]) for row in problems}
    try:
        response_ids = [int(item["problem_id"]) for item in responses]
    except (KeyError, TypeError, ValueError):
        return issues + ["regression sample responses must each carry an integer problem_id"]
    if len(response_ids) != len(problems) or set(response_ids) != expected_ids:
        return issues + [
            f"regression sample must answer each of the {len(problems)} frozen problems "
            "exactly once (no missing, extra, or duplicate problem_ids)"
        ]

    try:
        recomputed = compute_exact_match(responses, problems)
    except ValueError as exc:
        return issues + [str(exc)]

    reported = sample.get("exact_match")
    if not isinstance(reported, (int, float)):
        issues.append("regression sample missing exact_match score")
    elif abs(float(reported) - recomputed) > 1e-9:
        issues.append(f"regression sample exact_match {reported!r} does not match recomputed {recomputed!r}")

    if claimed_gsm8k is not None and abs(float(claimed_gsm8k) - recomputed) * 100.0 > score_tolerance_pct:
        issues.append(f"claimed gsm8k {claimed_gsm8k!r} diverges from attested regression sample {recomputed!r}")
    return issues


def check_gsm8k_no_regression(
    sample_score: float,
    frontier_gsm8k: float,
    *,
    triton_pct: float | None = None,
    floor_pct: float | None = None,
) -> list[str]:
    if floor_pct is None:
        floor_pct = regression_floor_pct(REGRESSION_BENCHMARK_KEY, triton_pct=triton_pct)
    if is_regression(REGRESSION_BENCHMARK_KEY, sample_score, frontier_gsm8k, triton_pct=triton_pct):
        pct_delta = (sample_score - frontier_gsm8k) / frontier_gsm8k * 100.0 if frontier_gsm8k else 0.0
        return [f"gsm8k regression: {pct_delta:.2f}% vs frontier exceeds -{floor_pct}% floor"]
    return []


def read_regression_sample(bundle_dir: Path) -> dict[str, Any] | None:
    sample_path = bundle_dir / REGRESSION_SAMPLE_FILENAME
    if not sample_path.exists():
        return None
    return json.loads(sample_path.read_text(encoding="utf-8"))

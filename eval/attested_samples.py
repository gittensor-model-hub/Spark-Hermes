"""Attested eval samples for no-GPU validator verification.

Miners export benchmark artifacts on a GPU CC + Intel TDX guest, bind them into the
proof bundle (`claim_sha256`), and attach GPU + TDX attestation. Validators re-check
scores from the bundled artifacts on CPU — no checkpoint reproduction or harness re-run.

GSM8K uses a frozen 50-problem set with lm-eval-aligned 5-shot prompts and
`exact_match,strict-match` grading. Other lm-eval
benchmarks bundle the harness results JSON; Triton bundles the TritonBench report so
composites can be recomputed from per-problem details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from eval.benchmarks import BENCHMARKS, _extract_metric, _locate_results_file
from eval.frontier import is_regression, regression_floor_pct, triton_pct_delta
from eval.regression_sample import (
    REGRESSION_BENCHMARK_KEY,
    REGRESSION_SAMPLE_FILENAME,
    build_regression_sample,
    read_regression_sample,
    verify_regression_sample,
)
from eval.triton_bench import summary_scores

ATTESTED_VERIFY_LIMIT = 50
ATTESTED_SAMPLES_VERSION = "sparkdistill-attested-eval-v2"
ATTESTED_SAMPLES_FILENAME = "attested_eval_samples.json"

# Legacy gsm8k-only bundles (superseded by attested_eval_samples.json).
LEGACY_GSM8K_ONLY = "legacy_gsm8k_regression"


def attested_claim_files() -> tuple[str, ...]:
    """Bundle files hashed into claim_sha256 when attested samples are present."""
    return ("eval_scores.json", "manifest.json", ATTESTED_SAMPLES_FILENAME, REGRESSION_SAMPLE_FILENAME)


def read_attested_samples(bundle_dir: Path) -> dict[str, Any] | None:
    """Load unified attested samples, or wrap a legacy gsm8k-only file."""
    unified = bundle_dir / ATTESTED_SAMPLES_FILENAME
    if unified.exists():
        return json.loads(unified.read_text(encoding="utf-8"))

    legacy = read_regression_sample(bundle_dir)
    if legacy is None:
        return None
    return {
        "version": LEGACY_GSM8K_ONLY,
        "benchmarks": {REGRESSION_BENCHMARK_KEY: {"type": "regression_responses", "sample": legacy}},
    }


def has_attested_samples(bundle_dir: Path) -> bool:
    return (bundle_dir / ATTESTED_SAMPLES_FILENAME).exists() or (bundle_dir / REGRESSION_SAMPLE_FILENAME).exists()


def check_attestation_bindings(
    bundle_dir: Path,
    attestation: dict | None,
    *,
    claim_binding: Callable[[Path, dict | None], bool | None],
    tdx_binding: Callable[[Path, dict | None], bool | None],
) -> list[str]:
    """GPU CC nonce + Intel TDX REPORTDATA must both commit to this bundle."""
    issues: list[str] = []
    if attestation is None or not attestation.get("passed"):
        issues.append("attested no-GPU verification requires a passed GPU CC attestation")
        return issues
    if claim_binding(bundle_dir, attestation) is not True:
        issues.append("attested no-GPU verification requires claim_sha256-bound GPU attestation")
    if tdx_binding(bundle_dir, attestation) is not True:
        issues.append("attested no-GPU verification requires TDX quote bound to claim_sha256")
    return issues


def check_benchmark_no_regression(
    benchmark_key: str,
    sample_score: float,
    frontier: dict[str, float] | None,
    *,
    claimed: dict[str, float] | None = None,
) -> list[str]:
    if frontier is None or benchmark_key not in frontier:
        return []
    if benchmark_key not in BENCHMARKS:
        return []
    frontier_score = float(frontier[benchmark_key])
    triton_pct = triton_pct_delta(claimed or {}, frontier) if claimed is not None else None
    if is_regression(benchmark_key, sample_score, frontier_score, triton_pct=triton_pct):
        floor = regression_floor_pct(benchmark_key, triton_pct=triton_pct)
        pct_delta = (sample_score - frontier_score) / frontier_score * 100.0 if frontier_score else 0.0
        return [f"{benchmark_key} regression: {pct_delta:.2f}% vs frontier exceeds -{floor}% floor"]
    return []


def _score_tolerance_pct(benchmark_key: str, default: float = 0.5) -> float:
    benchmark = BENCHMARKS.get(benchmark_key)
    if benchmark is None or benchmark.claim_tolerance_pct is None:
        return default
    return float(benchmark.claim_tolerance_pct)


def _claimed_score(claimed: dict[str, float], benchmark_key: str) -> float | None:
    if benchmark_key not in claimed:
        return None
    if benchmark_key == "triton" and "triton_quick" in claimed:
        return float(claimed["triton_quick"])
    return float(claimed[benchmark_key])


def verify_regression_responses(
    entry: dict[str, Any],
    *,
    claimed: dict[str, float],
    frontier: dict[str, float] | None,
) -> tuple[float | None, list[str]]:
    sample = entry.get("sample") or entry
    benchmark_key = str(sample.get("benchmark") or REGRESSION_BENCHMARK_KEY)
    claimed_value = _claimed_score(claimed, benchmark_key)
    issues = verify_regression_sample(
        sample,
        claimed_gsm8k=claimed_value,
        score_tolerance_pct=_score_tolerance_pct(benchmark_key),
    )
    recomputed = float(sample.get("exact_match", 0.0))
    if not issues:
        issues.extend(check_benchmark_no_regression(benchmark_key, recomputed, frontier, claimed=claimed))
    return recomputed if not issues else None, issues


def verify_lm_eval_results(
    entry: dict[str, Any],
    *,
    benchmark_key: str,
    claimed: dict[str, float],
    frontier: dict[str, float] | None,
) -> tuple[float | None, list[str]]:
    issues: list[str] = []
    benchmark = BENCHMARKS.get(benchmark_key)
    if benchmark is None:
        return None, [f"unknown benchmark {benchmark_key!r}"]

    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return None, [f"{benchmark_key}: attested sample missing lm-eval payload"]

    task_results = (payload.get("results") or {}).get(benchmark.lm_eval_task)
    if not isinstance(task_results, dict):
        return None, [f"{benchmark_key}: lm-eval payload missing task results for {benchmark.lm_eval_task!r}"]

    try:
        recomputed = _extract_metric(task_results, benchmark.metric)
    except KeyError as exc:
        return None, [f"{benchmark_key}: {exc}"]

    reported = entry.get("score")
    if reported is not None and abs(float(reported) - recomputed) > 1e-9:
        issues.append(f"{benchmark_key}: bundled score {reported!r} != recomputed {recomputed!r}")

    claimed_value = _claimed_score(claimed, benchmark_key)
    if claimed_value is not None:
        tolerance = _score_tolerance_pct(benchmark_key)
        if abs(claimed_value - recomputed) * 100.0 > tolerance:
            issues.append(f"claimed {benchmark_key} {claimed_value!r} diverges from attested sample {recomputed!r}")

    if not issues:
        issues.extend(check_benchmark_no_regression(benchmark_key, recomputed, frontier, claimed=claimed))
    return recomputed if not issues else None, issues


def verify_tritonbench_report(
    entry: dict[str, Any],
    *,
    claimed: dict[str, float],
    frontier: dict[str, float] | None,
) -> tuple[float | None, list[str]]:
    issues: list[str] = []
    report = entry.get("report")
    if not isinstance(report, dict):
        return None, ["triton: attested sample missing TritonBench report"]

    # The reward tier reads the FULL triton composite (claimed["triton"]). Recompute it
    # from the executed per-problem details — the mean of every problem's composite_score
    # (tritonbench reporter._summarize) — so it can't be forged through the miner-written
    # report.summary.avg_composite while honest quick-subset details still verify (#241).
    # A report without details cannot be independently recomputed, so it is rejected.
    details = report.get("details")
    if not isinstance(details, list) or not details:
        return None, ["triton: TritonBench report has no per-problem details to recompute the composite from"]
    composites: list[float] = []
    for problem in details:
        composite = problem.get("composite_score") if isinstance(problem, dict) else None
        if not isinstance(composite, (int, float)) or isinstance(composite, bool):
            return None, ["triton: TritonBench report details are missing a numeric composite_score"]
        composites.append(float(composite))
    full_composite = sum(composites) / len(composites)

    recomputed_scores = summary_scores(report)
    tolerance = _score_tolerance_pct("triton")

    # The report's own summary must be the true mean of its details, not an inflated headline.
    summary_composite = float((report.get("summary") or {}).get("avg_composite", 0.0))
    if abs(summary_composite - full_composite) > 1e-9:
        issues.append(f"triton: report summary avg_composite {summary_composite!r} != recomputed {full_composite!r}")

    reported_scores = entry.get("scores") or {}
    for key in ("triton", "triton_quick"):
        if key in reported_scores and abs(float(reported_scores[key]) - recomputed_scores[key]) > 1e-9:
            issues.append(f"triton: bundled {key} {reported_scores[key]!r} != recomputed {recomputed_scores[key]!r}")

    # Bind the claimed full composite (what the tier reads) to the recomputed one — not just
    # the quick subset that _claimed_score would otherwise check — and the claimed quick
    # subset to its recomputed value.
    claimed_full = claimed.get("triton")
    if claimed_full is not None and abs(float(claimed_full) - full_composite) * 100.0 > tolerance:
        issues.append(f"claimed triton {float(claimed_full)!r} diverges from recomputed composite {full_composite!r}")
    claimed_quick = claimed.get("triton_quick")
    if claimed_quick is not None and abs(float(claimed_quick) - recomputed_scores["triton_quick"]) * 100.0 > tolerance:
        issues.append(
            f"claimed triton_quick {float(claimed_quick)!r} diverges from attested sample "
            f"{recomputed_scores['triton_quick']!r}"
        )

    if not issues:
        issues.extend(check_benchmark_no_regression("triton", full_composite, frontier))
    return full_composite if not issues else None, issues


def verify_benchmark_entry(
    benchmark_key: str,
    entry: dict[str, Any],
    *,
    claimed: dict[str, float],
    frontier: dict[str, float] | None,
) -> tuple[float | None, list[str]]:
    sample_type = entry.get("type")
    if sample_type == "regression_responses":
        return verify_regression_responses(entry, claimed=claimed, frontier=frontier)
    if sample_type == "lm_eval_results":
        return verify_lm_eval_results(entry, benchmark_key=benchmark_key, claimed=claimed, frontier=frontier)
    if sample_type == "tritonbench_report":
        return verify_tritonbench_report(entry, claimed=claimed, frontier=frontier)
    return None, [f"{benchmark_key}: unknown attested sample type {sample_type!r}"]


def verify_attested_eval_samples(
    bundle_dir: Path,
    claimed: dict[str, float],
    frontier: dict[str, float] | None,
    attestation: dict | None,
    *,
    claim_binding: Callable[[Path, dict | None], bool | None],
    tdx_binding: Callable[[Path, dict | None], bool | None],
) -> tuple[set[str], list[str]]:
    """Return benchmark keys verified on CPU and any blocking issues."""
    samples = read_attested_samples(bundle_dir)
    if samples is None:
        return set(), []

    binding_issues = check_attestation_bindings(
        bundle_dir, attestation, claim_binding=claim_binding, tdx_binding=tdx_binding
    )
    if binding_issues:
        return set(), binding_issues

    version = samples.get("version")
    if version not in (ATTESTED_SAMPLES_VERSION, LEGACY_GSM8K_ONLY):
        return set(), [f"attested eval samples version must be {ATTESTED_SAMPLES_VERSION!r}"]

    benchmarks = samples.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return set(), ["attested eval samples missing benchmarks map"]

    verified: set[str] = set()
    issues: list[str] = []
    for benchmark_key, entry in benchmarks.items():
        if benchmark_key not in claimed or benchmark_key not in BENCHMARKS:
            continue
        if not isinstance(entry, dict):
            issues.append(f"{benchmark_key}: attested sample entry must be an object")
            continue
        _, entry_issues = verify_benchmark_entry(benchmark_key, entry, claimed=claimed, frontier=frontier)
        if entry_issues:
            issues.extend(entry_issues)
        else:
            verified.add(benchmark_key)

    return verified, issues


def build_attested_samples_document(benchmarks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"version": ATTESTED_SAMPLES_VERSION, "benchmarks": benchmarks}


def build_gsm8k_regression_entry(responses: list[dict[str, Any]]) -> dict[str, Any]:
    sample = build_regression_sample(responses)
    return {"type": "regression_responses", "sample": sample}


def build_lm_eval_entry(benchmark_key: str, payload: dict[str, Any], score: float) -> dict[str, Any]:
    benchmark = BENCHMARKS[benchmark_key]
    return {
        "type": "lm_eval_results",
        "task": benchmark.lm_eval_task,
        "metric": benchmark.metric,
        "score": score,
        "payload": payload,
    }


def build_triton_entry(report: dict[str, Any]) -> dict[str, Any]:
    scores = summary_scores(report)
    return {"type": "tritonbench_report", "report": report, "scores": scores}


def write_attested_samples(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_lm_eval_payload(work_dir: Path, benchmark_key: str) -> dict[str, Any]:
    result_path = work_dir / f"{benchmark_key}.json"
    return json.loads(_locate_results_file(result_path).read_text(encoding="utf-8"))

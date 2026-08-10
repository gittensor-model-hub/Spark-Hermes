"""Benchmark basket + adapters.

Mirrors the benchmark basket `sparkinfer`'s accuracy gate already tracks
(BFCL, GSM8K, HumanEval, IFEval, MMLU-Pro) so a distilled checkpoint's quality
claims are comparable across both repos, plus two hard-reasoning benchmarks
(AIME, GPQA-Diamond) since SparkDistill's goal is reasoning distillation and
the easier basket alone can't distinguish "learned to reason" from "learned
to answer". Each entry maps to an `lm-evaluation-harness`
(https://github.com/EleutherAI/lm-evaluation-harness) task name — the harness
itself is an external dependency, installed separately (like Axolotl), not
vendored here.

The `triton` entry is the exception: it is the domain-expertise signal for the
Triton track and runs through the vendored TritonBench harness (`eval.triton_bench`)
instead of lm-eval — each generated kernel is compiled and executed on the GPU. The
general basket can't measure kernel skill; it keeps its regression-guard role while
`triton` carries the improvement signal for Triton-focused recipes.

Score-unit convention: every benchmark score in this pipeline is a fraction in [0, 1]
(an accuracy / pass-rate / composite), never a 0-100 percentage. `assert_fraction_scores`
enforces this at ingestion — see its docstring for why the distinction is load-bearing."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    key: str
    # User-facing slug for the `regression-<slug>` PR label. Kept separate from `key`
    # because `key` mirrors the lm-eval task name (underscores, filter suffixes like
    # `gpqa_diamond_cot_zeroshot`), while labels use the hyphenated taxonomy documented
    # in docs/miner-guide.md. Coupling the two leaks lm-eval internals into PR labels.
    label_slug: str
    lm_eval_task: str
    metric: str
    regression_floor_pct: float  # max allowed drop vs. frontier before a regression-* label fires
    # Per-benchmark override for eval.verify's claim re-run tolerance (percentage
    # points); None uses the verifier's global default. For where honest re-runs
    # drift more than that default — e.g. triton's composite over only 3 problems
    # moves ~7pp when one generation diverges across vLLM server instances.
    claim_tolerance_pct: float | None = None


BENCHMARKS: dict[str, Benchmark] = {
    "bfcl": Benchmark(key="bfcl", label_slug="bfcl", lm_eval_task="bfcl", metric="acc", regression_floor_pct=1.0),
    "gsm8k": Benchmark(
        key="gsm8k", label_slug="gsm8k", lm_eval_task="gsm8k", metric="exact_match", regression_floor_pct=1.0
    ),
    "humaneval": Benchmark(
        key="humaneval", label_slug="humaneval", lm_eval_task="humaneval", metric="pass@1", regression_floor_pct=1.0
    ),
    "ifeval": Benchmark(
        key="ifeval",
        label_slug="ifeval",
        lm_eval_task="ifeval",
        metric="inst_level_strict_acc",
        regression_floor_pct=1.0,
    ),
    "mmlu_pro": Benchmark(
        key="mmlu_pro", label_slug="mmlu-pro", lm_eval_task="mmlu_pro", metric="acc", regression_floor_pct=1.0
    ),
    "aime24": Benchmark(
        key="aime24", label_slug="aime24", lm_eval_task="aime24", metric="exact_match", regression_floor_pct=2.0
    ),
    # lm-eval applies strict/flexible-extract filters on top of this task's base
    # "exact_match" metric; depending on the installed lm-eval version the results.json
    # key may come back suffixed as "exact_match,flexible-extract" instead of bare
    # "exact_match" — verify against your installed harness version before trusting this
    # key blindly, and adjust here if `run_benchmark` KeyErrors on it.
    "gpqa_diamond_cot_zeroshot": Benchmark(
        key="gpqa_diamond_cot_zeroshot",
        label_slug="gpqa-diamond",
        lm_eval_task="gpqa_diamond_cot_zeroshot",
        metric="exact_match",
        regression_floor_pct=2.0,
    ),
    # TritonBench composite (compile + execute + correctness + API modernity), run via
    # eval.triton_bench, not lm-eval. Its problems are quarantined from training data by
    # SparkProof's release gate, which is what keeps this a legitimate held-out eval.
    # claim_tolerance_pct: observed honest cross-server drift of 2.1pp on the current
    # 3-problem quick set — tighten back toward the 2pp default as problems grow.
    "triton": Benchmark(
        key="triton",
        label_slug="triton",
        lm_eval_task="",
        metric="avg_composite",
        regression_floor_pct=2.0,
        claim_tolerance_pct=5.0,
    ),
}


def assert_fraction_scores(scores: dict[str, float], source: str) -> None:
    """Enforce the pipeline-wide score-unit convention: every metric is a **fraction in
    `[0, 1]`**, never a `0`–`100` percentage.

    This is the invariant `runs/frontier.json` and the lm-eval / TritonBench adapters
    already follow, but it was implicit. `eval.verify.check_claim` converts a claim/re-run
    gap to percentage points with a hardcoded `* 100.0`, so a percentage-unit score would
    silently make its tolerance 100x too tight and reject honest submissions. Fail loudly
    at ingestion instead of scoring the wrong unit. See docs/miner-guide.md (proof-bundle
    score format).

    `scores` is miner-controlled (`eval_scores.json["scores"]` from the proof bundle),
    so every value is validated to be a finite, non-boolean real number *before* the
    range check: `float(value)` on a `null`/list/object value raised `TypeError`, and a
    non-numeric string raised a bare `float()` `ValueError` — either one escaped this
    function's fail-closed contract and crashed `eval.verify` (and the CI training-track
    gate, which does not wrap `verify_submission`) instead of rejecting the bundle. A
    JSON `true`/`false` also slipped through as `1.0`/`0.0`."""
    if not isinstance(scores, dict):
        raise ValueError(
            f"{source} scores must be a JSON object mapping benchmark -> fraction, got {type(scores).__name__}"
        )
    # bool is a subclass of int, so it must be rejected before the numeric check or a
    # JSON `true` would score 1.0. math.isfinite guards NaN/Infinity, which json.loads
    # accepts by default and which would otherwise sneak past a naive range comparison.
    malformed = {
        key: value
        for key, value in scores.items()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
    }
    if malformed:
        raise ValueError(
            f"{source} scores must be finite numeric fractions in [0, 1], got malformed {malformed}; "
            "eval scores are numeric accuracies / pass-rates, not strings, booleans, or non-finite values"
        )
    out_of_range = {key: value for key, value in scores.items() if not 0.0 <= float(value) <= 1.0}
    if out_of_range:
        raise ValueError(
            f"{source} scores must be fractions in [0, 1], got {out_of_range}; "
            "eval scores are accuracies / pass-rates, not 0-100 percentages"
        )


def run_benchmark(benchmark: Benchmark, model_path: str, output_dir: Path, limit: int | None = None) -> float:
    """Run a single benchmark via `lm-eval` and return the headline metric.

    `limit` caps the number of examples `lm-eval` samples (its own `--limit` flag) —
    use a small limit for cheap re-verification of a submitted claim, leave unset for
    a full basket run.

    Requires the `lm-eval` CLI on PATH (`pip install lm-eval`). Not invoked
    during import so the harness stays importable (and its `--help` usable)
    without the harness or a GPU present.
    """
    if benchmark.key == "triton":
        from eval.triton_bench import run_triton_benchmark

        return run_triton_benchmark(model_path, output_dir, limit=limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{benchmark.key}.json"
    command = [
        "lm-eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        benchmark.lm_eval_task,
        "--output_path",
        str(result_path),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    subprocess.run(command, check=True)
    payload = json.loads(_locate_results_file(result_path).read_text())
    return _extract_metric(payload["results"][benchmark.lm_eval_task], benchmark.metric)


def _locate_results_file(result_path: Path) -> Path:
    """Find the results JSON lm-eval actually wrote for `--output_path result_path`.

    lm-eval 0.4.x appends a date id to a `.json` output path (`gsm8k.json` becomes
    `gsm8k_<date>.json`); older versions wrote the exact path. Prefer the exact
    path, else the newest date-suffixed sibling.
    """
    if result_path.exists():
        return result_path
    candidates = list(result_path.parent.glob(f"{result_path.stem}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"lm-eval wrote no results for {result_path}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _extract_metric(task_results: dict, metric: str) -> float:
    """Read `metric` from a task's results, tolerating lm-eval's filter suffixes.

    0.4.x reports metrics as `<metric>,<filter>` (e.g. `exact_match,strict-match`);
    the strict/none variants are preferred over flexible extraction so scores stay
    comparable across submissions.
    """
    if metric in task_results:
        return float(task_results[metric])
    for preferred in (f"{metric},strict-match", f"{metric},none"):
        if preferred in task_results:
            return float(task_results[preferred])
    suffixed = sorted(key for key in task_results if key.split(",")[0] == metric)
    if suffixed:
        return float(task_results[suffixed[0]])
    raise KeyError(f"metric {metric!r} not in lm-eval results (have: {sorted(task_results)})")

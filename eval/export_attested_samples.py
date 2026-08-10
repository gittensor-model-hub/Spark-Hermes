"""Export attested eval samples from a checkpoint on GPU CC + TDX hardware.

Runs the cheap verification basket (`--limit 50`) and records per-benchmark artifacts
for CPU-only validator verification when GPU + TDX attestation binds the bundle.

    uv run python -m eval.export_attested_samples \\
        --checkpoint outputs/<run> \\
        --out eval/results/attested_eval_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.attested_samples import (
    ATTESTED_SAMPLES_VERSION,
    ATTESTED_VERIFY_LIMIT,
    build_attested_samples_document,
    build_gsm8k_regression_entry,
    build_lm_eval_entry,
    build_triton_entry,
    load_lm_eval_payload,
    write_attested_samples,
)
from eval.benchmarks import BENCHMARKS, _extract_metric, run_benchmark
from eval.export_gsm8k_regression_sample import export_gsm8k_regression_sample
from eval.harness import run_harness
from eval.regression_sample import REGRESSION_BENCHMARK_KEY, REGRESSION_PROBLEMS_PATH


def _gsm8k_responses_from_sample(sample: dict) -> list[dict]:
    return list(sample.get("responses") or [])


def export_attested_samples(
    checkpoint: Path,
    *,
    out_path: Path,
    work_dir: Path,
    limit: int = ATTESTED_VERIFY_LIMIT,
    benchmarks: list[str] | None = None,
) -> dict:
    selected = benchmarks or sorted(BENCHMARKS)
    unknown = [key for key in selected if key not in BENCHMARKS]
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")

    work_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}

    non_triton = [key for key in selected if key != "triton"]
    non_gsm8k = [key for key in non_triton if key != REGRESSION_BENCHMARK_KEY]
    if non_gsm8k:
        run_harness(str(checkpoint), non_gsm8k, work_dir, limit=limit)

    if REGRESSION_BENCHMARK_KEY in selected:
        gsm8k_sample = export_gsm8k_regression_sample(
            checkpoint, out_path=work_dir / "_gsm8k_sample.json", problems_path=REGRESSION_PROBLEMS_PATH
        )
        entries[REGRESSION_BENCHMARK_KEY] = build_gsm8k_regression_entry(_gsm8k_responses_from_sample(gsm8k_sample))

    for key in non_gsm8k:
        benchmark = BENCHMARKS[key]
        payload = load_lm_eval_payload(work_dir, key)
        task_results = payload["results"][benchmark.lm_eval_task]
        score = _extract_metric(task_results, benchmark.metric)
        entries[key] = build_lm_eval_entry(key, payload, score)

    if "triton" in selected:
        benchmark = BENCHMARKS["triton"]
        score = run_benchmark(benchmark, str(checkpoint), work_dir, limit=limit)
        detail = json.loads((work_dir / "triton.json").read_text(encoding="utf-8"))
        entries["triton"] = build_triton_entry(detail["report"])
        # Sanity: headline from bundled report matches harness return.
        if abs(float(detail["scores"]["triton"]) - score) > 1e-6:
            raise ValueError("triton harness score diverges from bundled report headline")

    document = build_attested_samples_document(entries)
    write_attested_samples(out_path, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval/results/attested_eval_samples.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("eval/results/_attested_export"))
    parser.add_argument(
        "--limit",
        type=int,
        default=ATTESTED_VERIFY_LIMIT,
        help="examples per lm-eval benchmark (cheap verify set)",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        choices=sorted(BENCHMARKS),
        default=None,
        help="benchmark to export (repeatable; default: full basket)",
    )
    args = parser.parse_args(argv)

    try:
        document = export_attested_samples(
            args.checkpoint,
            out_path=args.out,
            work_dir=args.work_dir,
            limit=args.limit,
            benchmarks=args.benchmarks,
        )
    except Exception as exc:
        print(f"export attested eval samples failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "version": ATTESTED_SAMPLES_VERSION,
        "benchmarks": sorted(document.get("benchmarks", {})),
        "out": str(args.out.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

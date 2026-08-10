"""Triton domain benchmark: score a student checkpoint on TritonBench.

The general basket (BFCL, GSM8K, ...) can't measure Triton expertise — it only guards
against catastrophic forgetting. This adapter is the improvement signal for the Triton
domain: it serves the student checkpoint behind an OpenAI-compatible endpoint (vLLM),
runs the vendored TritonBench harness against it (each generated kernel is compiled and
executed on the GPU, outputs checked against a reference), and reports the summary's
`avg_composite` as the headline metric for `eval.score` tiering.

Eval hygiene: TritonBench problems are quarantined from training data — SparkProof's
release gate blocks any generated row whose AST structure, prompt, or task fingerprint
matches them (see SparkProof's `decontaminate.py`). This benchmark must only ever run
TritonBench's own problems, never prompts derived from miner datasets.

Determinism caveats: pin the same GPU/driver as the frontier run (speed-derived scores
are hardware-sensitive), and keep the config's temperature/max_tokens unchanged between
candidate and frontier runs.

    # existing endpoint (e.g. vLLM already serving the checkpoint):
    python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 \\
        --endpoint http://localhost:8000/v1 --out eval/results/triton.json

    # or let it serve the checkpoint itself (requires `vllm` on PATH):
    python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 --serve \\
        --out eval/results/triton.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from eval.gpu_architecture import GpuArchitecture, normalize_gpu_architecture
from eval.serve_stack import python_headers_available, serve_path_env, vllm_serve_argv

HEADLINE_METRIC = "avg_composite"
_QUICK_LEVELS = [1]  # cheap re-verification, mirrors configs/eval_quick.yaml
_FULL_LEVELS = [1, 2, 3, 4]
# Config must match the run depth: eval_quick.yaml caps max_tokens/timeout below
# default.yaml, and a full run scored under quick's limits is not comparable to a
# miner's full run under default.yaml.
_QUICK_CONFIG = "configs/eval_quick.yaml"
_FULL_CONFIG = "configs/default.yaml"
# The problem subset a quick run covers (level 1 + bugfix, per eval_quick.yaml with
# include_bugfix). `triton_quick` is the composite over this subset, computable from
# both quick and full reports, so cheap re-verification compares like for like.
_QUICK_PROBLEM_LEVELS = frozenset({1, "1", "bugfix"})


def tritonbench_root() -> Path:
    """Where the vendored TritonBench tree lives, or why it does not.

    It is no longer in this repository. 1,118 files and 144 MB of upstream code, with its
    own LICENSE and pyproject, vendored for a domain benchmark that is not the product any
    more -- it now lives on the `archive/tritonbench` branch. Nothing here imports it; this
    adapter shells out to its CLI, so a checkout anywhere works.

    Raising with the location beats returning a path that does not exist. The caller would
    otherwise get `FileNotFoundError: tritonbench/configs/default.yaml` several frames
    later, which says nothing about where the tree went or how to get it back.
    """
    configured = os.environ.get("SPARKDISTILL_TRITONBENCH_ROOT")
    if configured:
        return Path(configured).expanduser()
    vendored = Path(__file__).resolve().parents[1] / "tritonbench"
    if vendored.is_dir():
        return vendored
    raise FileNotFoundError(
        "the vendored TritonBench tree is not in this repository; it moved to the "
        "`archive/tritonbench` branch. Check it out beside this one and point "
        "SPARKDISTILL_TRITONBENCH_ROOT at it:\n"
        "    git worktree add ../tritonbench archive/tritonbench\n"
        "    export SPARKDISTILL_TRITONBENCH_ROOT=../tritonbench/tritonbench"
    )


def detect_gpu_architecture() -> GpuArchitecture | None:
    """Best-effort architecture of the GPU TritonBench is about to run on.

    Recorded alongside the composite score so a claim/re-run comparison never
    silently mixes Blackwell and Hopper numbers — TritonBench's composite is
    speed-derived and hardware-sensitive, so `eval.score`/`eval.verify` must
    tier a run against the frontier bucket for the hardware it actually ran on.
    An explicit `SPARKDISTILL_GPU_ARCHITECTURE` override wins over detection
    (e.g. CI runners without nvidia-smi, or exercising the other bucket).
    """
    override = os.environ.get("SPARKDISTILL_GPU_ARCHITECTURE")
    if override:
        return normalize_gpu_architecture(override)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return normalize_gpu_architecture(name)


def _quick_subset_composite(report: dict) -> float | None:
    """Average composite over the quick-run problem subset (level 1 + bugfix).

    Returns None when the report carries no per-problem details for the subset
    (e.g. a bare summary), in which case callers fall back to the headline.
    """
    details = report.get("details") or []
    quick = [r for r in details if r.get("level") in _QUICK_PROBLEM_LEVELS]
    if not quick:
        return None
    return sum(float(r.get("composite_score", 0.0)) for r in quick) / len(quick)


def _covered_levels(report: dict) -> set[int] | None:
    """Numeric problem levels a report actually scored, or None if indeterminable.

    Reads the reporter's ``by_level`` map first (keyed by ``str(level)``), then the
    per-problem ``details``. A bare summary carries neither, so coverage can't be
    inferred and callers must not treat that as "everything missing".
    """
    by_level = report.get("by_level")
    if isinstance(by_level, dict) and by_level:
        return {int(k) for k in by_level if str(k).isdigit()}
    details = report.get("details")
    if isinstance(details, list) and details:
        return {int(r["level"]) for r in details if str(r.get("level")).isdigit()}
    return None


def level_coverage(report: dict, requested_levels: list[int]) -> dict[str, object]:
    """Compare the levels a run *requested* against the ones it actually scored.

    The runner silently skips level directories that don't exist, so a "full" run
    over ``[1, 2, 3, 4]`` can quietly collapse to whatever levels are populated
    (today: level 1 only). This surfaces that: ``missing`` lists requested levels
    that contributed zero problems, so a partial-coverage run is never mistaken for
    a full one in reports, frontier entries, or re-verification.
    """
    requested = [int(level) for level in requested_levels]
    covered = _covered_levels(report)
    missing = [] if covered is None else [level for level in requested if level not in covered]
    return {
        "requested": requested,
        "covered": None if covered is None else sorted(covered),
        "missing": missing,
        "problems_total": int(report.get("num_problems") or 0),
    }


def _strict_levels() -> bool:
    return os.environ.get("SPARKDISTILL_TRITONBENCH_STRICT_LEVELS", "").strip().lower() in {"1", "true", "yes", "on"}


def _enforce_level_coverage(coverage: dict[str, object]) -> None:
    """Warn (or, under strict mode, raise) when requested levels scored no problems."""
    if not coverage["missing"]:
        return
    message = (
        f"TritonBench requested levels {coverage['requested']} but only scored "
        f"{coverage['covered']} ({coverage['problems_total']} problems total); levels "
        f"{coverage['missing']} contributed no problems and were silently skipped. "
        "This run's scores are not comparable to a full-coverage run. Populate the "
        "missing problem levels, or set SPARKDISTILL_TRITONBENCH_STRICT_LEVELS=1 to fail."
    )
    if _strict_levels():
        raise RuntimeError(message)
    print(f"warning: {message}", file=sys.stderr)


def summary_scores(report: dict) -> dict[str, float]:
    """Flatten a TritonBench report's summary into the score fields we track.

    `triton` (the headline, used by eval.score) is the average composite score;
    the rest are kept alongside it in the detail json for human review.
    `triton_quick` is the composite over the quick-run subset — `eval.verify`
    compares its level-1-only re-run against this, not the full-run headline.
    """
    summary = report.get("summary") or {}
    headline = float(summary.get(HEADLINE_METRIC, 0.0))
    quick = _quick_subset_composite(report)
    return {
        "triton": headline,
        "triton_quick": headline if quick is None else quick,
        "triton_exec_pass_rate": float(summary.get("exec_pass_rate", 0.0)),
        "triton_correctness": float(summary.get("avg_correctness", 0.0)),
        "triton_syntax_pass_rate": float(summary.get("syntax_pass_rate", 0.0)),
    }


def latest_report(results_dir: Path, newer_than_ns: int | None = None) -> dict:
    """Load the newest report by mtime, optionally only those written after a stamp.

    Report filenames embed the model name before the timestamp, so lexicographic
    order sorts by model first — with the results dir reused across runs, that can
    resurrect a stale report for a different checkpoint. mtime (plus the
    `newer_than_ns` stamp taken before the run) pins the report to the run that
    just happened.
    """
    reports = list(results_dir.glob("tritonbench_*.json"))
    if newer_than_ns is not None:
        reports = [p for p in reports if p.stat().st_mtime_ns > newer_than_ns]
    if not reports:
        raise FileNotFoundError(f"TritonBench produced no results JSON under {results_dir}")
    newest = max(reports, key=lambda p: p.stat().st_mtime_ns)
    return json.loads(newest.read_text())


def _endpoint_ready(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


@contextmanager
def serve_checkpoint(
    model_path: str, port: int = 8000, startup_timeout_s: int = 600, served_model_name: str | None = None
):
    """Serve `model_path` via `vllm serve` and yield the OpenAI-compatible endpoint.

    Requires the `vllm` CLI on PATH (installed separately, like lm-eval and Axolotl).
    `served_model_name` must match the name the harness sends in its requests —
    without it vLLM registers the model under the full `model_path` string and
    rejects requests for the basename with a 404.
    """
    endpoint = f"http://127.0.0.1:{port}/v1"
    # --seed and no prefix caching: shrink cross-instance generation drift so a
    # miner's claim and the validator's re-run see the same numerics.
    command = vllm_serve_argv(model_path, port=port, served_model_name=served_model_name)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=serve_path_env(),
    )
    try:
        deadline = time.monotonic() + startup_timeout_s
        while not _endpoint_ready(endpoint):
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early (code {proc.returncode})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"vllm serve not ready after {startup_timeout_s}s")
            time.sleep(5)
        yield endpoint
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_tritonbench(
    endpoint: str,
    model_name: str,
    results_dir: Path,
    levels: list[int],
    bench_root: Path | None = None,
    timeout_s: int = 7200,
    config: str | None = None,
) -> dict:
    root = bench_root or tritonbench_root()
    if not root.is_dir():
        raise FileNotFoundError(f"TritonBench not found at {root} — set SPARKDISTILL_TRITONBENCH_ROOT or vendor it")
    if config is None:
        config = _QUICK_CONFIG if levels == _QUICK_LEVELS else _FULL_CONFIG
    results_dir.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    command = [
        sys.executable,
        "-m",
        "tritonbench.cli",
        "eval",
        "--config",
        config,
        "--endpoint",
        endpoint,
        "--model",
        model_name,
        "--levels",
        *[str(level) for level in levels],
        "--output",
        str(results_dir.resolve()),
    ]
    subprocess.run(command, cwd=root, check=True, timeout=timeout_s)
    report = latest_report(results_dir, newer_than_ns=started_ns)
    report["coverage"] = level_coverage(report, levels)
    _enforce_level_coverage(report["coverage"])
    return report


def run_triton_benchmark(
    model_path: str,
    output_dir: Path,
    limit: int | None = None,
    endpoint: str | None = None,
) -> float:
    """Adapter used by `eval.benchmarks.run_benchmark` for the `triton` entry.

    A non-None `limit` (the harness's cheap re-verification mode) maps to the
    level-1-only quick run; a full run covers all levels. Endpoint resolution:
    explicit argument, then SPARKDISTILL_STUDENT_ENDPOINT, else serve the
    checkpoint locally with vLLM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "_tritonbench"
    levels = _QUICK_LEVELS if limit is not None else _FULL_LEVELS
    model_name = Path(model_path).name or model_path

    # The interpreter that runs TritonBench (this one, via `python -m tritonbench.cli`)
    # JIT-compiles each generated kernel's launcher against Python.h. Without the headers
    # every kernel fails to build and exec_pass is silently 0 — warn early (issue #283).
    if not python_headers_available():
        print(
            "warning: Python.h (python3-dev) not found for the eval interpreter — Triton "
            "cannot compile generated kernels, so exec_pass/correctness will be 0. Install "
            "python3-dev, or run eval under a uv-managed Python (issue #283).",
            file=sys.stderr,
        )

    endpoint = endpoint or os.environ.get("SPARKDISTILL_STUDENT_ENDPOINT")
    if endpoint:
        report = run_tritonbench(endpoint, model_name, results_dir, levels)
    else:
        with serve_checkpoint(model_path, served_model_name=model_name) as served:
            report = run_tritonbench(served, model_name, results_dir, levels)

    scores = summary_scores(report)
    gpu_architecture = detect_gpu_architecture()
    coverage = report.get("coverage") or level_coverage(report, levels)
    (output_dir / "triton.json").write_text(
        json.dumps(
            {
                "scores": scores,
                "levels": levels,
                "coverage": coverage,
                "gpu_architecture": gpu_architecture,
                "report": report,
            },
            indent=2,
        )
    )
    return scores["triton"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local path or HF hub id of the student checkpoint")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible endpoint already serving the checkpoint")
    parser.add_argument("--serve", action="store_true", help="serve the checkpoint locally with vLLM")
    parser.add_argument("--quick", action="store_true", help="level-1-only quick run (cheap re-verification)")
    parser.add_argument("--work-dir", type=Path, default=Path("eval/results/_work"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.endpoint and not args.serve and not os.environ.get("SPARKDISTILL_STUDENT_ENDPOINT"):
        parser.error("pass --endpoint, set SPARKDISTILL_STUDENT_ENDPOINT, or use --serve")

    headline = run_triton_benchmark(
        args.checkpoint,
        args.work_dir,
        limit=1 if args.quick else None,
        endpoint=args.endpoint,
    )
    detail = json.loads((args.work_dir / "triton.json").read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "scores": detail["scores"],
                "coverage": detail.get("coverage"),
                "gpu_architecture": detail.get("gpu_architecture"),
            },
            indent=2,
        )
    )
    print(f"triton={headline:.3f} — wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

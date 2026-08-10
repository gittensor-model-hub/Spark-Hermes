"""CLI: prompt set -> teacher trajectories -> jsonl dataset.

    python -m teacher.generate \
        --prompts data/prompts/phase1.jsonl \
        --out data/processed/phase1_trajectories.jsonl \
        --provider anthropic --provider openai
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from teacher.cot_recovery import RECOVERY_PROVIDER, recover_trajectory_cot
from teacher.providers import ANTHROPIC_TEACHER_MODEL, OPENAI_TEACHER_MODEL, Trajectory, get_teacher


def _iter_prompts(path: Path, limit: int | None) -> Iterator[dict]:
    yielded = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and yielded >= limit:
                break
            yield json.loads(line)
            yielded += 1


def generate_trajectories(
    prompts_path: Path,
    providers: list[str],
    max_tokens: int,
    temperature: float,
    limit: int | None,
    concurrency: int,
    thinking_budget: int | None,
    recover_cot: bool = False,
) -> Iterator[Trajectory]:
    # Each provider is pinned to a single teacher model, so `--model` is ignored
    # (see main()'s argument help). Forwarding it here would raise from get_teacher
    # whenever it doesn't match a provider's pin — and with the default two-provider
    # basket no single value can match both, so any --model would abort the run.
    teachers = [get_teacher(name) for name in providers]
    prompts = list(_iter_prompts(prompts_path, limit))

    # A GPT trajectory with no usable reasoning is a bare-answer SFT row — useless
    # for a pipeline that trains students to reproduce reasoning. Recover a plaintext
    # `<think>` trace from Fable when asked. Reuse the basket's Fable client if it has
    # one; otherwise construct a dedicated one. A missing key disables recovery rather
    # than aborting the run.
    recovery_teacher = None
    if recover_cot:
        try:
            recovery_teacher = next((t for t in teachers if t.name == RECOVERY_PROVIDER), None) or get_teacher(
                RECOVERY_PROVIDER
            )
        except Exception as exc:  # noqa: BLE001 - recovery is optional; never abort the run over it
            print(
                f"warning: CoT recovery disabled (could not init {RECOVERY_PROVIDER} teacher): {exc}",
                file=sys.stderr,
            )

    def _run(teacher, record: dict) -> Trajectory:
        return teacher.generate(
            record["prompt"],
            system=record.get("system"),
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # Keep the (provider, prompt) for each future so a failed call can be
        # reported without aborting the whole batch. Teacher APIs routinely
        # return transient errors (rate limits, 5xx) mid-run; letting one such
        # error propagate here would discard every other already-completed
        # trajectory from an expensive generation pass.
        future_to_prompt = {
            pool.submit(_run, teacher, record): (teacher.name, record.get("prompt", ""))
            for teacher in teachers
            for record in prompts
        }
        succeeded = 0
        failed = 0
        for future in as_completed(future_to_prompt):
            provider, prompt = future_to_prompt[future]
            try:
                trajectory = future.result()
            except Exception as exc:  # noqa: BLE001 - one flaky teacher call must not abort the batch
                failed += 1
                preview = prompt if len(prompt) <= 60 else f"{prompt[:57]}..."
                print(
                    f"warning: {provider} teacher call failed, skipping prompt {preview!r}: {exc}",
                    file=sys.stderr,
                )
                continue
            if recovery_teacher is not None:
                # Best-effort: a flaky recovery call must never discard a trajectory
                # that already cost a full teacher generation.
                try:
                    trajectory = recover_trajectory_cot(
                        trajectory,
                        recovery_teacher,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the original trajectory on failure
                    preview = prompt if len(prompt) <= 60 else f"{prompt[:57]}..."
                    print(
                        f"warning: CoT recovery failed for {provider} prompt {preview!r}, keeping original: {exc}",
                        file=sys.stderr,
                    )
            succeeded += 1
            yield trajectory

        # A single flaky call is tolerated, but a run where *nothing* succeeded
        # (e.g. a bad API key or no connectivity) must still fail loudly rather
        # than silently writing an empty output file.
        if failed and not succeeded:
            raise RuntimeError(
                f"all {failed} teacher call(s) failed; no trajectories were generated "
                "(check teacher API keys and connectivity)"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="jsonl file of {prompt, system?} records")
    parser.add_argument("--out", type=Path, required=True, help="output jsonl of trajectory records")
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        choices=["anthropic", "openai"],
        default=None,
        help=(
            "teacher provider (repeatable). Supported: "
            f"anthropic/{ANTHROPIC_TEACHER_MODEL}, openai/{OPENAI_TEACHER_MODEL}"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"ignored — anthropic is fixed to {ANTHROPIC_TEACHER_MODEL}, openai to {OPENAI_TEACHER_MODEL}",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None, help="only sample the first N prompts")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=4096,
        help="reasoning token budget for teachers that support extended thinking (0 to disable)",
    )
    parser.add_argument(
        "--recover-cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "when a GPT trajectory has no usable reasoning (encrypted/empty CoT), ask "
            "Claude Fable 5 to explain how to reach the answer and attach that plaintext "
            "rationale as the <think> trace (tagged metadata.cot_recovery). Costs one extra "
            "Fable call per recovered row; use --no-recover-cot to disable."
        ),
    )
    args = parser.parse_args(argv)

    providers = args.providers or ["anthropic", "openai"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    thinking_budget = args.thinking_budget or None

    count = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as out_f:
        for trajectory in generate_trajectories(
            args.prompts,
            providers,
            args.max_tokens,
            args.temperature,
            args.limit,
            args.concurrency,
            thinking_budget,
            recover_cot=args.recover_cot,
        ):
            out_f.write(json.dumps(trajectory.to_record()) + "\n")
            count += 1

    print(f"wrote {count} trajectories to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

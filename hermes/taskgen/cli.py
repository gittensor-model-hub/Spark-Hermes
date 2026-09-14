"""Generate tasks at scale: seeds in, gated tasks out, every rejection counted.

    python -m hermes.taskgen.cli --count 350 --out var/tasks/gen-1 \\
        --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b

Parallel because generation is I/O bound on a served model and the gate is process bound, and the
two overlap. Resumable because a run of several hundred will be interrupted, and re-generating a
task that already passed costs a GPU minute for nothing.

## The acceptance rate is the headline, not a footnote

A generator that reports "300 tasks written" and not "300 of 900 attempts, 412 killed by the
disagreement check" tells an operator nothing about whether their prompt is working. The failure
histogram is where the information is: heavy `checks_disagree_on_a_cheat` means the instruction is
not conveying what a withheld check is for; heavy `setup_is_deterministic` means it is not conveying
determinism; heavy `parse` means the model is not following the output format at all and no amount
of GPU time will fix it.

## Nothing is written before it is accepted

An accepted task's YAML, its withheld check and its salted commitment are written together, after
the gate. A directory of maybe-tasks is worse than no directory: the point of the gate is that
everything downstream can trust what is in here without re-checking it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hermes.taskgen.dna import TaskDNA, similarity
from hermes.taskgen.gate import ALL_CHECKS, Verdict, gate
from hermes.taskgen.seeds import SOURCES, SeedStats, read
from hermes.taskgen.synth import SynthError, synthesise, to_task_yaml

# Above this, two task prompts are the same task told twice. Chosen against the real corpus: two
# hand-written tasks from the same family (tc-log-rotation-order and tc-nested-archive-manifest, both
# ordering traps over a directory of files) score 0.06, so a threshold this high cannot mistake
# "same skill" for "same task" -- which is the distinction that matters, since a corpus wants many
# tasks per skill and no task twice.
DUPLICATE_AT = 0.5


def _rejection_for(prompt: str, *, generated: list[str], evaluation: list[str]) -> str:
    """Why this prompt cannot be accepted, or "".

    Two outcomes, deliberately not one bucket. `duplicate` is a corpus-efficiency problem: the same
    task told twice teaches one thing and is counted as two. `evaluation_contamination` is a
    correctness problem: an evaluation prompt in the training corpus invalidates every number
    measured on that suite, including `overfit_rate`. Folding them together would hide the second
    inside a count of the first, and the second is the one that has to stop a release.
    """
    if evaluation and _is_duplicate(prompt, evaluation)[0]:
        return "evaluation_contamination"
    if generated and _is_duplicate(prompt, generated)[0]:
        return "duplicate"
    return ""


def _is_duplicate(prompt: str, accepted_prompts: list[str]) -> tuple[bool, float]:
    """Whether this task has already been generated, and how close the nearest one is."""
    if not accepted_prompts:
        return False, 0.0
    nearest = max(similarity(prompt, other) for other in accepted_prompts)
    return nearest >= DUPLICATE_AT, nearest


def _evaluation_prompts() -> list[str]:
    """Prompts of the committed evaluation suite, which a generated task may never duplicate.

    The 19 hand-written tasks are what every published number in this repository rests on, and
    `overfit_rate` -- the check that would notice a corpus trained on its own benchmark -- is
    measured on them too. A generated task that restates one of them puts an evaluation prompt into
    the training corpus, and the resulting score is real, reproducible and meaningless.

    `admin/split.py` already refuses a rollout set that reached an evaluation task. This is the same
    refusal one stage earlier, where it costs a rejected generation instead of a discarded run.

    Missing or unreadable suite files yield an empty list rather than raising: this guard makes the
    corpus safer and must not be the reason a generation run cannot start. The caller says out loud
    when it is checking against nothing.
    """
    import yaml

    from hermesbench.tasks import TASKS_ROOT

    prompts: list[str] = []
    for path in sorted(Path(TASKS_ROOT).rglob("*.yaml")):
        try:
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            prompt = str((record or {}).get("prompt") or "").strip()
        except Exception:  # noqa: BLE001 - an unreadable suite file must not stop generation
            continue
        if prompt:
            prompts.append(prompt)
    return prompts


def _existing_prompts(out: Path) -> list[str]:
    """The prompts of tasks a previous session already accepted into `out`.

    `accepted` was seeded from disk on resume and `prompts` was not, so duplicate detection compared
    only against the current session and every task written by the previous one was invisible to it.
    Resume exists because a run of several hundred WILL be interrupted -- so the run most likely to
    produce duplicates was the one where the guard was switched off. Reproduced: a resumed run
    reported `accepted 3 of 3 (100%)` and the three additions were byte-identical to three tasks
    already in the directory, similarity 1.000, zero detections.

    A task whose prompt cannot be read back is skipped rather than fatal: it makes dedup weaker for
    that one task, and refusing to resume at all over a single unreadable file is worse.
    """
    import yaml

    prompts: list[str] = []
    for path in sorted(out.glob("*.yaml")):
        try:
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            prompt = str((record or {}).get("prompt") or "").strip()
        except Exception:  # noqa: BLE001 - a corrupt file weakens dedup, it does not stop the run
            continue
        if prompt:
            prompts.append(prompt)
    return prompts


def _action_budget(dna: TaskDNA) -> int:
    """Room to work, not the expected number of calls.

    `max(6, horizon[1])` set the budget to the seed's own upper estimate, which leaves an agent no
    slack for looking around, being wrong once, or checking its work -- and this harness counts
    verification against a separate allowance precisely because it wants that behaviour. Measured on a
    real probe: a task budgeted at 6 hit `step budget exhausted (6)` after three productive calls, and
    a truncated episode is scored as a failure of the agent.

    Twice the upper estimate, floored at 12. Generous on purpose: the step budget is not where
    difficulty should come from -- a trap in the workspace is -- and a task made hard by an
    ungenerous budget measures the budget.
    """
    return max(12, dna.horizon[1] * 2)


def task_id_for(index: int, dna: TaskDNA) -> str:
    """Stable and readable: the domain says what it is, the number says which attempt made it."""
    return f"gen-{dna.domain.split('_')[0][:4]}-{index:04d}"


def _incomplete(verdict: Verdict) -> bool:
    """Whether a verdict says `accepted` on a SUBSET of the checks.

    `gate` is allowed to skip check 9 when a candidate carries no alternate solution, and records
    the skip by leaving the name out of `checks_run` -- a deliberate design with a test naming it.
    What was missing is anyone reading that: this driver branched on `accepted` alone, so a task
    that never ran the method-pinning check was written to disk indistinguishably from one that
    passed it, and `checks_run` was persisted only for REJECTS. A withheld check that grades the
    method scores a correct agent as `overfit`, which is the failure check 9 exists to prevent, so
    it is not a check this pipeline may skip quietly.
    """
    return list(verdict.checks_run) != list(ALL_CHECKS)


def _write_accepted(
    out: Path,
    withheld_out: Path,
    synthesised: Any,
    *,
    salt: str,
    max_steps: int,
    checks_run: list[str] | None = None,
) -> dict[str, Any]:
    from hermes.harness import derive_task_salt, salted_digest

    task_id = synthesised.candidate.task_id
    body = synthesised.candidate.withheld_verify
    # Under the PER-TASK salt derived from the master, never the master itself: revealing one spent
    # task's salt must not make every commitment still sealed brute-forceable.
    commitment = salted_digest(body, derive_task_salt(salt, task_id))

    out.mkdir(parents=True, exist_ok=True)
    withheld_out.mkdir(parents=True, exist_ok=True)
    (out / f"{task_id}.yaml").write_text(
        to_task_yaml(synthesised, commitment=commitment, max_steps=max_steps), encoding="utf-8"
    )
    (withheld_out / f"{task_id}.sh").write_text(body, encoding="utf-8")
    # The reference solution is kept beside the withheld check, not with the task. It is the proof
    # the task is solvable and it is also a complete answer, so it lives on the private side.
    (withheld_out / f"{task_id}.solution.sh").write_text(synthesised.candidate.reference_solution, encoding="utf-8")
    # The alternate lives beside the reference, on the private side, for the same reason the
    # reference does: it is a complete answer. It is kept at all because it is the only artefact
    # that can re-run gate check 9 -- "the withheld check grades the outcome, not the method" is a
    # claim about a script, and without the script the claim cannot be re-tested after the withheld
    # check is ever edited. Unlike the cheat it is NOT a shortcut: it is a correct solution, and
    # putting it in `shortcuts` would assert it should be caught.
    if synthesised.candidate.alternate_solution.strip():
        (withheld_out / f"{task_id}.alternate.sh").write_text(
            synthesised.candidate.alternate_solution, encoding="utf-8"
        )
    # `checks_run` travels with the accepted task so the manifest records WHICH checks cleared it.
    # Without it an 8-of-10 acceptance and a 10-of-10 one are the same row forever.
    return {"task_id": task_id, "commitment": commitment, "checks_run": list(checks_run or [])}


def _save_reject(rejects: Path, task_id: str, verdict: Verdict | None, synthesised: Any, error: str) -> None:
    """Everything about a rejected attempt, so the histogram is actionable rather than decorative.

    A count of `setup_exits_zero` says the instruction is not producing runnable scripts. It does not
    say WHY, and without the script and the shell's own complaint the only way to find out is to
    generate more and read them by hand -- which is what this exists to stop.

    Written for rejects only. An accepted task's artefacts are already on the public side.
    """
    rejects.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "error": error,
        "failed_check": getattr(verdict, "failed_check", ""),
        "detail": getattr(verdict, "detail", ""),
        "checks_run": list(getattr(verdict, "checks_run", [])),
    }
    (rejects / f"{task_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if isinstance(synthesised, SynthError):
        # A parse failure has no candidate, only a reply. Writing it is the whole point: `parse: 9`
        # with nothing beside it cannot be diagnosed.
        (rejects / f"{task_id}.reply.txt").write_text(synthesised.raw or "<empty content>", encoding="utf-8")
        if synthesised.reasoning:
            (rejects / f"{task_id}.reasoning.txt").write_text(synthesised.reasoning, encoding="utf-8")
        return
    if synthesised is not None:
        candidate = synthesised.candidate
        for name, script in (
            ("setup", candidate.setup),
            ("verify", candidate.verify),
            ("withheld", candidate.withheld_verify),
            ("reference", candidate.reference_solution),
            ("cheat", candidate.cheat_solution),
        ):
            (rejects / f"{task_id}.{name}.sh").write_text(script, encoding="utf-8")


# Two different 429s come back from a hosted gateway and they mean different things: a per-key
# concurrency cap ("at most N concurrent requests per key"), which is yours to schedule around, and
# upstream capacity ("no serving capacity right now"), which is not. Both are transient and neither
# is a fact about the generator.
_RATE_LIMIT_MARKERS = ("rate_limit", "429", "too many requests", "no serving capacity", "concurrent requests")


def _is_rate_limit(exc: BaseException) -> bool:
    if type(exc).__name__ in ("RateLimitError", "APIStatusError") and getattr(exc, "status_code", None) == 429:
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RATE_LIMIT_MARKERS)


def _with_rate_limit_retry(complete: Any, *, attempts: int = 5, base_delay: float = 4.0) -> Any:
    """Retry a rate-limited generation instead of recording it as one the model failed.

    Without this every capacity blip consumes a `--max-attempts` slot and lands in the `generate`
    bucket, which is the catch-all for "the model client raised". Measured on a live gateway at a
    concurrency BELOW its own stated limit: 11 of 15 rejections were 429s, and `stage.json` reported
    10% acceptance where the true figure over attempts that reached the model was 33%. An operator
    reading that tunes the prompt when the problem is the backend -- an absence recorded as a
    measurement, which is the defect this package exists to keep out of the corpus.
    """
    import random
    import time

    def wrapped(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        last: BaseException | None = None
        for attempt in range(attempts):
            try:
                return complete(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below if it is not a rate limit
                if not _is_rate_limit(exc):
                    raise
                last = exc
                if attempt == attempts - 1:
                    break
                # Jittered, because every worker in the pool is hitting the same ceiling at the same
                # moment and a fixed backoff just re-synchronises them into the next collision.
                time.sleep(base_delay * (2**attempt) * (0.5 + random.random()))
        raise RuntimeError(f"rate limited after {attempts} attempts: {last}") from last

    return wrapped


def _attempt(dna: TaskDNA, index: int, complete: Any, *, gate_timeout: int) -> tuple[str, Verdict | None, Any]:
    task_id = task_id_for(index, dna)
    try:
        synthesised = synthesise(dna, task_id=task_id, complete=complete)
    except SynthError as exc:
        # A reply that could not be read is a distinct outcome from a task that was read and failed.
        # Folding them together hides the case where the model has stopped following the format,
        # which is the one case more GPU time cannot fix. The reply travels with the error so the
        # histogram can be acted on rather than only counted.
        return f"parse: {exc}", None, exc
    except Exception as exc:  # noqa: BLE001 - a served model can fail in many ways; none should stop the run
        # Named separately so the histogram distinguishes "the backend had no capacity" from "the
        # model produced something unusable". Folding them together makes a busy gateway read as a
        # broken generator.
        bucket = "rate_limited" if _is_rate_limit(exc) else "generate"
        return f"{bucket}: {type(exc).__name__}: {exc}", None, None
    verdict = gate(synthesised.candidate, timeout_s=gate_timeout)
    return "", verdict, synthesised


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=50, help="tasks to ACCEPT, not attempts to make")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means count * 4")
    parser.add_argument("--source", default="lambda", choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path, default=Path("var/tasks/gen-1"))
    parser.add_argument("--withheld-out", type=Path, default=None, help="defaults to <out>/withheld")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra HTTP header on every request, repeatable. Some gateways admit only particular "
        "clients and reject the SDK's own User-Agent; without this those endpoints refuse every "
        "call. Recorded in stage.json by NAME only -- a header can carry a credential and the "
        "manifest is written beside the tasks",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gate-timeout", type=int, default=120)
    parser.add_argument("--salt-file", type=Path, default=None, help="master withheld salt; required to write")
    parser.add_argument("--temperature", type=float, default=1.0, help="task variety wants sampling, not greedy")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="ceiling on one generation. Without one, a reply that starts repeating runs to the 32k "
        "context limit, which at this throughput is the ~900s the timeouts were measuring -- a "
        "runaway now ends as an unparseable reply in a fraction of the time. Set at 10k rather than "
        "6k because 6k truncated real replies mid-section: `parse` jumped to 5 of 12 rejections, "
        "which trades a slow waste for a fast one. A task with a heredoc-heavy setup genuinely "
        "needs the room.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=900,
        help="seconds per generation. The default of 300 in `openai_completion` is tuned for an "
        "agent turn; one generation here is six shell scripts, and at concurrency 8 the server "
        "queues them. Measured on a real run: 8 of 29 attempts died on APITimeoutError -- 28% of "
        "the GPU time spent, discarded, for a client setting rather than anything about the task.",
    )
    args = parser.parse_args(argv)

    import os as _os

    if (
        not _os.environ.get(args.api_key_env, "").strip()
        and "://" in args.base_url
        and "127.0.0.1" not in args.base_url
    ):
        # A remote endpoint with no credential produces one 401 per attempt, and a gateway that sees
        # a run of them rate-limits the caller. Measured: 20 attempts, 20 failures, then a 120-second
        # block -- caused by a shell prefix assignment that was expanded before it took effect, so the
        # key arrived empty. The histogram said `generate: 20`, which is true and says nothing.
        print(
            f"hermes.taskgen: ${args.api_key_env} is empty and --base-url is remote ({args.base_url}). "
            "Every attempt would fail authentication and a gateway will rate-limit the run for it. "
            "Export the key first.",
            file=sys.stderr,
        )
        return 2

    if args.salt_file is None or not args.salt_file.is_file():
        print(
            "hermes.taskgen: --salt-file is required. Every accepted task publishes a commitment to "
            "its withheld check, and a commitment needs the master salt. Without one the tasks would "
            "have to ship their withheld check in the clear, which is not a withheld check.",
            file=sys.stderr,
        )
        return 2
    salt = args.salt_file.read_text(encoding="utf-8").strip()

    withheld_out = args.withheld_out or (args.out / "withheld")
    rejects_dir = args.out / "rejected"
    already = {path.stem for path in args.out.glob("*.yaml")} if args.out.is_dir() else set()
    if already:
        # Flushed, like the per-acceptance lines. Under nohup stdout is a pipe and therefore block
        # buffered, so this sat unwritten while stderr's warnings appeared above it -- which reads as
        # a resume that did not happen, on the one line whose whole job is to say that it did.
        print(f"resuming: {len(already)} task(s) already accepted in {args.out}", flush=True)

    import os

    from hermesbench.policy import openai_completion

    headers: dict[str, str] = {}
    for item in args.header:
        name, sep, value = str(item).partition("=")
        if not sep or not name.strip():
            print(f"hermes.taskgen: --header must be NAME=VALUE, got {item!r}", file=sys.stderr)
            return 2
        headers[name.strip()] = value

    complete = _with_rate_limit_retry(
        openai_completion(
            default_headers=headers or None,
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get(args.api_key_env, ""),
            timeout_s=args.request_timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    max_attempts = args.max_attempts or args.count * 4
    stats = SeedStats()
    pool = read(args.source, limit=max_attempts, stats=stats)

    # Seeded with what a previous run already wrote, so `--count` means "this many tasks in the
    # directory" rather than "this many MORE". Counting only the current session's acceptances made a
    # resumed run target 150 on top of the 116 it had just found -- and the progress line said 13/150
    # while 129 files sat on disk, which is the kind of number nobody re-derives.
    accepted: list[dict[str, Any]] = [{"task_id": task_id} for task_id in sorted(already)]
    # Seeded from disk for the same reason `accepted` is: see `_existing_prompts`.
    prompts: list[str] = _existing_prompts(args.out)
    evaluation = _evaluation_prompts()
    if not evaluation:
        print(
            "warning: no evaluation-suite prompts could be read, so generated tasks are NOT being "
            "checked against the committed benchmark. A task that restates an evaluation task would "
            "put eval material into the training corpus undetected.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"decontaminating against {len(evaluation)} committed evaluation task(s)", flush=True)
    if already and len(prompts) < len(already):
        print(
            f"warning: {len(already) - len(prompts)} resumed task(s) had no readable prompt, so "
            "duplicate detection cannot compare against them",
            file=sys.stderr,
            flush=True,
        )
    failures: Counter[str] = Counter()
    attempted = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = {}
        for index, dna in enumerate(pool):
            if len(accepted) >= args.count:
                break
            if task_id_for(index, dna) in already:
                continue
            pending[executor.submit(_attempt, dna, index, complete, gate_timeout=args.gate_timeout)] = index
            attempted += 1
            if len(pending) < args.concurrency * 2:
                continue
            for future in as_completed(list(pending)):
                del pending[future]
                error, verdict, synthesised = future.result()
                if error:
                    failures[error.split(":")[0]] += 1
                    _save_reject(rejects_dir, f"attempt-{index:04d}", verdict, synthesised, error)
                elif (
                    verdict is not None
                    and verdict.accepted
                    and (reason := _rejection_for(synthesised.prompt, generated=prompts, evaluation=evaluation))
                ):
                    failures[reason] += 1
                    _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, reason)
                elif verdict is not None and verdict.accepted and _incomplete(verdict):
                    failures["incomplete_checks"] += 1
                    _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "incomplete_checks")
                elif verdict is not None and verdict.accepted:
                    prompts.append(synthesised.prompt)
                    accepted.append(
                        _write_accepted(
                            args.out,
                            withheld_out,
                            synthesised,
                            salt=salt,
                            max_steps=_action_budget(synthesised.dna),
                            checks_run=list(verdict.checks_run),
                        )
                    )
                    print(f"  accepted {accepted[-1]['task_id']} ({len(accepted)}/{args.count})", flush=True)
                elif verdict is not None:
                    # `or "unnamed"` because an empty bucket is unreadable and, worse, was hiding a
                    # real bug: accepted-past-target tasks were landing here with no failed_check.
                    failures[verdict.failed_check or "unnamed"] += 1
                    _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "")
                break

        for future in as_completed(list(pending)):
            error, verdict, synthesised = future.result()
            if error:
                failures[error.split(":")[0]] += 1
            elif (
                verdict is not None
                and verdict.accepted
                and (reason := _rejection_for(synthesised.prompt, generated=prompts, evaluation=evaluation))
            ):
                failures[reason] += 1
            elif verdict is not None and verdict.accepted and _incomplete(verdict):
                failures["incomplete_checks"] += 1
            elif verdict is not None and verdict.accepted:
                # Written even past the target. Submission already stopped at `--count`, so what is
                # still in flight is bounded by the concurrency -- and discarding a task that cleared
                # every executed check to keep a round number is the wrong trade. An earlier version
                # dropped these AND counted them as rejections under an empty name, which made the
                # generator look worse than it was and hid that the work was being thrown away.
                prompts.append(synthesised.prompt)
                accepted.append(
                    _write_accepted(
                        args.out,
                        withheld_out,
                        synthesised,
                        salt=salt,
                        max_steps=_action_budget(synthesised.dna),
                        checks_run=list(verdict.checks_run),
                    )
                )
            elif verdict is not None:
                failures[verdict.failed_check or "unnamed"] += 1
                _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "")

    # The rate is over THIS session's work. Seeding `accepted` with the resumed ids made `--count`
    # mean the right thing and immediately made this ratio mean the wrong one: 160 accepted over 49
    # attempted reported an acceptance rate of 3.265. A resumed task was not attempted here, so it
    # cannot be in the numerator of a rate whose denominator is attempts.
    fresh = len(accepted) - len(already)
    report = {
        "accepted": [entry["task_id"] for entry in accepted],
        "checks_run": {entry["task_id"]: entry.get("checks_run", []) for entry in accepted if entry.get("checks_run")},
        "all_checks": list(ALL_CHECKS),
        # The acceptance rate over attempts that actually REACHED the model. `acceptance_rate`
        # divides by every attempt including the ones a gateway refused, so on a busy backend it
        # reports the generator as broken when the backend is: measured 10% against a true 33%.
        "rate_limited": failures.get("rate_limited", 0),
        # Names only. A header can carry a credential and this manifest is written beside the tasks.
        "extra_headers": sorted(headers),
        "acceptance_rate_excluding_infrastructure": (
            round(fresh / max(1, attempted - failures.get("rate_limited", 0)), 3)
        ),
        "attempted": attempted,
        "resumed": len(already),
        "accepted_this_session": fresh,
        "acceptance_rate": round(fresh / attempted, 3) if attempted else 0.0,
        "rejected_by": dict(failures.most_common()),
        "seeds": stats.to_record(),
        "out": str(args.out),
        "rejected_dir": str(rejects_dir),
        "withheld_out": str(withheld_out),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    if already:
        print(f"{len(accepted)} task(s) in {args.out}: {len(already)} resumed, {fresh} added here")
    print(f"accepted {fresh} of {attempted} attempt(s) this session  ({report['acceptance_rate']:.0%})")
    for check, count in failures.most_common():
        print(f"  {count:>4}  {check}")
    print(f"\nwrote {args.out}/stage.json")
    if failures:
        print(f"every rejected attempt's scripts and the shell's own complaint are in {rejects_dir}")
    if failures.get("checks_disagree_on_a_cheat", 0) > len(accepted):
        print(
            "\nMost rejections are the disagreement check. That is the instruction failing to convey "
            "what a withheld check is FOR, not the model failing to write shell: it keeps producing a "
            "second copy of the published check. Sharpen requirement 3 before spending more GPU time."
        )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

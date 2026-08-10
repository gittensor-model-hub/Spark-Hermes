"""Run one round's tasks across the teacher field and write a submittable batch.

`hermes.arena` holds the loop; this is the thing that can actually be invoked. Until now the
arena had no `main()`, no argument parsing and no caller outside tests, which left the whole
pipeline downstream of it -- exports, manifest, attestation, the PR gate -- reachable only by
writing a bespoke script. A miner cannot run a design.

What this adds over calling `run_task` directly is the wiring that is easy to get wrong and
expensive to get wrong silently:

**One harness digest for the batch, fixed before anything is paid for.** `Tournament` refuses
a field whose candidates disagree on the harness, so the digest has to exist before the first
run rather than be derived per run. Building it up front also means an unpinned harness, a
dirty tree, or a withheld check with no salt stops the batch before the first token is bought
instead of after the last one.

**The selection policy is chosen before the results exist and travels in the digest.**
`--policy` names a declared order and `harness_digest` folds that policy's digest in, so
reordering it after seeing who won makes the new results *incomparable* to the old ones
rather than quietly better. A cost-ordered policy with no price book is refused outright,
because a tie-break dimension that can never fire is indistinguishable from one that never
mattered.

**Teachers are resolved from the registry, never from a flag.** `eval.rollout_track.check_manifest_work`
resolves every manifest teacher against `hermes.teachers.REGISTRY` at merge; refusing an
undeclared id here turns a rejected submission into a refused run.

**Each (task, teacher) pair gets its own workspace.** Sharing one lets the second teacher
inherit the first one's edits, which is the fair fight failing in the least visible way
available: the second trajectory starts from a partly-solved task and looks efficient.

**Exports are written even when the batch is not fit to submit.** `check_batch` problems set
the exit status, but the rows still land. A round where a provider was down is evidence about
the round, and deleting it to keep the exit code clean destroys the only record of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from hermes.arena import ArenaBatch, candidate_from, check_batch, export_digests, run_task, write_exports
from hermes.pin import load_tool_schemas
from hermes.preflight import PreflightReport, probe_overhead
from hermes.preflight import compare as compare_profiles
from hermes.preflight import problems as endpoint_problems
from hermes.protocol import DIALECTS
from hermes.selection import COST, RECOVERY, TOOL_CALLS, SelectionPolicy
from hermes.teachers import REGISTRY, TEACHER_FIELD_V1, Teacher, get
from hermesbench import BENCH_VERSION
from hermesbench.runner import HARNESS_DIR, LocalToolExecutor, run_episode
from hermesbench.tasks import Task, load_suite
from hermesbench.verify import OBSERVATION_LIMIT

# Declared tie-break orders, named rather than free-form so a round records *which* policy
# ran and so the set anyone can select from is reviewable in one place. A policy is only
# consulted for candidates already on the Pareto frontier: these say how to break a genuine
# tie, not how to score a run.
POLICIES: dict[str, SelectionPolicy] = {
    # The conservative default: let dominance decide and break a true tie on model id.
    # Right whenever nobody has justified a priority between cost and thoroughness.
    "evidence-only": SelectionPolicy(order=(), compare=(TOOL_CALLS, COST, RECOVERY)),
    # Prefer the cheaper of two incomparable attempts. Reasonable for bulk generation,
    # where a tie broken toward cost buys more tasks per dollar. Needs a price book.
    "thrift": SelectionPolicy(order=(COST, TOOL_CALLS), compare=(TOOL_CALLS, COST, RECOVERY)),
    # Prefer the attempt that recovered from a failure it hit -- the behaviour the suite is
    # thinnest on, and the one hardest to get from a teacher that never stumbles.
    "recovery-first": SelectionPolicy(order=(RECOVERY, TOOL_CALLS, COST), compare=(TOOL_CALLS, COST, RECOVERY)),
}

DEFAULT_POLICY = "evidence-only"

# Stated once and used twice: it configures the executor *and* goes into the harness digest.
# The runner previously took it from an executor it had already built, which meant the value
# in the digest and the value in force could drift apart if either construction changed.
DEFAULT_TOOL_TIMEOUT_S = 120


class GenerateError(RuntimeError):
    """The batch cannot be run as configured."""


def resolve_teachers(names: str) -> tuple[Teacher, ...]:
    """Teachers by id from the shipped registry.

    Refused here rather than at export because the gate refuses them at merge, and a batch
    generated with an undeclared teacher is trajectories nobody can submit.
    """
    ids = [n.strip() for n in names.split(",") if n.strip()]
    if len(ids) < 2:
        raise GenerateError(
            f"a tournament needs at least two teachers, got {ids or ['none']}. A field of one produces "
            "trajectories but no comparison, so every task would be incomparable and the batch would "
            "yield no winners and no DPO pairs."
        )
    if len(set(ids)) != len(ids):
        raise GenerateError("the same teacher id was named twice; it would be entered against itself")
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        raise GenerateError(
            f"unknown teacher ids {unknown}; the registry declares {sorted(REGISTRY)}. The PR gate "
            "resolves every manifest teacher against this same registry, so this batch could not be merged."
        )
    return tuple(get(i) for i in ids)


def load_price_book(path: Path | None) -> Any:
    """A price book from JSON, or None.

    Without one `CandidateRun.cost` stays None and the cost tie-break is skipped rather than
    fed a zero -- `PriceBook.price_of` raises instead of returning 0.0 precisely so an
    unpriced model cannot win every cost comparison on the strength of being unknown.
    """
    if path is None:
        return None
    from hermes.cost import Price, PriceBook

    body = json.loads(path.read_text(encoding="utf-8"))
    book = PriceBook(revision=body["revision"])
    for entry in body["prices"]:
        book.add(Price(**entry))
    return book


def check_policy_is_usable(name: str, policy: SelectionPolicy, price_book: Any) -> None:
    """Refuse a policy whose ordering dimensions cannot be computed.

    A cost-ordered policy with no prices does not fail -- it silently falls through to the
    next dimension on every task, so the batch records that `thrift` ran while nothing about
    it was thrifty. That is the same failure `price_of` refuses in the tie-break itself,
    arriving one layer earlier.
    """
    if COST in policy.order and price_book is None:
        raise GenerateError(
            f"policy {name!r} orders on cost but no --prices file was given. Every candidate's cost "
            "would be None, the dimension would never fire, and the manifest would record a "
            "cost-ordered policy that never considered cost."
        )


def build_harness_digest(*, suite_name: str, tasks: list[Task], policy: SelectionPolicy, tool_timeout_s: int) -> str:
    """The digest every candidate in this batch is stamped with.

    Computed once, before any teacher is called. `harness_digest` refuses an unpinned pin and
    `build_pin` refuses a dirty tree, so a run that could not be described afterwards fails
    now rather than after the money is spent.
    """
    from hermes.harness import digest_suite, fingerprint_task, harness_digest
    from hermes.pin import build_pin

    salt = os.environ.get("HERMESBENCH_WITHHELD_SALT", "")
    if any(t.has_hidden_tests for t in tasks) and len(salt) < 16:
        raise GenerateError(
            "this suite has withheld checks, so the batch needs HERMESBENCH_WITHHELD_SALT set to at "
            "least 16 characters. Without it the suite digest would either omit the withheld tests -- "
            "so two suites with different ones digest the same -- or publish an unsalted digest of "
            "them, which for a short shell command is a verification oracle rather than a commitment."
        )
    pin = build_pin(
        system_prompt=(HARNESS_DIR / "system_prompt.txt").read_text(encoding="utf-8"),
        tool_schemas=load_tool_schemas(HARNESS_DIR / "tools.json"),
        container_image_digest=os.environ.get("HERMESBENCH_IMAGE_DIGEST", ""),
    )
    suite = digest_suite(suite_name, [fingerprint_task(t, salt=salt) for t in tasks])
    return harness_digest(
        pin,
        suite=suite,
        executor="local",
        observation_limit=OBSERVATION_LIMIT,
        tool_timeout_s=tool_timeout_s,
        selection_policy=policy,
    )


def make_run_one(
    *,
    workspace_root: Path,
    dialect_name: str,
    harness_digest: str,
    executor: LocalToolExecutor,
    api_key_env: str,
    price_book: Any = None,
) -> Callable[[Task, Teacher], tuple[Any, Any]]:
    """A `run_one` for `arena.run_task`, driving one served endpoint per teacher."""
    schemas = load_tool_schemas(HARNESS_DIR / "tools.json")
    system_prompt = (HARNESS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
    dialect = DIALECTS[dialect_name]

    from hermesbench.policy import ServedModelPolicy, openai_completion

    def run_one(task: Task, teacher: Teacher) -> tuple[Any, Any]:
        # `teacher.endpoint` rather than `teacher.base_url`: the declared value names where a
        # model lives, while a deployment routes through whatever gateway holds the key.
        complete = openai_completion(
            base_url=teacher.endpoint,
            model=teacher.model,
            api_key=os.environ.get(api_key_env, ""),
        )
        policy = ServedModelPolicy(
            complete=complete,
            dialect=dialect,
            tool_schemas=schemas,
            system=system_prompt,
            scratch_pad=dialect.supports_scratch_pad,
        )
        workspace = workspace_root / task.task_id / teacher.teacher_id
        result = run_episode(task, policy, executor, workspace, price_book=price_book, model=teacher.model)
        return candidate_from(result, teacher=teacher, harness_digest=harness_digest), result.trajectory

    return run_one


def profile_endpoints(teachers: tuple[Teacher, ...], *, api_key_env: str = "OPENAI_API_KEY") -> PreflightReport:
    """Measure what each teacher's endpoint prepends, before the batch runs.

    `harness_digest` pins the system prompt we wrote. It cannot pin what a gateway puts in
    front of it, and at least one does: measured on 2026-08-09, a one-character message with
    no system message billed 70 prompt tokens on one endpoint and 62 on another, and the
    model reproduced instructions we never sent.

    Recorded rather than refused. Generating against an endpoint that injects a prompt is a
    legitimate thing to do knowingly; doing it without the batch saying so is not.
    """
    from hermes.cost import OPENAI, usage_from_provider
    from hermesbench.policy import openai_completion

    def prompt_tokens_of(response: Any) -> int:
        # `complete` returns (text, usage). The prompt total is the uncached remainder plus
        # what the cache served -- `Usage.input_tokens` is the remainder alone by design,
        # so summing is what makes this comparable across a cache hit and a cache miss.
        _text, usage = response
        normalised = usage_from_provider(usage, shape=OPENAI)
        return normalised.input_tokens + normalised.cached_input_tokens

    profiles = []
    for teacher in teachers:
        complete = openai_completion(
            base_url=teacher.endpoint,
            model=teacher.model,
            api_key=os.environ.get(api_key_env, ""),
        )
        profiles.append(
            probe_overhead(
                teacher_id=teacher.teacher_id,
                model=teacher.model,
                complete=complete,
                usage_of=prompt_tokens_of,
            )
        )
    return PreflightReport(profiles=profiles)


def generate(
    *,
    round_id: str,
    miner_id: str,
    tasks: list[Task],
    teachers: tuple[Teacher, ...],
    suite_name: str,
    workspace_root: Path,
    out_dir: Path,
    policy: SelectionPolicy,
    dialect_name: str = "hermes-4",
    allow_unsandboxed: bool = False,
    api_key_env: str = "OPENAI_API_KEY",
    attempts: int = 3,
    tool_timeout_s: int = DEFAULT_TOOL_TIMEOUT_S,
    harness_digest: str = "",
    price_book: Any = None,
    progress: Callable[[str], None] | None = None,
    make_runner: Callable[[str], Callable[[Task, Teacher], tuple[Any, Any]]] | None = None,
    endpoints: PreflightReport | None = None,
    previous_profile: dict[str, Any] | None = None,
) -> tuple[ArenaBatch, dict[str, Any], list[str]]:
    """Run the batch, write its exports, and report whether it is fit to submit.

    Two parameters exist so this function can wire a batch it is not itself running, and
    they belong together:

    `make_runner` takes the batch's harness digest and returns the `run_one` to drive it
    with. Defaults to served OpenAI-compatible endpoints resolved from each teacher.

    `harness_digest` overrides the digest instead of building one from this repository.
    A caller supplying its own runner is, by definition, not running the shipped harness --
    digesting this repo for it would stamp every candidate with an identity describing
    something that did not run. Left empty (the production path) the digest is built from
    the repo, which refuses an unpinned pin and a dirty working tree.

    Together they make the digest/export/manifest wiring -- the actual content of this
    function -- reusable by a local vLLM policy, a replay harness, or a test, none of which
    should have to reimplement it to get a correctly assembled batch.
    """
    # The timeout is passed in rather than read back off an executor, so the value in the
    # digest is the value that was configured. It also means the local executor -- which
    # refuses to construct without `allow_unsandboxed` -- is only built when it is the thing
    # actually going to run, not merely to be measured.
    digest = harness_digest or build_harness_digest(
        suite_name=suite_name, tasks=tasks, policy=policy, tool_timeout_s=tool_timeout_s
    )

    def served_runner(batch_digest: str) -> Callable[[Task, Teacher], tuple[Any, Any]]:
        return make_run_one(
            workspace_root=workspace_root,
            dialect_name=dialect_name,
            harness_digest=batch_digest,
            executor=LocalToolExecutor(allow_unsandboxed=allow_unsandboxed, timeout_s=tool_timeout_s),
            api_key_env=api_key_env,
            price_book=price_book,
        )

    run_one = (make_runner or served_runner)(digest)

    batch = ArenaBatch(round_id=round_id, miner_id=miner_id, harness_digest=digest, teachers=teachers)
    for index, task in enumerate(tasks, start=1):
        if progress is not None:
            progress(f"[{index}/{len(tasks)}] {task.task_id}")
        outcome = run_task(
            task,
            teachers,
            run_one=run_one,
            harness_digest=digest,
            policy=policy,
            primary_evidence=policy.primary_evidence,
            attempts=attempts,
        )
        batch.outcomes.append(outcome)
        if progress is not None:
            progress(describe_outcome(outcome))

    counts = write_exports(batch, out_dir)
    # Rewritten with the export digests folded in, so the attested manifest covers the
    # published rows. `write_exports` cannot do this itself: the digests do not exist until
    # the files it is writing have been written.
    manifest = batch.manifest(export_digests=export_digests(out_dir))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Widened deliberately: `write_exports` returns row counts, and what this function
    # reports is row counts plus a digest plus, optionally, the endpoint profile. Keeping
    # the narrower type would mean the caller could not be handed the profile at all.
    written: dict[str, Any] = {**counts, "manifest_digest": manifest["manifest_digest"]}

    problems = check_batch(batch)
    if endpoints is not None:
        # Written beside the exports rather than into the manifest. The manifest is
        # deliberately digests-only -- it is what the sealed worker is handed, and it has no
        # egress to check a token count against anything. The profile is operator evidence
        # about the conditions the rows were generated under, which is a different job.
        (out_dir / "endpoints.json").write_text(
            json.dumps(endpoints.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written["endpoints"] = endpoints.to_record()
        problems.extend(endpoint_problems(endpoints))
        if previous_profile is not None:
            # The reason to record it at all. An injected prompt is tolerable if it is stable
            # and disclosed; the failure is a corpus whose halves were conditioned
            # differently with nothing in either half saying so.
            problems.extend(compare_profiles(previous_profile, endpoints))
    return batch, written, problems


def describe_outcome(outcome: Any) -> str:
    """One line per task, including why a task produced nothing."""
    if not outcome.comparable:
        reasons = ", ".join(f"{r.teacher_id}:{r.error}" for r in outcome.errored) or "fewer than two candidates"
        return f"    incomparable ({reasons})"
    artifacts = outcome.artifacts
    winner = artifacts.winner.model if artifacts is not None and artifacts.winner is not None else "none"
    return f"    winner={winner} by {'evidence' if outcome.decided_by_evidence else 'policy'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round-id", required=True, help="the seeded round this batch answers")
    parser.add_argument("--miner-id", required=True, help="the miner this batch is submitted as")
    parser.add_argument("--out", type=Path, required=True, help="directory for sft/dpo/router jsonl + manifest.json")
    parser.add_argument("--workspace-root", type=Path, required=True, help="scratch directory for task workspaces")
    parser.add_argument("--suite", default=BENCH_VERSION, help="bench versions: 'v1', 'v0,v1', or 'all'")
    parser.add_argument("--tags", default="", help="comma-separated tag filter")
    parser.add_argument("--tasks", default="", help="comma-separated task ids, selected from --suite")
    parser.add_argument(
        "--teachers",
        default=",".join(t.teacher_id for t in TEACHER_FIELD_V1),
        help="comma-separated teacher ids from the registry",
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY, choices=sorted(POLICIES), help="declared tie-break order")
    parser.add_argument("--prices", type=Path, default=None, help="price book JSON; without it costs stay unknown")
    parser.add_argument("--dialect", default="hermes-4", choices=sorted(DIALECTS), help="Hermes wire dialect")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the endpoint's key")
    parser.add_argument("--attempts", type=int, default=3, help="retries per teacher on a transient provider error")
    parser.add_argument("--round-record", type=Path, default=None, help="seeded round record; enforces task scope")
    parser.add_argument(
        "--previous-profile",
        type=Path,
        default=None,
        help="a prior batch's endpoints.json; reports any endpoint whose hidden prompt overhead moved",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="do not measure endpoint prompt overhead (two extra calls per teacher)",
    )
    parser.add_argument("--list", action="store_true", help="print what would run, and exit without paying for it")
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help=(
            "run model-authored shell on this host. Provides no isolation of its own -- it asserts "
            "that this process is already inside a container or a throwaway machine."
        ),
    )
    args = parser.parse_args(argv)

    try:
        teachers = resolve_teachers(args.teachers)
        tasks = select_tasks(suite=args.suite, tags=args.tags, task_ids=args.tasks)
        check_scope(round_record=args.round_record, miner_id=args.miner_id, tasks=tasks)
        policy = POLICIES[args.policy]
        price_book = load_price_book(args.prices)
        check_policy_is_usable(args.policy, policy, price_book)
    except GenerateError as exc:
        print(f"hermes.generate_batch: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print(f"round={args.round_id} miner={args.miner_id} policy={args.policy} dialect={args.dialect}")
        for teacher in teachers:
            print(f"  teacher {teacher.teacher_id:<16} pin={teacher.pin:<14} rights={teacher.training_rights}")
        for task in tasks:
            print(f"  task    {task.task_id:<32} {task.prompt[:60]}")
        print(f"{len(tasks)} tasks x {len(teachers)} teachers = {len(tasks) * len(teachers)} episodes")
        return 0

    endpoints = None
    if not args.skip_preflight:
        print("preflight: measuring endpoint prompt overhead", file=sys.stderr)
        endpoints = profile_endpoints(teachers, api_key_env=args.api_key_env)
        for profile in endpoints.profiles:
            state = f"+{profile.fixed_overhead} tokens" if profile.measured else f"unmeasured ({profile.error[:60]})"
            print(f"    {profile.teacher_id:<18} {state}", file=sys.stderr)

    previous = json.loads(args.previous_profile.read_text(encoding="utf-8")) if args.previous_profile else None

    try:
        batch, written, problems = generate(
            round_id=args.round_id,
            miner_id=args.miner_id,
            tasks=tasks,
            teachers=teachers,
            suite_name=args.suite,
            workspace_root=args.workspace_root,
            out_dir=args.out,
            policy=policy,
            dialect_name=args.dialect,
            allow_unsandboxed=args.allow_unsandboxed,
            api_key_env=args.api_key_env,
            attempts=args.attempts,
            price_book=price_book,
            progress=lambda line: print(line, file=sys.stderr),
            endpoints=endpoints,
            previous_profile=previous,
        )
    except GenerateError as exc:
        print(f"hermes.generate_batch: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({**batch.to_record(), "exports": written}, indent=2, sort_keys=True))
    if problems:
        print(f"\nwrote {args.out}, but this batch is not fit to submit:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nwrote {args.out}: {written}", file=sys.stderr)
    return 0


def select_tasks(*, suite: str, tags: str = "", task_ids: str = "") -> list[Task]:
    """The tasks this batch will run, in the order they were named."""
    selected = load_suite(suite, tags=tuple(t.strip() for t in tags.split(",") if t.strip()))
    if task_ids:
        wanted = [t.strip() for t in task_ids.split(",") if t.strip()]
        by_id = {t.task_id: t for t in selected}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise GenerateError(f"task ids {missing} are not in suite {suite!r}")
        selected = [by_id[w] for w in wanted]
    if not selected:
        raise GenerateError(f"suite {suite!r} with tags {tags or '()'} selected no tasks")
    return selected


def check_scope(*, round_record: Path | None, miner_id: str, tasks: list[Task]) -> None:
    """Refuse tasks this miner was not assigned, before they are paid for.

    The gate checks scope at merge. Checking it here costs one file read and turns a
    rejected submission into a refused run.
    """
    if round_record is None:
        return
    from hermes.seed import verify_submission_scope

    record = json.loads(round_record.read_text(encoding="utf-8"))
    out_of_scope = verify_submission_scope(record, miner_id=miner_id, task_ids=[t.task_id for t in tasks])
    if out_of_scope:
        raise GenerateError(
            f"tasks {out_of_scope} are not assigned to {miner_id!r} in round {record.get('round_id')!r}; "
            "the gate refuses out-of-scope rows at merge, so running them would buy trajectories that "
            "cannot be submitted"
        )


if __name__ == "__main__":
    raise SystemExit(main())

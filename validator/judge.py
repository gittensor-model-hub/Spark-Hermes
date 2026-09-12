"""Execute exact admitted bundles, score them, and record round verdicts.

    python -m validator.pr_admission --help
    python -m validator.judge freeze --round r-001 --store /srv/spark-hermes/rounds
    python -m validator.judge judge --help

Admission requires authenticated GitHub metadata and an exact uploaded receipt;
the legacy ``accept --miner`` command refuses unauthenticated admission. Freeze
closes the declared window before any verdict can exist. Judging verifies the
admitted bundle before and after execution and applies the published baseline,
epoch and score policy. It never selects a bundle by miner name or upload recency.

A partially judged FROZEN round resumes by skipping miners with recorded verdicts.
Invalid or missing evidence leaves the round frozen. Use ``--no-settle`` in the
durable competition workflow: crown then commits winner and outbox intent before
projecting the SETTLED/salt-release state into the round snapshot.

Explicit ``--fixture-episodes`` replays labelled CPU logs only in an immutable
fixture store. See docs/competition-settlement.md for the complete operator flow,
durable path configuration and production trust requirements.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from validator.intake import Intake, IntakeError, Receipt
from validator.score import Scorecard, ScoreError, read_metrics, score
from validator.store import RoundStore

SCORECARD_DIR = Path("var/scorecards")


class JudgeError(RuntimeError):
    """A submission cannot be accepted or judged."""


@dataclass(frozen=True)
class _ExecutionIdentity:
    miner_id: str
    bundle_path: Path
    context_json: str


_execution_identity: ContextVar[_ExecutionIdentity | None] = ContextVar("judge_execution_identity", default=None)


def execution_context(miner_id: str, bundle_path: Path | None) -> dict[str, Any] | None:
    """Copy the admitted identity for a synchronous execution adapter, if judging.

    The three-argument callback remains compatible, including wrappers delegating
    to runner_for. Authority travels in the call context, never in mutable files
    or attributes on a reusable callback. Adapters starting another thread/process
    must explicitly forward this context to their final capture consumer.
    """
    identity = _execution_identity.get()
    if identity is None:
        return None
    if miner_id != identity.miner_id or bundle_path != identity.bundle_path:
        raise JudgeError("execution adapter differs from admitted miner/bundle path")
    return json.loads(identity.context_json)


@dataclass(frozen=True)
class Judgement:
    miner_id: str
    scorecard: Scorecard | None
    problem: str = ""

    @property
    def ok(self) -> bool:
        return self.scorecard is not None and not self.problem


def surface_dir(round_id: str, miner_id: str, *, root: Path | None = None, intake: Intake | None = None) -> Path:
    """Where the bundle a miner committed to actually lives.

    The intake store, not a directory in the repository. The surface is uploaded privately and the
    pull request carries only its digest, so there is nothing in the tree to run -- an earlier
    version of this module read `submissions/<round>/<miner>/` from a checkout, which was the shape
    when the files themselves were in the pull request.

    `root` overrides the store's location for tests. It is not a path a submitter can influence:
    round and miner ids are single path segments, checked at intake.
    """
    store = intake or Intake()
    base = root if root is not None else store.root
    return base / round_id / miner_id


def bundle_dir_for(receipt: Receipt, *, root: Path | None = None, intake: Intake | None = None) -> Path:
    """The exact bundle a receipt names, which is what gets run."""
    return surface_dir(receipt.round_id, receipt.miner_id, root=root, intake=intake) / receipt.submission_id


def accept(
    *,
    round_id: str,
    miner_id: str,
    receipt: Receipt,
    store: RoundStore | None = None,
    intake: Intake | None = None,
    now: float | None = None,
) -> Any:
    """Record a committed bundle into an open window. Returns the round receipt.

    Takes the intake receipt rather than a directory: the digest on it is the one the miner
    published in their pull request, and recording anything else would let the thing that runs
    differ from the thing that was committed to.
    """
    if receipt.miner_id != miner_id or receipt.round_id != round_id:
        raise JudgeError(f"receipt belongs to {receipt.miner_id!r} in {receipt.round_id!r}")
    raise JudgeError("direct miner/receipt admission has no authority; use validator.pr_admission")


def judge_one(
    *,
    window: Any,
    miner_id: str,
    intake: Intake | None = None,
    run: Callable[[str, Path, Path], Path],
    model_revision: str,
    harness_digest: str,
    repo_root: Path | None = None,
    workspace: Path,
) -> Judgement:
    """Run and score one standing submission. Never raises for one miner's sake."""
    from validator.pr_admission import admission_for

    try:
        admission = admission_for(window, miner_id)
    except ValueError as exc:
        return Judgement(miner_id, None, str(exc))
    expected = admission["bundle_sha256"]
    intake = intake or Intake()
    try:
        match = next(
            (
                r
                for r in intake.read_receipts()
                if r.round_id == window.round_id
                and r.miner_id == miner_id
                and r.bundle_sha256 == expected
                and r.submission_id == admission["submission_id"]
            ),
            None,
        )
        if match is None:
            return Judgement(miner_id, None, "no stored bundle matches admitted receipt and miner")
        if match.origin != admission["receipt_origin"]:
            raise IntakeError("receipt origin differs from admission")
        root = intake.verify(match, root=repo_root)
        # Reject bad baseline/epoch before invoking an expensive runner.
        from validator.score import baseline_arm, epoch_issues

        issues = epoch_issues(window.challenge.epoch, model_revision=model_revision, harness_digest=harness_digest)
        if issues:
            raise ScoreError("; ".join(issues))
        from validator.pr_admission import verify_admission

        verify_admission(admission)
        baseline_arm(window.challenge, origin=admission["origin"])
        identity = _ExecutionIdentity(
            miner_id,
            root,
            json.dumps({key: admission[key] for key in ("epoch", "round_id", "origin", "bundle_sha256", "task_id")}),
        )
    except (IntakeError, ScoreError, OSError, ValueError) as exc:
        return Judgement(miner_id, None, str(exc))

    try:
        from validator.intake import bundle_digest, capture_surface

        # Guard recording/replay adapters too. This check is not the final capture
        # check: consumers must still compare what they consume to this identity.
        if bundle_digest(capture_surface(root)) != expected:
            raise JudgeError("stored bundle changed before execution callback")
        token = _execution_identity.set(identity)
        try:
            log = run(miner_id, root, workspace)
        finally:
            _execution_identity.reset(token)
        intake.verify(match, root=repo_root)
    except Exception as exc:  # noqa: BLE001 - one miner's broken run must not stop the round
        return Judgement(miner_id, None, f"the run failed: {type(exc).__name__}: {exc}")

    try:
        rows = read_metrics(log)
        card = score(
            window=window,
            miner_id=miner_id,
            rows=rows,
            model_revision=model_revision,
            harness_digest=harness_digest,
        )
    except (ScoreError, ValueError, OSError, RuntimeError) as exc:
        return Judgement(miner_id, None, str(exc))
    return Judgement(miner_id, card)


def judge_round(
    *,
    round_id: str,
    run: Callable[[str, Path, Path], Path],
    model_revision: str,
    harness_digest: str = "",
    store: RoundStore | None = None,
    intake: Intake | None = None,
    repo_root: Path | None = None,
    workspace: Path | None = None,
    scorecard_dir: Path | None = None,
    settle: bool = True,
) -> list[Judgement]:
    """Run, score and record every standing submission, then grade and settle.

    Refuses an unfrozen round rather than freezing one: closing a window is a decision about the
    miners still working in it, and a judge that closed it as a side effect would take that
    decision by accident.
    """
    from hermes.round import FROZEN

    store = store or RoundStore()
    window = store.load(round_id)
    if window.state != FROZEN:
        raise JudgeError(
            f"round {round_id} is {window.state!r}; judging needs a FROZEN round. Verdicts cannot "
            "exist before the freeze -- the token that constructs one is minted by it -- and "
            "freezing here would close a window on miners still working in it."
        )

    harness = harness_digest or str(window.challenge.epoch.get("harness_digest") or "")
    workspace = workspace or Path("var/judge") / round_id
    workspace.mkdir(parents=True, exist_ok=True)
    cards = scorecard_dir or SCORECARD_DIR
    cards.mkdir(parents=True, exist_ok=True)

    already = set(window.verdicts)
    out: list[Judgement] = []
    for miner_id in sorted(window.submissions):
        if miner_id in already:
            # Skipped rather than re-judged. `record_verdict` allows one per miner, so a second
            # pass would raise on the first and abandon everyone after them.
            out.append(Judgement(miner_id, None, "already judged; skipped"))
            continue
        result = judge_one(
            window=window,
            miner_id=miner_id,
            intake=intake,
            run=run,
            model_revision=model_revision,
            harness_digest=harness,
            repo_root=repo_root,
            workspace=workspace,
        )
        out.append(result)
        if result.scorecard is None:
            continue
        window.record_verdict(
            window.token(),
            miner_id,
            passed=result.scorecard.accepted,
            notes=(result.scorecard.decision.reasons[0][:120] if result.scorecard.decision.reasons else "accepted"),
        )
        from validator.persistence import atomic_write

        atomic_write(
            cards / f"{round_id}-{miner_id}.json",
            (json.dumps(result.scorecard.to_record(), indent=2, sort_keys=True) + "\n").encode(),
        )

    from hermes.round import GRADED

    # Grade only when every standing submission carries a verdict, and let the round say why not.
    #
    # `RoundWindow.grade` refuses otherwise, and its reason is the right one: "a round graded with
    # a miner missing reports a rate over the wrong denominator, and the omission is
    # indistinguishable from a miner who never submitted." The first version of this graded
    # unconditionally and hit exactly that.
    #
    # The tempting repair -- record `passed=False` for a submission that could not be run -- is
    # worse than leaving the round frozen. `passed` means the withheld check failed. A surface that
    # was tampered with after acceptance, or whose run died, did not fail the withheld check; it
    # was never put to it, and writing a false verdict makes an infrastructure problem permanently
    # indistinguishable from a miner's bad strategy in the ledger.
    unjudged = sorted(set(window.submissions) - set(window.verdicts))
    store.save(window)
    if unjudged:
        return out

    window.grade(now=time.time())
    if settle and window.state == GRADED:
        window.settle(now=time.time())
    store.save(window)
    return out


def runner_for(
    *,
    round_id: str,
    base_url: str,
    model: str,
    api_key_env: str,
    task_id: str,
    repeats: int,
    repo_root: Path | None,
    allow_unsandboxed: bool,
    intake: Intake | None = None,
    dialect: str = "",
    task_root: Path | None = None,
    evaluation_context: dict[str, Any] | None = None,
    release_root: Path | None = None,
    serving_config: Path | None = None,
    fixture_serving: Path | None = None,
    fixture_root: Path | None = None,
    profile: str | None = None,
) -> Callable[[str, Path | None, Path], Path]:
    """A `run` callable that invokes the real runner.

    Injected rather than called directly so `judge_round` is testable without a served model --
    the ordering, the digest check, the skip-if-already-judged and the lifecycle transitions are
    the parts most likely to be wrong, and none of them needs a GPU to exercise.

    `round_id` is a parameter rather than derived from the workspace path. The first version read
    it off `workspace.name`, which happened to be the round id and would have silently pointed at
    the wrong surface the moment anyone passed a different workspace.

    Judging inherits the admitted execution context even if the adapter was built
    without one. Outside judging, an explicit context must include the expected
    bundle digest; omitting the entire context is only exploratory execution.
    """
    from hermesbench.execution import context_snapshot, execution_scope

    configured_json = context_snapshot(evaluation_context) if evaluation_context is not None else None

    def run(miner_id: str, bundle_path: Path | None, workspace: Path) -> Path:
        from hermesbench import runner
        from miner.evaluate import runner_argv

        context = json.loads(configured_json) if configured_json is not None else None
        admitted = execution_context(miner_id, bundle_path)
        if admitted is not None:
            if admitted["round_id"] != round_id or admitted["task_id"] != task_id:
                raise JudgeError("runner round/task differs from admitted execution")
            if context is not None and any(
                context_snapshot({key: context[key]}) != context_snapshot({key: value})
                for key, value in admitted.items()
                if key in context
            ):
                raise JudgeError("runner context differs from admitted execution")
            context = {**(context or {}), **admitted}
        if context is not None:
            expected = context.get("bundle_sha256")
            if bundle_path is None:
                if expected not in (None, "") or admitted is not None:
                    raise JudgeError("baseline cannot replace a committed miner execution")
            elif not isinstance(expected, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
                raise JudgeError("evaluation context requires an expected bundle commitment")
            if context.get("round_id") != round_id:
                raise JudgeError("runner round differs from evaluation context")
            if "task_id" in context and context["task_id"] != task_id:
                raise JudgeError("runner task differs from evaluation context")

        log = workspace / f"{miner_id}.jsonl"
        if log.exists():
            log.unlink()
        from validator.persistence import atomic_write

        context_path = workspace / f"{miner_id}.context.json" if context is not None else None
        scope = (
            execution_scope(bundle_path, context_path, context)
            if context is not None and context_path is not None
            else nullcontext()
        )
        with scope:
            if context_path is not None:
                atomic_write(context_path, json.dumps(context).encode())
            code = runner.main(
                runner_argv(
                    task_id=task_id,
                    base_url=base_url,
                    model=model,
                    api_key_env=api_key_env,
                    workspace_root=workspace / f"ws-{miner_id}",
                    episodes_out=log,
                    repeats=repeats,
                    miner_dir=bundle_path,
                    task_root=task_root,
                    evaluation_context=context_path,
                    allow_unsandboxed=allow_unsandboxed,
                    dialect=dialect,
                    release_root=release_root,
                    serving_config=serving_config,
                    fixture_serving=fixture_serving,
                    fixture_root=fixture_root,
                    profile=profile,
                )
            )
        if code != 0:
            raise JudgeError(f"the runner exited {code}")
        return log

    return run


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["accept", "freeze", "judge"])
    parser.add_argument("--round", dest="round_id", required=True)
    parser.add_argument("--miner", dest="miner_id", default="", help="accept only")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-env", default="NONE")
    parser.add_argument("--release-root", type=Path, help="active exact model/agent release authority")
    parser.add_argument("--serving-config", type=Path, help="trusted exact-model HTTPS deployment configuration")
    parser.add_argument("--fixture-serving", type=Path, help="explicit CPU serving responses; fixture stores only")
    parser.add_argument("--fixture-root", type=Path, help="existing fixture response authority root")
    parser.add_argument("--profile", choices=("bf16", "rtx5090-poc"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--intake-root", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--workspace", type=Path, help="durable episode root; logs use ROUND/MINER.jsonl")
    parser.add_argument("--scorecards", type=Path)
    parser.add_argument("--reason", default="", help="explicit reason required to freeze before the deadline")
    parser.add_argument("--fixture-episodes", type=Path, help="fixture stores only: replay labelled CPU episode logs")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--allow-unsandboxed", action="store_true")
    parser.add_argument(
        "--dialect",
        default="",
        help="wire dialect for the served model. Empty takes it from hermes/base_model.json, which "
        "is right when the served model IS the pinned one. Pass it when serving something else -- a "
        "mismatch shows up as malformed turns or as a model that never calls a tool, both of which "
        "read as the model being bad rather than as the harness instructing the wrong format.",
    )
    parser.add_argument("--no-settle", action="store_true", help="grade but leave the salt unreleased")
    args = parser.parse_args(argv)

    from hermes.base_model import load as load_pin

    try:
        store = RoundStore(args.store)
        if args.action == "accept":
            print("validator.judge: use validator.pr_admission with authenticated GitHub metadata", file=sys.stderr)
            return 2

        if args.action == "freeze":
            with store.lock(args.round_id):
                window = store.load(args.round_id)
                window.freeze(now=time.time(), reason=args.reason)
                store.save(window)
            print(f"round {args.round_id} is now FROZEN")
            return 0
        if not args.model and args.fixture_episodes is None:
            print("validator.judge: judge needs --model", file=sys.stderr)
            return 2
        window = store.load(args.round_id)
        intake = Intake()
        if args.intake_root is not None:
            intake.root = args.intake_root
        if args.receipts is not None:
            intake.receipts = args.receipts
        run = runner_for(
            round_id=args.round_id,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            task_id=window.task_id,
            repeats=args.repeats,
            repo_root=args.repo_root,
            allow_unsandboxed=args.allow_unsandboxed,
            dialect=args.dialect,
            task_root=Path(window.challenge.epoch["task_root"]) if window.challenge.epoch.get("task_root") else None,
            evaluation_context={"epoch": window.challenge.epoch, "round_id": window.round_id, "origin": store.identity},
            release_root=args.release_root,
            serving_config=args.serving_config,
            fixture_serving=args.fixture_serving,
            fixture_root=args.fixture_root,
            profile=args.profile,
        )
        revision = load_pin().revision
        if args.release_root is not None:
            from admin.competition_pair import active_pair, check_pair

            pair = active_pair(args.release_root)
            check_pair(pair, epoch=window.challenge.epoch, origin=store.identity, model=args.model)
            revision = pair["model_id"]
        elif args.profile is not None:
            from admin.competition_pair import base_profile

            revision = base_profile(args.profile)["revision"]
        if args.fixture_episodes is not None:
            if store.identity["mode"] != "fixture":
                raise JudgeError("fixture episode replay requires a fixture state root")
            source = args.fixture_episodes.resolve()
            revision = window.challenge.epoch["model_revision"]

            def fixture_run(miner_id: str, bundle_path: Path, workspace: Path) -> Path:
                from validator.persistence import atomic_write

                log = workspace / f"{miner_id}.jsonl"
                atomic_write(log, (source / args.round_id / f"{miner_id}.jsonl").read_bytes())
                return log

            run = fixture_run

        results = judge_round(
            round_id=args.round_id,
            run=run,
            model_revision=revision,
            store=store,
            intake=intake,
            repo_root=args.repo_root,
            workspace=(args.workspace or Path("var/judge")) / args.round_id,
            scorecard_dir=args.scorecards,
            settle=not args.no_settle,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"validator.judge: {exc}", file=sys.stderr)
        return 2

    for result in results:
        if result.scorecard is None:
            print(f"  {result.miner_id:<16} SKIPPED  {result.problem[:96]}")
        else:
            verdict = "ACCEPTED" if result.scorecard.accepted else "REFUSED"
            print(
                f"  {result.miner_id:<16} {verdict:<9} {result.scorecard.candidate.passes}/{result.scorecard.candidate.attempts} verified"
            )
    judged = [r for r in results if r.ok]
    print(f"\njudged {len(judged)} of {len(results)} standing submission(s)")
    window = store.load(args.round_id)
    print(f"round {args.round_id} is now {window.state.upper()}")
    if len(judged) != len(results):
        print(
            "the round was left unfrozen-of-verdicts and NOT graded: a round graded with a miner "
            "missing reports a rate over the wrong denominator. Re-run judge once the skipped "
            "submissions are resolved, or close them out deliberately.",
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "SCORECARD_DIR",
    "JudgeError",
    "Judgement",
    "accept",
    "execution_context",
    "judge_one",
    "judge_round",
    "main",
    "runner_for",
    "bundle_dir_for",
    "surface_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Take a merged surface, run it, score it, and record the verdict. The turn of the crank.

    python -m validator.judge accept --round e2e-1 --miner carol
    python -m validator.judge judge  --round e2e-1 --model qwen3.6-27b --repeats 10

`eval.strategy_track` decides whether a surface may merge and `validator.score` says whether it
beat the bar. Between them the three steps that actually happen -- run, score, record -- were three
commands typed by hand in an end-to-end test. This is those three, in the order the round lifecycle
requires.

## Two commands because the lifecycle has two moments

`accept` records a merged submission into an **open** window and returns a receipt. That is a
merge-time event: it is what a miner is owed the moment their pull request lands, and it carries
no correctness information because none exists yet.

`judge` runs every standing submission after the window has **frozen**. Verdicts cannot exist
before then -- `Verdict` refuses to be constructed without the `FreezeToken` the freeze mints,
which is what stops a correctness result existing during an open round. Then it grades, which is
what makes verdicts publishable, and settles, which releases the per-task salt for audit.

## The surface is re-verified before it is run

The gate checked the pull request head. This runs whatever is in the merged tree now, and those
are not the same object: a later commit can touch a directory an earlier gate approved. So the
digest recorded on the receipt at `accept` time is recomputed here and compared, and a mismatch
refuses the run.

Without that, "the gate approved this surface" and "this is the surface being executed" are two
claims joined by an assumption. The whole reason validator-side execution answers the cheating
question is that the files being run are the files that were checked.

## Judging is resumable and does not double-count

`record_verdict` allows one verdict per miner, so a second `judge` on the same round would raise
on the first miner already judged and abandon the rest. Miners already carrying a verdict are
skipped instead, which makes a partially-completed judge safe to re-run -- and a run that dies
after three of ten submissions is the normal way this fails, not an exotic one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from validator.intake import Intake, Receipt
from validator.score import Scorecard, ScoreError, score
from validator.store import RoundStore

SCORECARD_DIR = Path("var/scorecards")


class JudgeError(RuntimeError):
    """A submission cannot be accepted or judged."""


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
    store = store or RoundStore()
    window = store.load(round_id)
    if receipt.miner_id != miner_id or receipt.round_id != round_id:
        raise JudgeError(
            f"receipt {receipt.submission_id} belongs to {receipt.miner_id!r} in {receipt.round_id!r}, "
            f"not {miner_id!r} in {round_id!r}"
        )
    # The bundle's own file paths, read from the store. `RoundWindow.submit` checks them against
    # the contract, which is defence in depth rather than duplication: intake validated them on the
    # way in, and this re-checks what is actually on disk at the moment it is accepted into a round.
    # Passing the submission id here instead -- a hex string -- made every accept come back
    # `refused`, correctly, because a hex string is not an allowed miner file.
    root = bundle_dir_for(receipt, intake=intake)
    paths = sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file() and f.name != "bundle.json")
    round_receipt = window.submit(
        miner_id,
        paths=paths,
        payload_digest=receipt.bundle_sha256,
        received_at=receipt.received_at if now is None else now,
    )
    store.save(window)
    return round_receipt


def judge_one(
    *,
    window: Any,
    miner_id: str,
    intake: Intake | None = None,
    run: Callable[[str, Path], Path],
    model_revision: str,
    harness_digest: str,
    repo_root: Path | None = None,
    workspace: Path,
) -> Judgement:
    """Run and score one standing submission. Never raises for one miner's sake."""
    from hermes.challenge import episode_metrics_of
    from hermesbench.sink import read_episodes

    standing = window.submissions.get(miner_id)
    expected = getattr(standing, "payload_digest", "") if standing else ""
    intake = intake or Intake()
    match = next(
        (r for r in intake.read_receipts() if r.round_id == window.round_id and r.bundle_sha256 == expected),
        None,
    )
    if match is None:
        return Judgement(miner_id, None, f"no stored bundle digests to {expected[:23]}...; nothing to run")

    root = bundle_dir_for(match, root=repo_root, intake=intake)
    if not root.is_dir():
        return Judgement(miner_id, None, f"{root.as_posix()} is missing from the intake store")

    # The bundle on disk must still digest to what the miner committed to. The store is the
    # validator's own, so this catches corruption and local tampering rather than a miner -- but an
    # unchecked store is one where "the digest was published" and "this is what ran" are two claims
    # joined by an assumption.
    from validator.intake import bundle_digest

    files = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "bundle.json"
    }
    current = bundle_digest(files)
    if expected and current != expected:
        # The gate checked the pull request head; this runs the merged tree. A later commit can
        # touch a directory an earlier gate approved, and then "the gate approved this" and "this
        # is what ran" are two claims joined by an assumption.
        return Judgement(
            miner_id,
            None,
            f"the stored bundle no longer digests to what was committed: the commitment names "
            f"{expected[:23]}... and the store holds {current[:23]}.... Refusing to run something "
            "other than what was published.",
        )

    try:
        log = run(miner_id, workspace)
    except Exception as exc:  # noqa: BLE001 - one miner's broken run must not stop the round
        return Judgement(miner_id, None, f"the run failed: {type(exc).__name__}: {exc}")

    try:
        rows = [episode_metrics_of(r) for r in read_episodes(log)]
        card = score(
            window=window,
            miner_id=miner_id,
            rows=rows,
            model_revision=model_revision,
            harness_digest=harness_digest,
        )
    except ScoreError as exc:
        return Judgement(miner_id, None, str(exc))
    return Judgement(miner_id, card)


def judge_round(
    *,
    round_id: str,
    run: Callable[[str, Path], Path],
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
        (cards / f"{round_id}-{miner_id}.json").write_text(
            json.dumps(result.scorecard.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
) -> Callable[[str, Path], Path]:
    """A `run` callable that invokes the real runner.

    Injected rather than called directly so `judge_round` is testable without a served model --
    the ordering, the digest check, the skip-if-already-judged and the lifecycle transitions are
    the parts most likely to be wrong, and none of them needs a GPU to exercise.

    `round_id` is a parameter rather than derived from the workspace path. The first version read
    it off `workspace.name`, which happened to be the round id and would have silently pointed at
    the wrong surface the moment anyone passed a different workspace.
    """

    def run(miner_id: str, workspace: Path) -> Path:
        from hermesbench import runner
        from miner.evaluate import runner_argv

        log = workspace / f"{miner_id}.jsonl"
        if log.exists():
            log.unlink()
        code = runner.main(
            runner_argv(
                task_id=task_id,
                base_url=base_url,
                model=model,
                api_key_env=api_key_env,
                workspace_root=workspace / f"ws-{miner_id}",
                episodes_out=log,
                repeats=repeats,
                miner_dir=_bundle_for(round_id, miner_id, repo_root, intake),
                allow_unsandboxed=allow_unsandboxed,
            )
        )
        if code != 0:
            raise JudgeError(f"the runner exited {code}")
        return log

    return run


def _bundle_for(round_id: str, miner_id: str, repo_root: Path | None, intake: Intake | None = None) -> Path:
    """The stored bundle for a miner's standing commitment in this round.

    Looked up by (round, miner) rather than derived from a path. An earlier version read the round
    id off `workspace.name`, which happened to be right and would have pointed at the wrong bundle
    the moment anyone passed a different workspace.
    """
    intake = intake or Intake()
    match = next((r for r in intake.read_receipts() if r.round_id == round_id and r.miner_id == miner_id), None)
    if match is None:
        raise JudgeError(f"no stored bundle for {miner_id!r} in round {round_id!r}")
    return bundle_dir_for(match, root=repo_root, intake=intake)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["accept", "judge"])
    parser.add_argument("--round", dest="round_id", required=True)
    parser.add_argument("--miner", dest="miner_id", default="", help="accept only")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-env", default="NONE")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--allow-unsandboxed", action="store_true")
    parser.add_argument("--no-settle", action="store_true", help="grade but leave the salt unreleased")
    args = parser.parse_args(argv)

    from hermes.base_model import load as load_pin

    store = RoundStore(args.store)
    try:
        if args.action == "accept":
            if not args.miner_id:
                print("validator.judge: accept needs --miner", file=sys.stderr)
                return 2
            # The CLI resolves the miner's standing upload; `accept` takes the receipt itself so
            # the digest recorded on the round is the one the miner published, not one re-derived
            # from whatever is on disk at accept time.
            intake = Intake()
            standing = [
                r for r in intake.read_receipts() if r.round_id == args.round_id and r.miner_id == args.miner_id
            ]
            if not standing:
                print(
                    f"validator.judge: {args.miner_id!r} has uploaded nothing for round {args.round_id!r}",
                    file=sys.stderr,
                )
                return 2
            if len(standing) > 1:
                # Which bundle is judged is decided by the digest in the pull request, not by
                # recency. Guessing here would evaluate something the miner did not commit to.
                print(
                    f"validator.judge: {args.miner_id!r} has {len(standing)} uploads in round "
                    f"{args.round_id!r}; the pull request's digest decides which is judged, so accept "
                    "it through the gate rather than from the command line",
                    file=sys.stderr,
                )
                return 2
            round_receipt = accept(
                round_id=args.round_id, miner_id=args.miner_id, receipt=standing[0], store=store, intake=intake
            )
            print(f"{round_receipt.outcome}  {round_receipt.miner}  {standing[0].bundle_sha256[:23]}...")
            return 0 if round_receipt.outcome == "accepted" else 1

        if not args.model:
            print("validator.judge: judge needs --model", file=sys.stderr)
            return 2
        window = store.load(args.round_id)
        run = runner_for(
            round_id=args.round_id,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            task_id=window.task_id,
            repeats=args.repeats,
            repo_root=args.repo_root,
            allow_unsandboxed=args.allow_unsandboxed,
        )
        results = judge_round(
            round_id=args.round_id,
            run=run,
            model_revision=load_pin().revision,
            store=store,
            repo_root=args.repo_root,
            workspace=Path("var/judge") / args.round_id,
            settle=not args.no_settle,
        )
    except (JudgeError, ScoreError) as exc:
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
    "judge_one",
    "judge_round",
    "main",
    "runner_for",
    "bundle_dir_for",
    "surface_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())

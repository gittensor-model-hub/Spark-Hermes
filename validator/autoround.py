"""Decide the next action in the round cycle, and open the round nothing else opens.

    python -m validator.autoround next --store /srv/spark-hermes/rounds \
        --challenges datasets/challenges --episodes /srv/spark-hermes/baselines
    python -m validator.autoround open --store ... --challenges ... --episodes ... --hours 1

## Why this decides rather than executes

Everything the cycle needs already exists as a command, and `crown.yml` already runs one of them
on a schedule. What is missing is the thing that looks at a stored round and says which command is
next -- so today a person reads the state and dispatches by hand, every hour, forever.

This does NOT run freeze, admission, judging or delivery. Those carry credentials and a model
endpoint, and the settlement runbook separates them on purpose: repository credentials must never
reach evaluation code, and scheduled runs must never deliver. A single process that did all of it
would have to hold every credential at once, which is the arrangement that separation exists to
prevent. So `next` reports, the workflow dispatches the job that already has the right scope, and
the blast radius of a bug here stays "wrong thing reported".

The one exception is `open`. Opening the next round is pure local state -- a packet, an episode log
and a store write, no credentials, no model -- and nothing in the repository does it, so the cycle
cannot close without a human. That one action is performed here.

## One step per invocation

`next` reads state and names one action. It changes nothing, so it is safe to run every minute and
safe to run twice. That matters because the state machine is already the authority -- `freeze` is
idempotent from FROZEN, `grade` refuses while any submission is unscored -- and a driver that
tried to hold its own schedule would be a second, disagreeing source of truth.

## Stalls are reported, never inferred from silence

"Invalid or missing evidence leaves the round frozen" is a real outcome of judging, and it produces
no crown, no next round and no error. Left alone it reads exactly like a quiet hour. `next` reports
`stalled` once a round has sat in one state past `--stall-after`, because the difference between
"nothing to do" and "nothing has happened for six hours" is the whole reason anyone is watching.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes.round import FROZEN, GRADED, OPEN, SETTLED
from validator.crown import available_tasks, next_task
from validator.round_loop import LoopError, open_from_packet
from validator.store import RoundStore, StoreError

# Actions this can report. Each names a command an operator or workflow already has.
OPEN_NEXT = "open_next_round"
WAIT = "wait"
FREEZE = "freeze"
JUDGE = "judge"
CROWN = "crown"
STALLED = "stalled"
BLOCKED = "blocked"

# A round that has not moved in this long is not quiet, it is stuck. Six hours rather than one:
# a barren round is legitimate and an hourly cadence has real jitter, so alerting at the first
# missed hour would train whoever reads it to ignore the alert.
DEFAULT_STALL_AFTER_S = 6 * 3600.0


class AutoRoundError(RuntimeError):
    """The cycle cannot be advanced from what is stored."""


@dataclass(frozen=True)
class Decision:
    """What to do next, and the evidence for saying so."""

    action: str
    round_id: str = ""
    reason: str = ""
    detail: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"action": self.action, "round_id": self.round_id, "reason": self.reason}
        if self.detail:
            record["detail"] = self.detail
        return record


def _active(store: RoundStore) -> tuple[Any | None, list[tuple[str, str]]]:
    """The one round that is not SETTLED, or None. Unreadable rounds travel with it.

    More than one unsettled round is refused rather than picked between. Two open windows means
    miners could be submitting to either and a crown would be selected over one of them, which is
    a fact about the operator's state that no default here can repair.
    """
    windows, failed = store.load_all()
    live = [w for w in windows.values() if w.state != SETTLED]
    if len(live) > 1:
        raise AutoRoundError(
            "more than one round is unsettled ("
            + ", ".join(sorted(w.round_id for w in live))
            + "); settle or remove all but one before the cycle can advance"
        )
    return (live[0] if live else None), failed


def _last_settled_task(store: RoundStore) -> str:
    """The task the most recently opened round used, so rotation continues from it."""
    windows, _ = store.load_all()
    if not windows:
        return ""
    newest = max(windows.values(), key=lambda w: w.opened_at)
    return str(getattr(newest.challenge, "task_id", "") or "")


def next_round_id(store: RoundStore, *, prefix: str = "r") -> str:
    """The next id in sequence, continuing whatever numbering is already stored.

    Derived rather than timestamped: a round id appears in every receipt and every published proof,
    and `r-013` is something a person can hold in their head while reading a ledger.
    """
    highest = 0
    for round_id in store.round_ids():
        tail = round_id.rsplit("-", 1)[-1]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return f"{prefix}-{highest + 1:03d}"


def resolve_episodes(episodes: Path, task_id: str) -> Path:
    """The episode log a packet's baseline was measured in.

    A packet cannot name its own log -- `Challenge.to_record()` publishes aggregate statistics and
    strips everything else -- and `open_from_packet` needs the attempts, not the summary, or
    `token_spread` comes out zero and the efficiency gate accepts margins it should refuse.

    So: a file is used as given, and a directory is searched for `<task_id>.jsonl` first, then any
    log naming the task. Sorted, so the same directory always resolves the same way.
    """
    if episodes.is_file():
        return episodes
    if not episodes.is_dir():
        raise AutoRoundError(f"no episode log or directory at {episodes}")
    direct = episodes / f"{task_id}.jsonl"
    if direct.is_file():
        return direct
    for candidate in sorted(episodes.glob("*.jsonl")):
        try:
            if f'"{task_id}"' in candidate.read_text(encoding="utf-8"):
                return candidate
        except OSError:
            continue
    raise AutoRoundError(
        f"no episode log under {episodes} mentions {task_id}. A round must be built from the same "
        "evidence its baseline was measured in"
    )


def choose_task(challenges: Path, *, current: str) -> str:
    """The next task from the pool, rotating round-robin from the current one."""
    pool = available_tasks(challenges)
    if not pool:
        raise AutoRoundError(
            f"no challenge packets in {challenges}; the round cycle consumes a pool and nothing is "
            "stocking it. Run generation, probe and `hermes.challenge` to produce packets"
        )
    return next_task(current, pool) if current else pool[0]


def decide(
    store: RoundStore,
    *,
    challenges: Path,
    now: float | None = None,
    stall_after_s: float = DEFAULT_STALL_AFTER_S,
) -> Decision:
    """Read the stored state and name exactly one next action. Changes nothing."""
    moment = time.time() if now is None else now
    try:
        window, failed = _active(store)
    except StoreError as exc:
        return Decision(BLOCKED, reason=f"the round store could not be read: {exc}")
    except AutoRoundError as exc:
        # Two live windows is the one condition that most needs a readable message, and the
        # first version of this raised straight through `main`, so it arrived as a traceback.
        return Decision(BLOCKED, reason=str(exc))

    if failed:
        # Reported, not fatal: `load_all` deliberately survives one bad snapshot so a validator can
        # keep serving. But a round nobody can load is also a round nobody can settle.
        names = ", ".join(f"{rid} ({why})" for rid, why in failed)
        return Decision(BLOCKED, reason=f"stored round(s) will not load: {names}")

    if window is None:
        try:
            task = choose_task(challenges, current=_last_settled_task(store))
        except AutoRoundError as exc:
            return Decision(BLOCKED, reason=str(exc))
        return Decision(
            OPEN_NEXT,
            round_id=next_round_id(store),
            reason=f"no unsettled round; next task is {task}",
            detail={"task_id": task},
        )

    # Stall is measured from when the round left OPEN, not from when it opened: a two-hour window
    # followed by four quiet hours is four hours stuck, not six. `frozen_at` is the public record
    # of that transition and a GRADED round passed through it too, so it serves both states.
    frozen = window.frozen_at
    since = float(frozen.frozen_at) if frozen is not None else float(window.opened_at)
    age = moment - since
    stalled = Decision(
        STALLED,
        round_id=window.round_id,
        reason=f"round has been {window.state} for {age / 3600:.1f}h",
        detail={"state": window.state, "since": since},
    )

    if window.state == OPEN:
        remaining = float(window.deadline) - moment
        if remaining > 0:
            return Decision(
                WAIT,
                round_id=window.round_id,
                reason=f"window open for another {remaining / 60:.0f} min",
                detail={"deadline": window.deadline, "seconds_remaining": round(remaining, 1)},
            )
        # Past the deadline the window is closed to submissions whatever its state says -- the
        # deadline admits, the freeze only records -- so this is bookkeeping, not a cutoff.
        return Decision(
            FREEZE,
            round_id=window.round_id,
            reason=f"deadline passed {-remaining / 60:.0f} min ago",
            detail={"deadline": window.deadline},
        )

    if window.state == FROZEN:
        if age > stall_after_s:
            return stalled
        return Decision(JUDGE, round_id=window.round_id, reason="frozen and awaiting verdicts")

    if window.state == GRADED:
        if age > stall_after_s:
            return stalled
        return Decision(CROWN, round_id=window.round_id, reason="graded and awaiting crown selection")

    return Decision(BLOCKED, round_id=window.round_id, reason=f"unrecognised round state {window.state!r}")


def open_next(
    store: RoundStore,
    *,
    challenges: Path,
    episodes: Path,
    hours: float,
    now: float | None = None,
    round_id: str = "",
) -> Any:
    """Open the next round over the next task in the pool. The one action this module performs.

    Refuses while a round is unsettled. `open_from_packet` already refuses to overwrite a stored
    round -- reopening one would reset a window miners may have submitted to and lose their
    receipts -- and this adds the case that file check cannot see: a NEW id opened while the
    previous round is still live, which is two open windows rather than one overwritten.
    """
    window, failed = _active(store)
    if failed:
        raise AutoRoundError("stored round(s) will not load: " + ", ".join(rid for rid, _ in failed))
    if window is not None:
        raise AutoRoundError(
            f"round {window.round_id} is {window.state}, not settled; opening another would leave two "
            "windows taking submissions for one crown"
        )
    task = choose_task(challenges, current=_last_settled_task(store))
    packet = challenges / f"{task}.json"
    if not packet.is_file():
        raise AutoRoundError(f"{packet} is missing; the pool listed {task} but the packet is not there")
    return open_from_packet(
        packet_path=packet,
        episodes_path=resolve_episodes(episodes, task),
        round_id=round_id or next_round_id(store),
        hours=hours,
        store=store,
        now=now,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["next", "open"])
    parser.add_argument("--store", type=Path, default=None, help="round store root (default var/rounds)")
    parser.add_argument("--challenges", type=Path, default=Path("datasets/challenges"), help="the packet pool")
    parser.add_argument("--episodes", type=Path, default=None, help="episode log, or a directory of them")
    parser.add_argument("--hours", type=float, default=1.0, help="window length for a round this opens")
    parser.add_argument("--round", dest="round_id", default="", help="override the derived round id")
    parser.add_argument(
        "--stall-after",
        type=float,
        default=DEFAULT_STALL_AFTER_S,
        help="seconds in one state before a round is reported as stalled rather than pending",
    )
    parser.add_argument("--json", action="store_true", help="print the decision as JSON")
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="append action/round_id as GITHUB_OUTPUT key=value lines, for a workflow to dispatch on",
    )
    args = parser.parse_args(argv)

    store = RoundStore(args.store) if args.store else RoundStore()

    if args.action == "open":
        if args.episodes is None:
            print("validator.autoround: --episodes is required to open a round", file=sys.stderr)
            return 2
        try:
            window = open_next(
                store,
                challenges=args.challenges,
                episodes=args.episodes,
                hours=args.hours,
                round_id=args.round_id,
            )
        except (AutoRoundError, LoopError, StoreError) as exc:
            print(f"validator.autoround: {exc}", file=sys.stderr)
            return 1
        print(f"opened {window.round_id} over {window.challenge.task_id}, window {args.hours}h")
        return 0

    decision = decide(store, challenges=args.challenges, stall_after_s=args.stall_after)
    if args.json:
        print(json.dumps(decision.to_record(), indent=2, sort_keys=True))
    else:
        suffix = f" [{decision.round_id}]" if decision.round_id else ""
        print(f"{decision.action}{suffix}: {decision.reason}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"action={decision.action}\nround_id={decision.round_id}\n")
    # A distinct code for the two outcomes a human has to look at, so a scheduled run can alert on
    # them without parsing text. WAIT and the ordinary actions are success: nothing is wrong.
    if decision.action in (STALLED, BLOCKED):
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

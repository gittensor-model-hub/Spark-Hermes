"""Open a round over a challenge packet, and advance it. The driver nothing else provided.

    python -m validator.round_loop open  --challenge datasets/challenges/tc-log-rotation-order.json \\
                                         --round r-001 --hours 24
    python -m validator.round_loop list
    python -m validator.round_loop freeze --round r-001
    python -m validator.round_loop show   --round r-001

Every part of this existed and none of it was connected. `hermes.challenge` writes packets,
`hermes.round.open_round` builds a window over one, `validator.store` persists it and
`validator.api` serves it -- and no code path called `open_round` outside its tests, so a live
validator had nothing to serve and there was no way to give it anything.

## Why the packet on disk is not enough to rebuild a challenge

`Challenge.to_record()` is deliberately lossy: it publishes the baseline as aggregate statistics
and strips every task key outside `PUBLISHABLE_TASK_KEYS`. A window built from it would have an
empty attempt list, so `token_spread` would be zero -- and a zero spread does not read as
"missing data", it reads as a perfectly repeatable task, which is the strongest possible evidence
a miner's token margin is real. The efficiency gate would then accept margins it should refuse.

So `open` needs the private snapshot as well: `--episodes` re-derives the baseline from the same
log the packet was opened from. Refused rather than guessed at, because the alternative failure is
silent and lands on the acceptance gate.

## The clock

Deadlines are wall-clock seconds, taken from `time.time()` at open. `hermes.round` compares
submissions against `deadline` using whatever `now` the caller passes, so the units only have to
agree with themselves -- but they have to agree with what the *gate* sees, and a rollout PR's
timestamps are wall clock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from hermes.challenge import CHALLENGES_DIR, Challenge, ChallengeError, from_episode_log
from hermes.round import RoundError, open_round
from validator.store import RoundStore, StoreError


class LoopError(RoundError):
    """A round cannot be opened or advanced from what was supplied."""


def challenge_from_packet_and_log(packet_path: Path, episodes_path: Path) -> Challenge:
    """Rebuild the full challenge for a published packet, using the log it was opened from.

    The packet supplies identity and the withheld commitment; the log supplies the attempts. Both
    are needed and neither is sufficient: without the log there is no spread, and without the
    packet there is no commitment to say which withheld check will grade the round.
    """
    from hermesbench.sink import read_episodes

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    task_id = str(packet.get("task_id") or "")
    if not task_id:
        raise LoopError(f"{packet_path} has no task_id; it is not a challenge packet")

    rows = [r for r in read_episodes(episodes_path)]
    opened, refused = from_episode_log(
        rows,
        epoch=dict(packet.get("epoch") or {}),
        task_pins={task_id: dict(packet.get("task") or {})},
    )
    for challenge in opened:
        if challenge.task_id == task_id:
            return challenge

    why = dict(refused).get(task_id, "it is not in the log at all")
    raise LoopError(
        f"{episodes_path} cannot reopen {task_id}: {why}. The packet was opened from some log; a "
        "round must be built from that same evidence, because the baseline's attempts are what "
        "every later noise judgement is computed from."
    )


def open_from_packet(
    *,
    packet_path: Path,
    episodes_path: Path,
    round_id: str,
    hours: float,
    store: RoundStore | None = None,
    now: float | None = None,
    assignment: Any = None,
) -> Any:
    """Open a round over one packet and persist it. Refuses to overwrite a stored round."""
    store = store or RoundStore()
    if store.path_for(round_id).exists():
        raise LoopError(
            f"round {round_id!r} is already stored. Reopening it would reset a window miners may "
            "already have submitted to, and their receipts would be gone with it."
        )
    challenge = challenge_from_packet_and_log(packet_path, episodes_path)
    opened_at = time.time() if now is None else now
    window = open_round(
        challenge,
        round_id=round_id,
        opened_at=opened_at,
        deadline=opened_at + hours * 3600.0,
        assignment=assignment,
    )
    store.save(window)
    return window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["open", "list", "freeze", "show"])
    parser.add_argument("--round", dest="round_id", default="", help="round id")
    parser.add_argument("--challenge", type=Path, default=None, help="a packet in datasets/challenges/")
    parser.add_argument("--episodes", type=Path, default=None, help="the episode log the packet was opened from")
    parser.add_argument("--hours", type=float, default=24.0, help="how long the window stays open")
    parser.add_argument("--store", type=Path, default=None, help="override var/rounds")
    parser.add_argument(
        "--reason",
        default="",
        help="why a round is being frozen before its announced deadline; `hermes.round` refuses an "
        "early freeze without one, because a shortened round that looks like a normal one is "
        "indistinguishable afterwards from one that ran its full length",
    )
    args = parser.parse_args(argv)

    try:
        store = RoundStore(args.store)
    except StoreError as exc:
        print(f"validator.round_loop: {exc}", file=sys.stderr)
        return 2

    try:
        if args.action == "open":
            if not args.round_id or args.challenge is None or args.episodes is None:
                print(
                    "validator.round_loop: open needs --round, --challenge and --episodes. The log is "
                    "not optional: a packet alone rebuilds a baseline with no attempts, whose spread "
                    "is zero, and a zero spread reads as a perfectly repeatable task rather than as "
                    "missing data -- so the efficiency gate would accept margins it should refuse.",
                    file=sys.stderr,
                )
                return 2
            window = open_from_packet(
                packet_path=args.challenge,
                episodes_path=args.episodes,
                round_id=args.round_id,
                hours=args.hours,
                store=store,
            )
            print(f"opened {window.round_id} over {window.task_id}")
            print(f"  baseline    {window.challenge.baseline.passes}/{len(window.challenge.baseline.attempts)} passes")
            print(f"  spread      {window.challenge.baseline.token_spread:.1%}")
            print(f"  deadline    {window.deadline:.0f} ({args.hours}h)")
            print(f"  scope       {'enforced' if window.scope_enforced else 'UNSCOPED (no assignment attached)'}")
            print(f"  stored      {store.path_for(window.round_id)}")
        elif args.action == "list":
            loaded, failed = store.load_all()
            if not loaded and not failed:
                print("no rounds stored")
            for round_id, window in sorted(loaded.items()):
                # Read the count off the ledger rather than `window.verdicts`, which raises while
                # a round is OPEN -- correctly. The first version of this line called it anyway
                # and every listing died on the guard's own message: "a caller expecting an empty
                # dict here is a caller about to publish one". A count is not a score, so the
                # ledger publishes `verdicts_recorded` for exactly this purpose.
                ledger = window.to_record()
                graded = ledger.get("verdicts")
                recorded = len(graded) if graded is not None else ledger.get("verdicts_recorded", 0)
                print(
                    f"  {round_id:<12} {window.state:<8} {window.task_id:<30} "
                    f"subs={len(window.submissions)} verdicts={recorded}"
                )
            for round_id, why in failed:
                print(f"  {round_id:<12} UNREADABLE  {why}", file=sys.stderr)
        elif args.action == "freeze":
            window = store.load(args.round_id)
            frozen = window.freeze(now=time.time(), reason=args.reason)
            store.save(window)
            print(f"froze {args.round_id} at {frozen.frozen_at:.0f}; sealed {frozen.sealed_submissions} submission(s)")
            print(f"  seal {frozen.seal_digest}")
        else:
            window = store.load(args.round_id)
            print(json.dumps(window.public_view(), indent=2, sort_keys=True))
    except (LoopError, StoreError, RoundError, ChallengeError) as exc:
        print(f"validator.round_loop: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"validator.round_loop: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = ["CHALLENGES_DIR", "LoopError", "challenge_from_packet_and_log", "main", "open_from_packet"]


if __name__ == "__main__":
    raise SystemExit(main())

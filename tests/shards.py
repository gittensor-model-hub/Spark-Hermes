"""Which tests run on which CI runner, defined once so the shards cannot disagree.

The suite is ~4000 tests and 51 minutes on a single runner. `pytest-xdist` cannot shorten that:
`admin.runtime_protocol.campaign_lock` is one global file lock, so in-process workers would
serialise on it or deadlock against it -- which is not theoretical here, an orphaned `cycle
resume` once held that lock for 1h20m and blocked everything behind it. Separate RUNNERS have
separate filesystems and therefore separate locks, so the split is at the job level.

## The shards come from measurement, not from intuition

`pytest --durations` over the whole suite: **25 tests of 4003 accounted for 50% of the runtime.**
`tests/test_cycles.py` alone was 905s; runtime-transition and release together ~500s. Those tests
spawn real subprocesses -- installed interpreters built into pytest tmpdirs, supervised
train/merge jobs, crash-and-reconcile fixtures -- and one single 147s cost is fixture *setup*
rather than any assertion. None of that is tuneable without weakening what the tests check, so it
is isolated instead: the remaining ~2700 tests should not queue behind it.

## Why this is a module and not three hard-coded pytest invocations

Three shards that each silently run a subset still report three greens, and three greens then
mean less than the one job they replaced. The failure is invisible in exactly the way a split
invites: add a test file, forget the ignore list, and it runs nowhere.

So the shard membership is computed from one list, and `verify` asserts the partition against
pytest's own collection -- every collected test in exactly one shard, none dropped, none run
twice. `verify` runs as its own CI job, so the guarantee is checked on every pull request rather
than assumed to still hold.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent

# Measured cost, highest first. Membership is by FILE because pytest ignores at file granularity
# and because these files are expensive as a whole, not in a few nameable tests.
SHARDS: dict[str, tuple[str, ...]] = {
    # MEASURED, not estimated. `tests/shards.py run <shard>` timed on one host:
    #
    #   campaign  32 tests   1445s   release  259 tests  1426s   core  3716 tests  ~283s
    #
    # The distribution is the point: 291 tests are ~48 minutes and the other 3716 are ~5. An
    # earlier version of this file guessed those two at 1075s and 500s from a --durations top-25
    # sample, and both were badly low -- the sample showed the worst individual tests, not the
    # totals. Re-measure before rebalancing; do not extrapolate from --durations again.
    #
    # test_cycles.py alone is the floor. It is ~1200s of the campaign figure, so no arrangement
    # of the others gets wall time below it, and splitting it is the only thing that would.
    "cycles": ("test_cycles.py",),
    # ~725s: the demo driver plus the cutover fixture that builds a retained installation.
    "campaign": (
        "test_cycle_demo.py",
        "test_runtime_transition.py",
    ),
    # ~950s: the release and crossed-evaluation gates.
    "release": (
        "test_crossed_release.py",
        "test_release_integrity.py",
        "test_release_catalog_versions.py",
        "test_release_execution.py",
        "test_runtime_protocol.py",
    ),
}

# Everything not named above. Defined as the complement rather than a list, because a list would
# need editing every time a test file is added -- and forgetting is silent.
CORE = "core"

__all__ = ["CORE", "SHARDS", "ShardError", "collect", "pytest_args", "verify"]


class ShardError(RuntimeError):
    """The shards do not partition the suite."""


def _missing() -> list[str]:
    return [name for names in SHARDS.values() for name in names if not (TESTS / name).is_file()]


def pytest_args(shard: str) -> list[str]:
    """The pytest arguments for one shard.

    A named shard runs exactly its files. `core` runs everything else by ignoring them, so a new
    test file lands in `core` automatically instead of being dropped.
    """
    if shard == CORE:
        args = ["tests"]
        for names in SHARDS.values():
            args += [f"--ignore=tests/{name}" for name in names]
        return args
    if shard not in SHARDS:
        raise ShardError(f"unknown shard {shard!r}; known: {[*SHARDS, CORE]}")
    return [f"tests/{name}" for name in SHARDS[shard]]


def collect(args: list[str]) -> set[str]:
    """Test node ids pytest would run for these arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "--no-header", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ShardError(f"collection failed for {args}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


def verify() -> int:
    """Assert the shards partition the suite exactly. Returns a process exit code."""
    if absent := _missing():
        print(f"shard names a file that does not exist: {absent}", file=sys.stderr)
        return 1

    whole = collect(["tests"])
    seen: dict[str, str] = {}
    problems: list[str] = []
    for shard in (*SHARDS, CORE):
        for node in collect(pytest_args(shard)):
            if node in seen:
                problems.append(f"{node} runs in both {seen[node]} and {shard}")
            seen[node] = shard

    if dropped := sorted(whole - set(seen)):
        problems.append(f"{len(dropped)} test(s) run in NO shard, e.g. {dropped[:5]}")
    if extra := sorted(set(seen) - whole):
        problems.append(f"{len(extra)} test(s) collected by a shard but not by the suite, e.g. {extra[:5]}")

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    counts = {shard: sum(1 for s in seen.values() if s == shard) for shard in (*SHARDS, CORE)}
    print(f"shards partition {len(whole)} collected tests exactly: {counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one shard")
    run.add_argument("shard", choices=[*SHARDS, CORE])
    sub.add_parser("verify", help="assert the shards partition the suite")
    args = parser.parse_args(argv)

    if args.command == "verify":
        return verify()
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *pytest_args(args.shard)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())

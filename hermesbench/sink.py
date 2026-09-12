"""Incremental persistence for a bench run: one JSON object per episode, as it finishes.

`run_suite` accumulates every result in memory and the CLI writes nothing until the last
episode has returned, so a run that dies at episode 189 of 190 produces *nothing* -- not a
partial score, not even a list of which tasks got as far as running. This was measured on a
real baseline: 190 episodes across 8 shards, and all eight shard logs sat at 0 bytes until
their shard finished. Hours of paid inference bought a traceback.

So an episode's metrics are appended the moment that episode completes. A `kill -9` -- the
OOM killer, a preempted spot instance, a CI step timeout -- then loses at most the episode
that was in flight, because every earlier line is already in the kernel's hands. `flush()`
and not `fsync()` on purpose: bytes handed to the kernel survive the death of the process,
which is the failure this exists for, and an fsync per episode buys only power-loss
durability while making the log the slowest thing in a run.

**JSONL, because the reader has to cope with a file whose last line is torn in half.** A
single JSON array is unreadable until its closing bracket arrives, which is precisely the
property that made the old behavior useless -- the file existed and told you nothing.
`read_episode_prefix` observes completed records in a live or interrupted log.
`read_episodes` requires a complete snapshot for authoritative consumers and refuses a torn tail. A corrupt line in the *middle* is not a crash:
it is two runs interleaved into one path, or a damaged disk, and skipping it silently would
turn missing episodes into a lower success rate with no trace of why.

Each line carries the same `EpisodeMetrics.to_record()` the finished run already prints
under `per_episode`, so a partial log is a **prefix of the final report** rather than a
second format for consumers to learn. Order is run order, so the k-th line for a given
`task_id` is the attempt that ran in workspace `task_id` (k=0) or `task_id#k` -- enough to
walk from a line in the log to the directory it left behind. The trajectory is deliberately
not written: it is orders of magnitude larger than the metrics, nothing in the final output
carries it either, and paying that per line would make the sink expensive enough that
someone would turn it off.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from hermes.evidence_json import evidence_object


class SinkError(RuntimeError):
    """Refused to write, or to read back, an episode log whose contents cannot be trusted."""


class EpisodeRecord(Protocol):
    """The parts of `runner.EpisodeResult` a sink reads.

    Structural rather than an import of `EpisodeResult`: `hermesbench.runner` imports this
    module to wire its flag, and importing the runner back would be a cycle.
    """

    task_id: str
    setup_failed: bool
    metrics: Any
    integrity: Any


class EpisodeSink(Protocol):
    """Where `run_suite` hands an episode the instant it finishes.

    A protocol, not the concrete class, so a caller that already has somewhere to put
    episodes -- a shard aggregator, a test double -- does not have to route them through a
    file to get them out of `run_suite` one at a time.
    """

    def append(self, result: EpisodeRecord) -> None: ...


class JsonlEpisodeSink:
    """Appends one flushed line per episode to a JSONL file.

    Holds an open handle and a counter, and nothing else. Keeping the records here too
    would double the memory a suite already spends on trajectories, and a sink that makes a
    long run likelier to be OOM-killed is a sink people switch off -- which is the state
    this module exists to end.
    """

    def __init__(self, path: Path, *, keep_trajectories: bool = False) -> None:
        self.path = Path(path)
        # See `append`: off by default because a trajectory is the whole conversation, and a run
        # that asked for counts should not silently start writing transcripts.
        self.keep_trajectories = bool(keep_trajectories)
        if self.path.exists() and self.path.stat().st_size > 0:
            # Refused rather than appended or truncated. Appending would interleave two
            # runs into one file with nothing in the lines to separate them, and a reader
            # would report the pair as one suite; truncating would destroy the partial log
            # that is the entire reason this flag exists.
            #
            # This catches the sequential case -- a rerun aimed at the last run's log --
            # and not two processes that open the same *empty* path in the same moment.
            # Parallel shards must be given distinct paths; nothing here can detect two
            # live writers, since their lines would each be individually well-formed.
            raise SinkError(
                f"{self.path} already holds {self.path.stat().st_size} bytes of episodes. "
                "Two runs in one log cannot be told apart by a reader, and truncating would "
                "destroy the partial log this flag exists to preserve. Point it at a fresh "
                "path, or move the old one aside."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.episodes_written = 0

    def append(self, result: EpisodeRecord) -> None:
        record = {
            "episode": self.episodes_written,
            "task_id": result.task_id,
            "setup_failed": result.setup_failed,
            # Not in `EpisodeMetrics`: `success` already accounts for a disqualification,
            # but "failed" and "cheated and was caught" are different post-mortems and a
            # crashed run's log is read precisely to tell them apart.
            "disqualified": result.integrity.disqualified,
            "metrics": result.metrics.to_record(),
            "integrity": (result.integrity.to_record() if hasattr(result.integrity, "to_record") else None),
            "evidence": getattr(result, "evidence", {}),
        }
        verification = getattr(result, "verification", None)
        if verification is not None:
            record["verification"] = verification.to_record()
        # The trajectory, when the caller asked for it.
        #
        # Off by default and not by accident. A trajectory is the whole conversation -- every tool
        # output the agent read -- so it is orders of magnitude larger than the metrics beside it
        # and it carries whatever the workspace held. A run that wanted counts should not silently
        # start writing transcripts.
        #
        # But without it nothing downstream can exist. `hermes.format.to_messages_record` builds
        # training rows *from* a trajectory, and this sink threw the trajectory away -- so an
        # accepted result could be scored and crowned and never turned into a single SFT row. The
        # data was in memory at the moment it was dropped.
        if self.keep_trajectories:
            trajectory = getattr(result, "trajectory", None)
            serialise = getattr(trajectory, "to_record", None)
            # `None` when the result carried no trajectory, which is different from an episode that
            # produced an empty one -- the reader counts those apart, so they must not collapse here.
            record["trajectory"] = serialise() if callable(serialise) else None
        # One write of one complete line, then flush. The newline must never reach the file
        # ahead of the payload it terminates: a reader splitting on newlines would then see
        # a truncated record as a complete one, and `read_episodes`' whole tolerance rests
        # on "no trailing newline" meaning "this is the episode the run died inside".
        # ensure_ascii also keeps any newline inside the record escaped, so one line stays
        # one episode even when a field carries a shell transcript.
        self._handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        self._handle.flush()
        self.episodes_written += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlEpisodeSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _decode_line(raw: bytes, *, source: str, number: int) -> dict[str, Any]:
    try:
        return evidence_object(raw)
    except (ValueError, RecursionError) as exc:
        raise SinkError(f"{source}: line {number} is not valid complete episode evidence ({exc})") from exc


def decode_episodes(raw: bytes, *, source: str = "episode snapshot") -> list[dict[str, Any]]:
    """Decode the exact complete byte snapshot, without yielding a partial schedule."""
    if raw and not raw.endswith(b"\n"):
        raise SinkError(f"{source}: truncated episode log (unfinished final record)")
    return [
        _decode_line(line, source=source, number=number)
        for number, line in enumerate(raw.split(b"\n")[:-1], start=1)
        if line.strip()
    ]


def read_episodes(path: Path) -> Iterator[dict[str, Any]]:
    """Read complete authoritative evidence; malformed or unfinished records refuse.

    Materialize and check one byte snapshot before exposing any rows. For progress
    observation of an unfinished run, explicitly use `read_episode_prefix` instead.
    """
    path = Path(path)
    return iter(decode_episodes(path.read_bytes(), source=str(path)))


def read_episode_prefix(path: Path) -> Iterator[dict[str, Any]]:
    """Observe only a live log's finished prefix; this is not complete run evidence."""
    path = Path(path)
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                return
            if raw.strip():
                yield _decode_line(raw, source=str(path), number=number)

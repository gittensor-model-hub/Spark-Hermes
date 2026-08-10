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
`read_episodes` drops an unterminated final line, the one the writer died inside, and
refuses anything else that fails to parse. A corrupt line in the *middle* is not a crash:
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

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
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
        }
        # One write of one complete line, then flush. The newline must never reach the file
        # ahead of the payload it terminates: a reader splitting on newlines would then see
        # a truncated record as a complete one, and `read_episodes`' whole tolerance rests
        # on "no trailing newline" meaning "this is the episode the run died inside".
        # ensure_ascii also keeps any newline inside the record escaped, so one line stays
        # one episode even when a field carries a shell transcript.
        self._handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._handle.flush()
        self.episodes_written += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlEpisodeSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_episodes(path: Path) -> Iterator[dict[str, Any]]:
    """Yield a log's episodes, tolerating a torn final line and nothing else.

    A generator, so a consumer asking "how far did the run get" over a long log does not
    have to hold the whole thing -- the sink refuses to buffer a run, and a reader that
    buffers it instead would have moved the problem rather than solved it.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n"):
                # Only a file's final line can be missing its terminator, so this is the
                # episode the writer was killed inside. Dropping it is the point: half a
                # JSON object is not an episode, and raising here would leave a crashed
                # run's log exactly as useful as the nothing it used to produce.
                return
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SinkError(
                    f"{path}: line {number} is complete but does not parse ({exc}). That is "
                    "corruption or two runs written to one path, not an interrupted write -- "
                    "skipping it would drop episodes silently and move the reported score."
                ) from exc

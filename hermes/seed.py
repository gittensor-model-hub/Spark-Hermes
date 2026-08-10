"""Validator seeding: deciding who works on what, so two miners do not do one task twice.

`Round` here is the ASSIGNMENT axis: its states -- committed, open, closed -- describe when the
seed is visible and therefore when the assignment is computable. The submission window is a
different thing and lives in `hermes.round.RoundWindow`, whose states describe what the
validator will accept and publish. Both were once called `Round`; see that module for why the
distinction matters and what still is not wired between them.

A network of miners generating rollouts has an obvious failure that costs nothing to
create and everything to detect later: two miners pick the same task, run the same three
teachers, and submit two datasets whose rows are near-duplicates. The corpus looks twice
as large and teaches the same thing once. `hermesbench.identity` catches identical rows
after the fact; this stops the work from being duplicated in the first place.

The mechanism is deterministic assignment from a published seed. A validator announces a
round -- a seed, the task pool, and the registered miners -- and every participant can
compute the same assignment independently. Nobody has to be told what to work on, and
nobody has to trust the telling.

Three properties this has to have, and each one rules out a simpler design:

**Verifiable without the validator.** A miner submitting work for a task it was not
assigned should be rejectable by anyone holding the round announcement, not only by the
validator that issued it. So assignment is a pure function of `(seed, round, task_id,
miners)` and the check is recomputation, not a lookup in someone's database.

**Stable under a miner leaving.** If assignment were "hash the task id modulo the miner
count", one miner dropping out would reshuffle every task in the round and invalidate work
already in progress. Rendezvous hashing gives each task to its highest-scoring miner, so
losing a miner only reassigns that miner's own tasks.

**Unpredictable before the round opens.** A seed a miner can guess in advance is a seed a
miner can grind against -- registering identities until it is assigned the tasks it has
already solved. The seed is a commitment: the validator publishes its digest first and the
value later, so the assignment is fixed before anyone knows what it is.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1

# A round's lifecycle. `committed` means the seed exists and its digest is published but
# the value is not; `open` means the value is out and the assignment is computable.
COMMITTED = "committed"
OPEN = "open"
CLOSED = "closed"

STATES = (COMMITTED, OPEN, CLOSED)


class SeedError(ValueError):
    """A round or an assignment is malformed."""


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def new_seed() -> str:
    """A fresh 256-bit seed, hex encoded."""
    return secrets.token_hex(32)


def seed_commitment(seed: str) -> str:
    """The digest a validator publishes before revealing the seed.

    Publishing the commitment first is what stops the validator choosing a seed *after*
    seeing who registered -- which would let it hand a favoured miner the easy tasks and
    leave no trace, because any seed looks as random as any other.
    """
    if len(seed) < 32:
        raise SeedError("a seed needs at least 128 bits of entropy to be worth committing to")
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def score(seed: str, round_id: str, task_id: str, miner_id: str) -> str:
    """Rendezvous score for one (task, miner) pair. Highest wins.

    Hex rather than an integer so ordering is lexicographic and identical in every
    language a miner might reimplement this in. A validator and a miner disagreeing about
    integer width would produce different assignments from the same seed, and the
    disagreement would look like cheating.
    """
    return _digest(seed, round_id, task_id, miner_id)


@dataclass(frozen=True)
class Assignment:
    """One task, and the miner that owes work on it."""

    task_id: str
    miner_id: str
    replica: int = 0

    def to_record(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "miner_id": self.miner_id, "replica": self.replica}


@dataclass(frozen=True)
class Round:
    """A published assignment round.

    `replicas` is how many *different* miners each task goes to. One is the efficient
    setting and zero is not an option; more than one is deliberate duplication, which is how
    a fabricated result gets caught: two independent runs of a deterministic verifier must
    agree.

    It catches *independent* fabrication only. The seed is public, so every miner can
    compute who else drew a task and therefore knows when it is being cross-checked -- and
    two miners who collude can agree on the same lie. Duplication raises the cost of
    fabricating alone; it does not make collusion detectable.
    """

    round_id: str
    seed: str
    task_ids: tuple[str, ...]
    miner_ids: tuple[str, ...]
    replicas: int = 1
    state: str = OPEN

    def __post_init__(self) -> None:
        if not self.round_id:
            raise SeedError("a round needs an id")
        # Shape-checked on construction, not only when publishing. Otherwise a round rebuilt
        # from a record could carry a seed too weak to commit to, and every read path --
        # assignees, owns, the gate's scope check -- would use it happily.
        seed_commitment(self.seed)
        if self.state not in STATES:
            raise SeedError(f"unknown round state {self.state!r}; expected one of {list(STATES)}")
        if not self.task_ids:
            raise SeedError(f"{self.round_id}: a round with no tasks assigns nothing")
        if not self.miner_ids:
            raise SeedError(f"{self.round_id}: a round with no miners has nobody to assign to")
        for name, values in (("task", self.task_ids), ("miner", self.miner_ids)):
            if len(set(values)) != len(values):
                raise SeedError(f"{self.round_id}: duplicate {name} id; the assignment would be ambiguous")
        if self.replicas < 1:
            raise SeedError("a task assigned to nobody is not in the round")
        if self.replicas > len(self.miner_ids):
            raise SeedError(
                f"{self.round_id}: {self.replicas} replicas requested but only {len(self.miner_ids)} miners; "
                "a task cannot go to more distinct miners than exist"
            )

    @property
    def commitment(self) -> str:
        """A commitment to the whole announcement, not only the seed.

        Committing to the seed alone leaves the task pool and the miner list free to change
        between commitment and reveal -- and those decide the assignment just as much as the
        seed does. A validator could publish a digest, watch who registers, then add or drop
        a task and still match its own commitment.
        """
        body = json.dumps(
            {
                "round_id": self.round_id,
                "seed": self.seed,
                "task_ids": sorted(self.task_ids),
                "miner_ids": sorted(self.miner_ids),
                "replicas": self.replicas,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()

    def assignees(self, task_id: str) -> tuple[str, ...]:
        """The miners this task belongs to, best first.

        Rendezvous: every miner scores the task and the top `replicas` win. A miner
        leaving only reassigns the tasks it held, because every other task's winner is
        unchanged -- which is the property a modulo scheme does not have.
        """
        if task_id not in self.task_ids:
            raise SeedError(f"{task_id!r} is not in round {self.round_id}")
        ranked = sorted(self.miner_ids, key=lambda m: score(self.seed, self.round_id, task_id, m), reverse=True)
        return tuple(ranked[: self.replicas])

    def assigned_to(self, miner_id: str) -> tuple[str, ...]:
        """Every task this miner owes work on."""
        if miner_id not in self.miner_ids:
            raise SeedError(f"{miner_id!r} is not registered in round {self.round_id}")
        return tuple(t for t in self.task_ids if miner_id in self.assignees(t))

    def owns(self, miner_id: str, task_id: str) -> bool:
        """Whether a submission for this task from this miner is in scope.

        The check anyone can run. A validator is not required: the round announcement is
        enough, which is what keeps a rejected submission arguable rather than arbitrary.
        """
        return miner_id in self.assignees(task_id)

    def assignments(self) -> tuple[Assignment, ...]:
        return tuple(
            Assignment(task_id=t, miner_id=m, replica=i) for t in self.task_ids for i, m in enumerate(self.assignees(t))
        )

    def to_record(self, *, reveal_seed: bool = True) -> dict[str, Any]:
        """The published announcement.

        With `reveal_seed=False` this is the *commitment* announcement: it fixes the round,
        the pool and the participants without letting anyone compute the assignment yet.
        """
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "round_id": self.round_id,
            "state": self.state,
            "commitment": self.commitment,
            "task_ids": list(self.task_ids),
            "miner_ids": list(self.miner_ids),
            "replicas": self.replicas,
        }
        if reveal_seed:
            record["seed"] = self.seed
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Round:
        seed = str(record.get("seed") or "")
        if not seed:
            raise SeedError("cannot rebuild a round from a commitment announcement; the seed is not revealed yet")
        instance = cls(
            round_id=str(record.get("round_id") or ""),
            seed=seed,
            task_ids=tuple(str(t) for t in record.get("task_ids") or ()),
            miner_ids=tuple(str(m) for m in record.get("miner_ids") or ()),
            replicas=int(record.get("replicas", 1)),
            state=str(record.get("state", OPEN)),
        )
        published = record.get("commitment")
        if not published:
            # An announcement with no commitment fixes nothing: the assignment could have
            # been chosen after seeing who registered, and nothing would show it.
            raise SeedError(f"{instance.round_id}: announcement carries no commitment; the round was never fixed")
        if published != instance.commitment:
            # The seed does not hash to the digest that was published before the round
            # opened, which means the seed was chosen or swapped after the fact.
            raise SeedError(
                f"{instance.round_id}: revealed seed does not match the published commitment; "
                "the assignment was not fixed before the round opened"
            )
        return instance


def open_round(round_id: str, task_ids: list[str], miner_ids: list[str], *, replicas: int = 1, seed: str = "") -> Round:
    """Open a round, generating a seed if one was not supplied."""
    return Round(
        round_id=round_id,
        seed=seed or new_seed(),
        task_ids=tuple(task_ids),
        miner_ids=tuple(miner_ids),
        replicas=replicas,
        state=OPEN,
    )


@dataclass(frozen=True)
class Duplication:
    """Where two miners were asked for the same task on purpose."""

    task_id: str
    miners: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "miners": list(self.miners)}


def duplicated(round_: Round) -> tuple[Duplication, ...]:
    """Tasks deliberately given to more than one miner.

    Reported so a validator knows which submissions it can *cross-check* rather than
    merely verify. A deterministic verifier run twice on the same task must reach the same
    verdict, so a disagreement between two replicas is evidence about the miners rather
    than about the task.
    """
    if round_.replicas < 2:
        return ()
    return tuple(Duplication(task_id=t, miners=round_.assignees(t)) for t in round_.task_ids)


def coverage(round_: Round) -> dict[str, int]:
    """How many tasks each miner drew.

    Rendezvous hashing balances in expectation, not exactly, and a round where one miner
    drew most of the pool is worth seeing before the work starts rather than after.
    """
    counts = dict.fromkeys(round_.miner_ids, 0)
    for assignment in round_.assignments():
        counts[assignment.miner_id] += 1
    return counts


def verify_submission_scope(round_record: dict[str, Any], *, miner_id: str, task_ids: list[str]) -> list[str]:
    """Which of a miner's submitted tasks it was not assigned. Empty means in scope.

    Returns the out-of-scope ids rather than raising, because a submission that is partly
    out of scope is a partial acceptance rather than a rejection, and the reviewer needs
    to see which rows to drop.
    """
    round_ = Round.from_record(round_record)
    return [t for t in task_ids if t not in round_.task_ids or not round_.owns(miner_id, t)]

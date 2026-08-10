"""Multi-session tasks: measuring whether an agent carries anything forward.

Every task so far has been one episode. That cannot express the capability a Hermes
worker is actually judged on across sessions -- *"optimize kernel A"* on Monday and
*"continue the optimization"* on Tuesday -- and so memory and skill reuse have been
unmeasurable rather than merely unmeasured.

The mechanism is a task with several `sessions`. The workspace and a memory store persist
between them; only the prompt changes.

**The control is the whole design.** An agent doing well in session 2 proves nothing on
its own: the second objective might simply be easy, or already satisfied by side effects
of the first. So every multi-session task is run twice --

    warm:  session 1, then session 2 with everything session 1 left behind
    cold:  session 2 alone, from a fresh workspace and an empty memory

-- and the score is the *difference*. That difference is the only part attributable to
carrying something forward. Without the cold arm, a memory benchmark measures task
difficulty and calls it recall.

A negative carryover is a real and interesting result, not a bug: it means the agent was
*worse* for having the earlier session's context, which is what context corruption looks
like from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# What a session is allowed to see from earlier ones.
CARRY_WORKSPACE = "workspace"
CARRY_MEMORY = "memory"
CARRY_NOTHING = "nothing"


class SessionError(ValueError):
    """A multi-session specification is malformed."""


@dataclass(frozen=True)
class SessionSpec:
    """One session of a multi-session task.

    `verify` is per-session so each leg has its own objective; the task's own `verify`
    still decides the overall outcome. A session without a check contributes nothing
    measurable, which is why one is required.
    """

    session_id: str
    prompt: str
    verify: str
    max_steps: int = 30

    def __post_init__(self) -> None:
        if not self.session_id or not self.prompt.strip():
            raise SessionError("a session needs an id and a prompt")
        if not self.verify.strip():
            raise SessionError(f"session {self.session_id!r} has no verify; it would measure nothing")

    def to_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "verify": self.verify,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], *, origin: str = "<memory>") -> SessionSpec:
        missing = [k for k in ("session_id", "prompt", "verify") if not record.get(k)]
        if missing:
            raise SessionError(f"{origin}: session is missing {', '.join(missing)}")
        return cls(
            session_id=str(record["session_id"]),
            prompt=str(record["prompt"]),
            verify=str(record["verify"]),
            max_steps=int(record.get("max_steps", 30)),
        )


@dataclass
class MemoryStore:
    """What an agent chose to persist between sessions.

    Deliberately dumb -- an append-only list of notes with a search. The interesting
    behaviour is not the store, it is *when the agent writes to it and whether it reads
    before starting over*, and a cleverer store would do that judgement for the model and
    hide the thing being measured.
    """

    notes: list[str] = field(default_factory=list)

    def write(self, note: str) -> None:
        text = note.strip()
        if text:
            self.notes.append(text)

    def search(self, query: str) -> list[str]:
        """Substring match on whitespace-separated terms. Any term hits."""
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        return [n for n in self.notes if any(t in n.lower() for t in terms)]

    def clear(self) -> None:
        self.notes.clear()

    @property
    def empty(self) -> bool:
        return not self.notes

    def to_record(self) -> dict[str, Any]:
        return {"notes": list(self.notes), "count": len(self.notes)}


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    passed: bool
    steps: int
    memory_reads: int = 0
    memory_writes: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "passed": self.passed,
            "steps": self.steps,
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
        }


@dataclass(frozen=True)
class CarryoverReport:
    """Warm run against cold control, for the final session.

    `steps_saved` is positive when the warm run reached the same objective in fewer
    steps. That is the skill-reuse signal -- twenty steps becoming three -- and it is
    only meaningful alongside `carried`, since finishing faster while failing is not an
    improvement.
    """

    task_id: str
    warm_passed: bool
    cold_passed: bool
    warm_steps: int
    cold_steps: int
    memory_reads: int = 0

    @property
    def carried(self) -> bool:
        """Succeeded with the earlier session's context and failed without it."""
        return self.warm_passed and not self.cold_passed

    @property
    def regressed(self) -> bool:
        """Failed *because of* the earlier context -- context corruption, from outside."""
        return self.cold_passed and not self.warm_passed

    @property
    def steps_saved(self) -> int:
        if not (self.warm_passed and self.cold_passed):
            return 0
        return self.cold_steps - self.warm_steps

    @property
    def inconclusive(self) -> bool:
        """Both arms did the same thing, so the task separates nothing here.

        Reported rather than scored as a pass: a task both arms solve measures ease, and
        a task neither solves measures nothing at all.
        """
        return self.warm_passed == self.cold_passed and self.steps_saved == 0

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "warm_passed": self.warm_passed,
            "cold_passed": self.cold_passed,
            "warm_steps": self.warm_steps,
            "cold_steps": self.cold_steps,
            "carried": self.carried,
            "regressed": self.regressed,
            "steps_saved": self.steps_saved,
            "inconclusive": self.inconclusive,
            "memory_reads": self.memory_reads,
        }


@dataclass(frozen=True)
class MemoryMetrics:
    """Aggregate carryover across a suite of multi-session tasks."""

    tasks: int
    carried: int
    regressed: int
    inconclusive: int
    mean_steps_saved: float

    @property
    def carryover_rate(self) -> float:
        """Share of *conclusive* tasks where the earlier session helped.

        Inconclusive tasks are excluded from the denominator rather than counted as
        failures: a task both arms solve says nothing about memory, and letting it drag
        the rate down would make an easy suite look like a forgetful agent.
        """
        conclusive = self.tasks - self.inconclusive
        return self.carried / conclusive if conclusive else 0.0

    @property
    def regression_rate(self) -> float:
        conclusive = self.tasks - self.inconclusive
        return self.regressed / conclusive if conclusive else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "carried": self.carried,
            "regressed": self.regressed,
            "inconclusive": self.inconclusive,
            "conclusive": self.tasks - self.inconclusive,
            "carryover_rate": round(self.carryover_rate, 4),
            "regression_rate": round(self.regression_rate, 4),
            "mean_steps_saved": round(self.mean_steps_saved, 2),
        }


def summarize(reports: list[CarryoverReport]) -> MemoryMetrics:
    if not reports:
        return MemoryMetrics(tasks=0, carried=0, regressed=0, inconclusive=0, mean_steps_saved=0.0)
    saved = [r.steps_saved for r in reports if r.steps_saved]
    return MemoryMetrics(
        tasks=len(reports),
        carried=sum(1 for r in reports if r.carried),
        regressed=sum(1 for r in reports if r.regressed),
        inconclusive=sum(1 for r in reports if r.inconclusive),
        mean_steps_saved=(sum(saved) / len(saved)) if saved else 0.0,
    )


def carryover(
    task_id: str,
    warm: SessionResult,
    cold: SessionResult,
    *,
    memory_reads: int = 0,
) -> CarryoverReport:
    """Compare the warm final session against its cold control."""
    if warm.session_id != cold.session_id:
        # Comparing different objectives would produce a number that looks like carryover
        # and measures the gap between two unrelated tasks.
        raise SessionError(
            f"{task_id}: warm session {warm.session_id!r} and cold control {cold.session_id!r} "
            "are different sessions; the comparison would be meaningless"
        )
    return CarryoverReport(
        task_id=task_id,
        warm_passed=warm.passed,
        cold_passed=cold.passed,
        warm_steps=warm.steps,
        cold_steps=cold.steps,
        memory_reads=memory_reads,
    )

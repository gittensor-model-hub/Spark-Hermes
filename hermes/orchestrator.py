"""The Hermes Orchestrator: what work is needed, not who does it.

Four responsibilities stay separate, and this module owns exactly one:

    Orchestrator   what subtasks must be completed?
    Router         which model-agent bundle performs each subtask?
    Expert worker  how is this specific subtask solved?
    Verifier       did the worker actually solve it?

The separation is the design. An orchestrator that also picks models becomes a second
router with worse information; one that also solves subtasks becomes a giant agent and
loses the plot on long horizons -- which is the failure the whole architecture exists to
avoid. So nothing here names a model, and a test asserts it: a `Plan` carries domains and
actions, and the router turns those into workers.

Three things live here beyond decomposition, because they are all "what work is needed
next" questions rather than "who should do it" questions:

**Sticky routing.** An expert keeps a subtask until it finishes or something specific
breaks. Re-deciding every message thrashes context and loses the thread mid-repair.

**Handoff state.** What passes between experts is a compact summary -- hypothesis, files
touched, test status, next action -- not the whole conversation. Forwarding the raw
transcript is how context corruption spreads from one worker to the next.

**Escalation.** Named, checkable conditions for giving up on the current assignment,
rather than a model deciding it feels stuck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hermes.router.spec import ACTIONS, DOMAINS, TaskSpecError

# Why a subtask left its current expert.
REROUTE_COMPLETED = "subtask_completed"
REROUTE_DOMAIN_CHANGED = "domain_changed"
REROUTE_REPEATED_FAILURE = "repeated_failure"
REROUTE_VERIFIER_FAILED = "verifier_failed_after_claim"
REROUTE_INVALID_TOOL_CALLS = "consecutive_invalid_tool_calls"
REROUTE_NO_PROGRESS = "no_progress"
REROUTE_TOOL_UNAVAILABLE = "required_tool_unavailable"
REROUTE_LOW_CONFIDENCE = "expert_reported_low_confidence"

# Defaults chosen to be forgiving of one bad step and intolerant of a pattern. A single
# invalid tool call is a typo; two in a row is a model that cannot drive the tool.
MAX_CONSECUTIVE_INVALID_CALLS = 2
MAX_REPEATED_FAILURES = 2
MAX_STEPS_WITHOUT_PROGRESS = 5


class PlanError(ValueError):
    """A plan is malformed or cannot be executed."""


@dataclass(frozen=True)
class Subtask:
    """One unit of work in a plan. Describes the work, never the worker."""

    subtask_id: str
    description: str
    domain: str
    action: str
    depends_on: tuple[str, ...] = ()
    verification: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subtask_id or not self.description.strip():
            raise PlanError("subtask needs an id and a description")
        if self.domain not in DOMAINS:
            raise TaskSpecError(f"unknown domain {self.domain!r}")
        if self.action not in ACTIONS:
            raise TaskSpecError(f"unknown action {self.action!r}")

    def to_record(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "description": self.description,
            "domain": self.domain,
            "action": self.action,
            "depends_on": list(self.depends_on),
            "verification": self.verification,
        }


@dataclass(frozen=True)
class Plan:
    """A DAG of subtasks answering "what has to happen", in dependency order."""

    request: str
    subtasks: tuple[Subtask, ...]

    def __post_init__(self) -> None:
        if not self.subtasks:
            raise PlanError("a plan needs at least one subtask")
        ids = [s.subtask_id for s in self.subtasks]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate subtask_id in plan")
        known = set(ids)
        for subtask in self.subtasks:
            missing = [d for d in subtask.depends_on if d not in known]
            if missing:
                raise PlanError(f"{subtask.subtask_id} depends on unknown subtask(s) {missing}")
        # Cycle detection here rather than at execution: a plan that cannot finish should
        # fail while it is still a plan, not halfway through a paid run.
        self.execution_order()

    def by_id(self, subtask_id: str) -> Subtask:
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        raise PlanError(f"unknown subtask {subtask_id!r}")

    def ready(self, completed: set[str]) -> tuple[Subtask, ...]:
        """Subtasks whose dependencies are all satisfied and which are not done.

        Returns every runnable subtask rather than one, so a caller that can run work in
        parallel is not forced into a sequence the plan never required.
        """
        return tuple(s for s in self.subtasks if s.subtask_id not in completed and set(s.depends_on) <= completed)

    def execution_order(self) -> tuple[str, ...]:
        """Topological order. Raises if the plan contains a cycle."""
        remaining = {s.subtask_id: set(s.depends_on) for s in self.subtasks}
        order: list[str] = []
        while remaining:
            free = sorted(sid for sid, deps in remaining.items() if not deps)
            if not free:
                raise PlanError(f"plan has a dependency cycle among {sorted(remaining)}")
            for sid in free:
                order.append(sid)
                del remaining[sid]
            for deps in remaining.values():
                deps.difference_update(free)
        return tuple(order)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.domain for s in self.subtasks))

    def to_record(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "subtasks": [s.to_record() for s in self.subtasks],
            "execution_order": list(self.execution_order()),
            "domains": list(self.domains),
        }


@dataclass(frozen=True)
class HandoffState:
    """What travels between experts when a subtask changes hands.

    Deliberately small. Forwarding the whole conversation is how one worker's context
    corruption becomes the next worker's, and a long transcript buries the two or three
    facts that actually carry the work forward.
    """

    subtask_id: str
    current_hypothesis: str = ""
    files_changed: tuple[str, ...] = ()
    tests: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    next_action: str = ""
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "current_hypothesis": self.current_hypothesis,
            "files_changed": list(self.files_changed),
            "tests": dict(self.tests),
            "artifacts": list(self.artifacts),
            "next_action": self.next_action,
            "notes": self.notes,
        }


@dataclass
class ExpertSession:
    """Tracks one expert's tenure on one subtask, and when it should end.

    Sticky by default: the assignment survives ordinary friction, including a failed
    tool call or an unhelpful result, because changing worker mid-repair discards
    everything the current one has established. It ends on specific, named conditions.
    """

    subtask_id: str
    agent_module: str
    domain: str
    consecutive_invalid_calls: int = 0
    repeated_failures: int = 0
    steps_without_progress: int = 0
    completed: bool = False

    def record_step(
        self,
        *,
        invalid_tool_call: bool = False,
        failed: bool = False,
        progressed: bool = False,
    ) -> None:
        self.consecutive_invalid_calls = self.consecutive_invalid_calls + 1 if invalid_tool_call else 0
        if failed:
            self.repeated_failures += 1
        if progressed:
            # Progress clears the failure streak: an agent that tried three things and
            # then got somewhere has recovered, which is the behaviour we want to keep.
            self.repeated_failures = 0
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

    def should_reroute(
        self,
        *,
        current_domain: str | None = None,
        verifier_failed_after_claim: bool = False,
        required_tool_missing: bool = False,
        expert_low_confidence: bool = False,
        max_invalid_calls: int = MAX_CONSECUTIVE_INVALID_CALLS,
        max_failures: int = MAX_REPEATED_FAILURES,
        max_idle_steps: int = MAX_STEPS_WITHOUT_PROGRESS,
    ) -> str | None:
        """The reason to hand off, or None to stay put."""
        if self.completed:
            return REROUTE_COMPLETED
        if verifier_failed_after_claim:
            # The strongest signal available: the expert said it was done and the
            # verifier disagreed, so its own judgement is what is unreliable here.
            return REROUTE_VERIFIER_FAILED
        if required_tool_missing:
            return REROUTE_TOOL_UNAVAILABLE
        if current_domain is not None and current_domain != self.domain:
            return REROUTE_DOMAIN_CHANGED
        if self.consecutive_invalid_calls >= max_invalid_calls:
            return REROUTE_INVALID_TOOL_CALLS
        if self.repeated_failures >= max_failures:
            return REROUTE_REPEATED_FAILURE
        if self.steps_without_progress >= max_idle_steps:
            return REROUTE_NO_PROGRESS
        if expert_low_confidence:
            return REROUTE_LOW_CONFIDENCE
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "agent_module": self.agent_module,
            "domain": self.domain,
            "consecutive_invalid_calls": self.consecutive_invalid_calls,
            "repeated_failures": self.repeated_failures,
            "steps_without_progress": self.steps_without_progress,
            "completed": self.completed,
        }


class Orchestrator:
    """Turns a request into a plan, and tracks who is on what.

    It never selects a model. `assign` takes a module id the *router* chose and records
    it; asking the orchestrator to choose would give the decision to the component with
    the least information about worker capability.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ExpertSession] = {}
        self._completed: set[str] = set()
        self._handoffs: dict[str, HandoffState] = {}

    def assign(self, subtask: Subtask, agent_module: str) -> ExpertSession:
        session = ExpertSession(subtask_id=subtask.subtask_id, agent_module=agent_module, domain=subtask.domain)
        self._sessions[subtask.subtask_id] = session
        return session

    def session_for(self, subtask_id: str) -> ExpertSession | None:
        return self._sessions.get(subtask_id)

    def complete(self, subtask_id: str, handoff: HandoffState | None = None) -> None:
        """Mark a subtask done and record what the next worker needs to know."""
        if subtask_id not in self._sessions:
            raise PlanError(f"cannot complete unassigned subtask {subtask_id!r}")
        self._sessions[subtask_id].completed = True
        self._completed.add(subtask_id)
        if handoff is not None:
            self._handoffs[subtask_id] = handoff

    def context_for(self, subtask: Subtask) -> tuple[HandoffState, ...]:
        """Handoff states from this subtask's direct dependencies only.

        Direct dependencies, not the whole history: a subtask three hops downstream does
        not need the intermediate hypotheses, and passing them along is how a long run
        accumulates context that is no longer true.
        """
        return tuple(self._handoffs[dep] for dep in subtask.depends_on if dep in self._handoffs)

    @property
    def completed(self) -> set[str]:
        return set(self._completed)

    def next_subtasks(self, plan: Plan) -> tuple[Subtask, ...]:
        return plan.ready(self._completed)

    def is_finished(self, plan: Plan) -> bool:
        return self._completed >= {s.subtask_id for s in plan.subtasks}

    def to_record(self) -> dict[str, Any]:
        return {
            "completed": sorted(self._completed),
            "sessions": {k: v.to_record() for k, v in sorted(self._sessions.items())},
            "handoffs": {k: v.to_record() for k, v in sorted(self._handoffs.items())},
        }

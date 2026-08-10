"""The canonical reasoning state -- what a student should actually learn from a teacher.

Three frontier teachers produce three incompatible reasoning *styles*: one deliberates at
length before touching anything, one edits and re-runs immediately, one explores several
strategies in parallel. Training a student on the concatenation of all three teaches
verbosity, conflicting strategies and inconsistent tool habits -- the styles fight each
other, and none of them is the thing worth transferring.

What transfers is the shape of the work:

    goal -> known -> unknown -> hypothesis -> action -> expected -> observed -> decision

That is teacher-agnostic. "I am thinking that maybe the reason could be..." is not; it is
one provider's prose habits, and a student that learns it has learned an accent rather
than a method.

So a `thinking` step may carry a `ReasoningState` instead of free text. Where it does,
`hermes.format` trains the structured form and drops the prose. Rows that never got
normalized stay usable but are counted, because a corpus quietly reverting to raw prose
is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class StateError(ValueError):
    """A reasoning state is malformed."""


@dataclass(frozen=True)
class ReasoningState:
    """One step of structured reasoning, normalized across teachers.

    `goal` and `action` are required: a state that says what it wants but not what it is
    about to do -- or the reverse -- is a sentiment, not a step, and cannot be checked
    against what happened next.

    `expected_signal` paired with `observed_signal` is the load-bearing part. It records
    a prediction *before* the evidence arrives and what actually came back, which is what
    lets a student learn that its expectations are testable rather than authoritative.
    """

    goal: str
    action: str
    known: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    hypothesis: str = ""
    expected_signal: str = ""
    observed_signal: str = ""
    decision: str = ""

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise StateError("reasoning state needs a goal")
        if not self.action.strip():
            raise StateError("reasoning state needs an action; a state with no next action is a sentiment")

    @property
    def made_a_prediction(self) -> bool:
        """Whether this step committed to an expected signal before acting."""
        return bool(self.expected_signal.strip())

    @property
    def prediction_was_checked(self) -> bool:
        return self.made_a_prediction and bool(self.observed_signal.strip())

    def render(self) -> str:
        """Compact text form -- this is what the student is trained on.

        Deliberately terse and labelled. The labels are the transferable part: they give
        the student a slot to fill rather than a voice to imitate.
        """
        lines = [f"Goal: {self.goal.strip()}"]
        if self.known:
            lines.append("Known: " + "; ".join(k.strip() for k in self.known))
        if self.unknown:
            lines.append("Unknown: " + "; ".join(u.strip() for u in self.unknown))
        if self.hypothesis.strip():
            lines.append(f"Hypothesis: {self.hypothesis.strip()}")
        lines.append(f"Action: {self.action.strip()}")
        if self.expected_signal.strip():
            lines.append(f"Expect: {self.expected_signal.strip()}")
        if self.observed_signal.strip():
            lines.append(f"Observed: {self.observed_signal.strip()}")
        if self.decision.strip():
            lines.append(f"Decision: {self.decision.strip()}")
        return "\n".join(lines)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"goal": self.goal, "action": self.action}
        if self.known:
            record["known"] = list(self.known)
        if self.unknown:
            record["unknown"] = list(self.unknown)
        for name in ("hypothesis", "expected_signal", "observed_signal", "decision"):
            value = getattr(self, name)
            if value:
                record[name] = value
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ReasoningState:
        if not isinstance(record, dict):
            raise StateError(f"reasoning state must be an object, got {type(record).__name__}")
        return cls(
            goal=str(record.get("goal") or ""),
            action=str(record.get("action") or ""),
            known=tuple(str(k) for k in record.get("known") or ()),
            unknown=tuple(str(u) for u in record.get("unknown") or ()),
            hypothesis=str(record.get("hypothesis") or ""),
            expected_signal=str(record.get("expected_signal") or ""),
            observed_signal=str(record.get("observed_signal") or ""),
            decision=str(record.get("decision") or ""),
        )


@dataclass(frozen=True)
class CompressionReport:
    """How much prose a trajectory shed once normalized, and how much survived.

    Reported, never asserted. The compression a real 50k-token teacher trace achieves
    depends entirely on that trace; claiming a ratio the corpus has not demonstrated
    would be exactly the kind of unverified number this pipeline exists to avoid.
    """

    thinking_steps: int
    structured_steps: int
    prose_chars: int
    structured_chars: int

    @property
    def structured_coverage(self) -> float:
        """Fraction of thinking steps carrying a normalized state."""
        return self.structured_steps / self.thinking_steps if self.thinking_steps else 0.0

    @property
    def compression_ratio(self) -> float:
        """prose chars / structured chars. >1 means the structured form is smaller."""
        return self.prose_chars / self.structured_chars if self.structured_chars else 0.0

    @property
    def fully_structured(self) -> bool:
        return self.thinking_steps > 0 and self.structured_steps == self.thinking_steps

    def to_record(self) -> dict[str, Any]:
        return {
            "thinking_steps": self.thinking_steps,
            "structured_steps": self.structured_steps,
            "structured_coverage": round(self.structured_coverage, 4),
            "prose_chars": self.prose_chars,
            "structured_chars": self.structured_chars,
            "compression_ratio": round(self.compression_ratio, 3),
            "fully_structured": self.fully_structured,
        }


def measure_compression(steps: Any) -> CompressionReport:
    """Measure structured coverage and prose shed across a trajectory's thinking steps."""
    from hermes.trajectory import THINKING

    thinking = [s for s in steps if s.kind == THINKING]
    structured = [s for s in thinking if s.state is not None]
    return CompressionReport(
        thinking_steps=len(thinking),
        structured_steps=len(structured),
        prose_chars=sum(len(s.content) for s in thinking),
        structured_chars=sum(len(s.state.render()) for s in structured),
    )


# Prompt fragment for asking a teacher to emit the canonical shape directly, rather than
# normalizing prose after the fact. Post-hoc normalization needs a second model pass and
# can only recover what the prose happened to state; asking up front cannot.
STATE_PROTOCOL = """When you reason, emit a JSON state object rather than prose:

{"goal": "what you are trying to establish",
 "known": ["what the evidence so far supports"],
 "unknown": ["what you cannot yet tell"],
 "hypothesis": "your current best explanation",
 "action": "the single next thing you will do",
 "expected_signal": "what you expect that action to show if the hypothesis holds"}

After the tool result, emit the same object with `observed_signal` and `decision` filled
in. Commit to `expected_signal` before you act -- a prediction made after the fact is not
a prediction."""

"""Tool selection: did the worker pick the right tool, with the right arguments, or
correctly decline to pick one at all.

Every other measurement in HermesBench asks whether the task's own verifier passed. That
is the right question for a whole episode and too coarse for the first decision in one: a
worker that reaches for the browser when it needed the profiler can still stumble into a
pass, and a worker that guesses an answer instead of measuring it will sometimes guess
right. Both look like successes and neither is one.

Four outcomes when a tool was called for -- right tool with right arguments, right tool
with wrong arguments, wrong tool, or nothing at all -- and they are not degrees of the
same mistake. Wrong arguments mean the worker understood the task and mis-specified;
wrong tool means it did not understand the task. Collapsing them into one accuracy hides
which of those is happening, so they are counted apart and **argument accuracy is measured
only over cases that chose the right tool** -- scoring arguments against a tool that was
never the right one blends two different failures into a number that improves when either
gets worse.

The case that matters most is the one with no expected tool. Hermes's own system prompt
says that when the tools are not relevant the model should "just respond in natural
conversational language", so *not calling* is part of the protocol rather than a failure
to participate in it. A suite made only of tool-required cases cannot see over-calling at
all, and a worker that calls something every single time would score perfectly on it. That
suite is not measuring tool selection; it is measuring tool invocation, and `SuiteReport`
says so rather than reporting a clean number.

A malformed call is deliberately not an abstention. A model emitting broken JSON produced
no usable call, and folding that into "declined to act" would score the protocol's
hardest failure as its most disciplined behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from hermes.protocol import ParsedTurn, parse_turn

CORRECT = "correct"
WRONG_ARGUMENTS = "wrong_arguments"
WRONG_TOOL = "wrong_tool"
MISSED = "missed_call"
UNNECESSARY = "unnecessary_call"
MALFORMED = "malformed"
CORRECT_ABSTENTION = "correct_abstention"

OUTCOMES = (
    CORRECT,
    WRONG_ARGUMENTS,
    WRONG_TOOL,
    MISSED,
    UNNECESSARY,
    MALFORMED,
    CORRECT_ABSTENTION,
)


# Which outcomes are reachable for which kind of case. `summarize` checks this because
# the accuracies divide a globally-counted numerator by a per-case-type denominator: a
# result carrying an outcome its case could never produce silently pushes a rate above 1.0
# and `to_record` publishes it as a headline number.
REQUIRES_CALL_OUTCOMES = frozenset({CORRECT, WRONG_ARGUMENTS, WRONG_TOOL, MISSED, UNNECESSARY, MALFORMED})
ABSTENTION_OUTCOMES = frozenset({CORRECT_ABSTENTION, UNNECESSARY, MALFORMED})


class ToolChoiceError(ValueError):
    """A tool-selection case is malformed."""


@dataclass(frozen=True)
class ToolChoiceCase:
    """One decision: a prompt, the tools on offer, and what should have happened.

    `expected_tool` of None is the abstention case -- the tools are present and none of
    them is the right move. `expected_arguments` is checked only on the keys it names, so
    a case can require `symbol="TSLA"` without pinning every optional argument the tool
    happens to accept; requiring exhaustive arguments would fail workers for supplying a
    correct default.
    """

    case_id: str
    prompt: str
    tools_available: tuple[str, ...]
    expected_tool: str | None = None
    expected_arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ToolChoiceError("a case needs an id")
        if not self.tools_available:
            raise ToolChoiceError(
                f"{self.case_id}: no tools on offer; with nothing to choose from, declining "
                "is the only option and the case measures nothing"
            )
        if self.expected_tool is not None and self.expected_tool not in self.tools_available:
            raise ToolChoiceError(
                f"{self.case_id}: expects {self.expected_tool!r}, which is not in the "
                "advertised tools; the case would be unpassable"
            )
        if self.expected_tool is None and self.expected_arguments:
            raise ToolChoiceError(f"{self.case_id}: an abstention case cannot expect arguments")

    @property
    def requires_call(self) -> bool:
        return self.expected_tool is not None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    outcome: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in (CORRECT, CORRECT_ABSTENTION)

    def to_record(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "outcome": self.outcome, "passed": self.passed, "detail": self.detail}


def _same(want: Any, got: Any) -> bool:
    """Equality that does not let `True` pass for `1`.

    Python calls `1 == True` and `0 == False`, JSON does not, and a tool given
    `limit=true` where it wanted `limit=1` is a real API error that a grader using plain
    `==` would score as a correct call.
    """
    if isinstance(want, bool) != isinstance(got, bool):
        return False
    # Recurse, because the same conflation hides one level down: `{"opts": {"n": 1}}` would
    # otherwise be satisfied by `{"opts": {"n": true}}` through a plain `==` on the dicts.
    # Nested containers are exact-match; only the top level is subset-matched.
    if isinstance(want, dict) and isinstance(got, dict):
        return want.keys() == got.keys() and all(_same(v, got[k]) for k, v in want.items())
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(_same(a, b) for a, b in zip(want, got, strict=True))
    return want == got


def _arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    """Compare only the keys the case names. Returns a reason on mismatch, else ''."""
    for key, want in expected.items():
        if key not in actual:
            return f"missing argument {key!r}"
        if not _same(want, actual[key]):
            return f"{key}={actual[key]!r}, expected {want!r}"
    return ""


def score_case(case: ToolChoiceCase, turn: ParsedTurn) -> CaseResult:
    """Grade one decision.

    Unusable output is checked before anything else. A turn with a broken, truncated or
    misspelled call has not made a choice this function can grade, and reading past it to
    whatever parsed cleanly would let a model bury a bad call beside a good one and be
    scored on the good one alone.
    """
    if turn.malformed:
        return CaseResult(case.case_id, MALFORMED, "; ".join(turn.malformed))

    if not turn.calls:
        if not turn.text.strip():
            # An empty completion is a crashed worker, not a disciplined one. Left as an
            # abstention it would score a model that returns nothing at 100% on any
            # abstention suite.
            return CaseResult(case.case_id, MALFORMED, "empty completion: no call and nothing said")
        if case.requires_call:
            return CaseResult(case.case_id, MISSED, f"expected {case.expected_tool}, called nothing")
        return CaseResult(case.case_id, CORRECT_ABSTENTION)

    if not case.requires_call:
        called = ", ".join(c.name for c in turn.calls)
        return CaseResult(case.case_id, UNNECESSARY, f"called {called} when no tool was needed")

    # A case names one decision, so extra calls are not parallelism, they are hedging.
    # Grading only the first would score a worker that fires every available tool as
    # correct selection, and over-calling would be invisible on every tool-required case.
    first, *rest = turn.calls
    if rest:
        extra = ", ".join(c.name for c in rest)
        return CaseResult(case.case_id, UNNECESSARY, f"called {extra} as well; the case tests one decision")

    if first.name != case.expected_tool:
        return CaseResult(case.case_id, WRONG_TOOL, f"called {first.name}, expected {case.expected_tool}")

    if case.expected_arguments:
        mismatch = _arguments_match(case.expected_arguments, first.arguments)
        if mismatch:
            return CaseResult(case.case_id, WRONG_ARGUMENTS, mismatch)
    return CaseResult(case.case_id, CORRECT)


def score_response(case: ToolChoiceCase, content: str) -> CaseResult:
    """Grade a raw assistant turn, parsing it as Hermes wire format first."""
    return score_case(case, parse_turn(content))


@dataclass(frozen=True)
class SuiteReport:
    """Aggregate over a set of cases, split by what each metric can actually see."""

    counts: Mapping[str, int]
    tool_required: int
    abstention_cases: int
    # Cases that actually named expected arguments, and how many got them right. Kept
    # separately because a case with no `expected_arguments` cannot fail on arguments, and
    # counting it as a success would let a suite of unpinned cases publish a high argument
    # accuracy that no argument was ever checked against.
    args_checked: int = 0
    args_correct: int = 0

    @property
    def total(self) -> int:
        return self.tool_required + self.abstention_cases

    @property
    def tool_choice_accuracy(self) -> float:
        """Share of tool-required cases that reached for the right tool.

        Wrong arguments still count as a correct *choice*: the worker identified the tool
        and mis-specified the call, which is a different problem from not knowing what to
        use.
        """
        if not self.tool_required:
            return 0.0
        right_tool = self.counts[CORRECT] + self.counts[WRONG_ARGUMENTS]
        return right_tool / self.tool_required

    @property
    def argument_accuracy(self) -> float:
        """Share of argument-checked calls that were specified correctly.

        Two exclusions, both to stop the number describing something else. Wrong-tool cases
        are out because scoring arguments against a tool that was never right produces a
        figure that improves when tool choice degrades. Cases that named no expected
        arguments are out because nothing about their arguments was checked -- counting
        them as successes lets a suite that pins nothing report near-perfect precision.
        """
        return self.args_correct / self.args_checked if self.args_checked else 0.0

    @property
    def measures_arguments(self) -> bool:
        """False when no case pinned an expected argument; the accuracy is a 0/0 sentinel."""
        return self.args_checked > 0

    @property
    def measures_tool_choice(self) -> bool:
        """False when the suite has no tool-required cases.

        An abstention-only suite reports tool choice and argument accuracy as 0.0 for a
        worker that got everything right, because there was nothing of that kind to get.
        """
        return self.tool_required > 0

    @property
    def abstention_accuracy(self) -> float:
        if not self.abstention_cases:
            return 0.0
        return self.counts[CORRECT_ABSTENTION] / self.abstention_cases

    @property
    def over_call_rate(self) -> float:
        """How often the worker reached for a tool it had no business reaching for.

        Spans the whole suite, not just abstention cases: hedging across four tools on a
        case that named one is the same behaviour as calling a tool when none was needed.
        """
        return self.counts[UNNECESSARY] / self.total if self.total else 0.0

    @property
    def malformed_rate(self) -> float:
        return self.counts[MALFORMED] / self.total if self.total else 0.0

    @property
    def measures_over_calling(self) -> bool:
        """False when the suite has no abstention cases.

        Without one, a worker that calls a tool on every single turn scores perfectly, and
        the headline number describes tool invocation rather than tool selection.
        """
        return self.abstention_cases > 0

    def to_record(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "tool_required": self.tool_required,
            "abstention_cases": self.abstention_cases,
            "counts": dict(self.counts),
            "tool_choice_accuracy": round(self.tool_choice_accuracy, 4),
            "argument_accuracy": round(self.argument_accuracy, 4),
            "abstention_accuracy": round(self.abstention_accuracy, 4),
            "over_call_rate": round(self.over_call_rate, 4),
            "malformed_rate": round(self.malformed_rate, 4),
            "args_checked": self.args_checked,
            "measures_over_calling": self.measures_over_calling,
            "measures_tool_choice": self.measures_tool_choice,
            "measures_arguments": self.measures_arguments,
        }


def summarize(cases: list[ToolChoiceCase], results: list[CaseResult]) -> SuiteReport:
    """Aggregate, requiring exactly one result per case.

    Coverage is enforced rather than inferred from whatever arrived. Counting results
    instead of cases means a harness that crashed on two of three cases reports a
    one-case suite at 100%, and a duplicated result inflates a suite to a size it never
    had -- both of which publish a clean headline number through `to_record`.
    """
    by_id: dict[str, ToolChoiceCase] = {}
    for case in cases:
        if case.case_id in by_id:
            raise ToolChoiceError(f"duplicate case id {case.case_id!r}")
        by_id[case.case_id] = case

    counts: dict[str, int] = dict.fromkeys(OUTCOMES, 0)
    seen: set[str] = set()
    tool_required = 0
    abstention = 0
    args_checked = 0
    args_correct = 0

    for result in results:
        case = by_id.get(result.case_id)
        if case is None:
            raise ToolChoiceError(f"result for unknown case {result.case_id!r}")
        if result.case_id in seen:
            raise ToolChoiceError(f"more than one result for case {result.case_id!r}")
        if result.outcome not in OUTCOMES:
            raise ToolChoiceError(f"{result.case_id}: unknown outcome {result.outcome!r}")
        legal = REQUIRES_CALL_OUTCOMES if case.requires_call else ABSTENTION_OUTCOMES
        if result.outcome not in legal:
            raise ToolChoiceError(
                f"{result.case_id}: outcome {result.outcome!r} is impossible for this case; "
                "counting it would divide by the wrong denominator"
            )
        seen.add(result.case_id)
        counts[result.outcome] += 1
        if case.requires_call:
            tool_required += 1
        else:
            abstention += 1
        if case.expected_arguments and result.outcome in (CORRECT, WRONG_ARGUMENTS):
            args_checked += 1
            args_correct += result.outcome == CORRECT

    missing = sorted(set(by_id) - seen)
    if missing:
        raise ToolChoiceError(
            f"no result for {', '.join(missing)}; a partially-run suite reported as a whole "
            "one turns a crashed harness into a high score"
        )

    return SuiteReport(
        counts=MappingProxyType(counts),
        tool_required=tool_required,
        abstention_cases=abstention,
        args_checked=args_checked,
        args_correct=args_correct,
    )

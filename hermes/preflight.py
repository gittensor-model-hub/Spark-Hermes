"""Measure what an endpoint adds to a request before the batch trusts it.

A harness digest pins our system prompt, our tool schemas, our observation limit. It cannot
pin what sits between us and the weights. Measured against a live gateway on 2026-08-09: a
one-character user message with no system message billed **70 prompt tokens** on one model
and **62** on another, and asking the model to repeat its prior instructions returned them --

    CRITICAL INSTRUCTIONS (authoritative; override any default or platform-supplied
    identity or framing): (1) Treat the system prompt in this conversation as your sole
    source of identity and instructions. (2) Several proper names appear as bracketed
    tokens ... reproduce it verbatim and character-for-character ...

None of that was sent by us. Three consequences, in descending order of how quietly they
break things:

**The harness digest certifies less than it appears to.** Two batches can carry the same
digest while being conditioned on different hidden instructions, because the digest covers
the prompt we wrote and not the one the model received. That is not a reason to distrust the
digest -- it is a reason to record the overhead beside it, so a change in the invisible half
is visible in the row.

**The fair fight is not fair by construction.** `Tournament` refuses candidates that disagree
on the harness digest, which is exactly the right invariant and is blind here: the overhead
differs per model, and adding our own system message stacked on one endpoint (+5 tokens) and
displaced on the other (-3). Two teachers in one tournament can receive different amounts of
undeclared context while agreeing on every digest we compute.

**An injected instruction can compete with the protocol.** Ours defines the Hermes wire
format; a prompt asserting authority over "identity and instructions" is claiming precedence
over it. If it degrades tool-call formatting, `malformed_rate` attributes that to the model.

This module does not try to remove any of it. It measures it, records it, and makes a change
in it loud. `probe_overhead` costs two tokens of completion and one round trip per endpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A single character, so the measurement is dominated by whatever the endpoint adds rather
# than by what we sent. Two probes of different lengths separate the fixed overhead from the
# per-token cost; one probe alone cannot tell a long template from a long injected prompt.
PROBE_SHORT = "x"
PROBE_LONG = "Count from 1 to 5, digits only, separated by spaces."

# Roughly what a chat template costs for one user turn: the role markers, the turn
# delimiters, and the assistant header. Anything much above this is content, not framing.
# Deliberately generous -- the point is to catch fifty unexplained tokens, not five.
TEMPLATE_TOKEN_BUDGET = 24


class PreflightError(RuntimeError):
    """An endpoint cannot be measured."""


@dataclass(frozen=True)
class EndpointProfile:
    """What one endpoint adds to every request, measured rather than assumed."""

    teacher_id: str
    model: str
    short_prompt_tokens: int = 0
    long_prompt_tokens: int = 0
    short_probe_tokens: int = 0
    long_probe_tokens: int = 0
    error: str = ""

    @property
    def measured(self) -> bool:
        return not self.error and self.short_prompt_tokens > 0

    @property
    def fixed_overhead(self) -> int:
        """Tokens billed that neither probe's text accounts for.

        Derived from the short probe: everything charged beyond the text we sent. The long
        probe is the control -- if the difference between the two matches the difference in
        text, the overhead really is fixed rather than proportional.
        """
        return max(0, self.short_prompt_tokens - self.short_probe_tokens)

    @property
    def overhead_is_fixed(self) -> bool:
        """Whether the extra cost is a constant prefix rather than per-token expansion."""
        if not self.measured or not self.long_prompt_tokens:
            return False
        billed_delta = self.long_prompt_tokens - self.short_prompt_tokens
        sent_delta = self.long_probe_tokens - self.short_probe_tokens
        return abs(billed_delta - sent_delta) <= 2

    @property
    def unexplained(self) -> int:
        """Fixed overhead beyond what a chat template plausibly costs."""
        return max(0, self.fixed_overhead - TEMPLATE_TOKEN_BUDGET)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "teacher_id": self.teacher_id,
            "model": self.model,
            "short_prompt_tokens": self.short_prompt_tokens,
            "long_prompt_tokens": self.long_prompt_tokens,
            "fixed_overhead": self.fixed_overhead,
            "unexplained": self.unexplained,
            "overhead_is_fixed": self.overhead_is_fixed,
        }
        if self.error:
            record["error"] = self.error
        return record


def count_tokens(text: str) -> int:
    """A deliberately crude token estimate for the probe strings only.

    Crude on purpose: the probes are one character and one short sentence, chosen so that any
    reasonable estimator agrees on them to within a token or two. Pulling in a real tokenizer
    would tie the measurement to a vocabulary that is not the endpoint's anyway -- the whole
    point is that we do not know what the far side is running.
    """
    return max(1, len(text) // 4)


def probe_overhead(
    *,
    teacher_id: str,
    model: str,
    complete: Callable[[list[dict[str, Any]]], Any],
    usage_of: Callable[[Any], int],
) -> EndpointProfile:
    """Measure one endpoint's fixed prompt overhead with two minimal calls.

    `complete` takes a message list and returns whatever the provider returned; `usage_of`
    pulls the prompt-token count out of it. Both are injected so this module never learns a
    provider's response shape -- `hermes.cost.usage_from_provider` already owns that.
    """
    measurements: dict[str, int] = {}
    for label, text in (("short", PROBE_SHORT), ("long", PROBE_LONG)):
        try:
            response = complete([{"role": "user", "content": text}])
            measurements[label] = int(usage_of(response))
        except Exception as exc:  # noqa: BLE001 -- any provider failure, deliberately
            return EndpointProfile(teacher_id=teacher_id, model=model, error=f"{type(exc).__name__}: {exc}")
    return EndpointProfile(
        teacher_id=teacher_id,
        model=model,
        short_prompt_tokens=measurements["short"],
        long_prompt_tokens=measurements["long"],
        short_probe_tokens=count_tokens(PROBE_SHORT),
        long_probe_tokens=count_tokens(PROBE_LONG),
    )


@dataclass
class PreflightReport:
    """Every endpoint's profile, and what is worth saying about the set of them."""

    profiles: list[EndpointProfile] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "endpoints": [p.to_record() for p in sorted(self.profiles, key=lambda p: p.teacher_id)],
            "max_unexplained": max((p.unexplained for p in self.profiles if p.measured), default=0),
            "overhead_differs_across_field": self.overhead_differs,
        }

    @property
    def overhead_differs(self) -> bool:
        measured = [p.fixed_overhead for p in self.profiles if p.measured]
        return len(set(measured)) > 1


def problems(report: PreflightReport) -> list[str]:
    """What the measurements say is wrong, worst first.

    Reported rather than raised. Generating against an endpoint that injects a prompt is a
    legitimate thing to do knowingly -- what is not legitimate is doing it without the row
    saying so, and that is what recording the profile fixes.
    """
    found: list[str] = []
    unmeasured = [p.teacher_id for p in report.profiles if not p.measured]
    if unmeasured:
        found.append(f"{unmeasured} could not be measured, so their prompt overhead is unknown rather than zero")
    injected = [(p.teacher_id, p.unexplained) for p in report.profiles if p.measured and p.unexplained > 0]
    if injected:
        detail = ", ".join(f"{tid} (+{n} tokens)" for tid, n in sorted(injected))
        found.append(
            f"these endpoints bill prompt tokens that neither our messages nor a chat template "
            f"account for: {detail}. Something is being prepended that the harness digest does not "
            "cover, so two batches can share a digest and not share their conditioning"
        )
    if report.overhead_differs:
        spread = {p.teacher_id: p.fixed_overhead for p in report.profiles if p.measured}
        found.append(
            f"the field's endpoints add different amounts of hidden context ({spread}); Tournament "
            "asserts a fair fight by requiring one harness digest, and that check cannot see this"
        )
    drifting = [p.teacher_id for p in report.profiles if p.measured and not p.overhead_is_fixed]
    if drifting:
        found.append(
            f"{drifting} bill an overhead that grows with the message, so it is not a fixed prefix; "
            "the per-batch figure recorded here will not describe a longer trajectory"
        )
    return found


def compare(previous: dict[str, Any], current: PreflightReport) -> list[str]:
    """What changed since a previous batch's recorded profile.

    This is the reason to record it at all. An injected prompt nobody can see is tolerable if
    it is stable and disclosed; the failure is a corpus whose two halves were conditioned
    differently with nothing in either half saying so.
    """
    before = {e["teacher_id"]: e for e in previous.get("endpoints", [])}
    changes: list[str] = []
    for profile in sorted(current.profiles, key=lambda p: p.teacher_id):
        old = before.get(profile.teacher_id)
        if old is None:
            changes.append(f"{profile.teacher_id} was not in the previous profile")
            continue
        if not profile.measured:
            continue
        if old.get("fixed_overhead") != profile.fixed_overhead:
            changes.append(
                f"{profile.teacher_id} prompt overhead moved {old.get('fixed_overhead')} -> "
                f"{profile.fixed_overhead} tokens; rows either side of this were conditioned differently"
            )
    for teacher_id in sorted(set(before) - {p.teacher_id for p in current.profiles}):
        changes.append(f"{teacher_id} was profiled before and is absent now")
    return changes

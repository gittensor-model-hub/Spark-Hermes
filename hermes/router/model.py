"""Model-backed router: the 3B-7B classifier Phase 4 actually calls for.

No model is served here. This wires the prompt, the parse, and the guard rails around a
`complete(prompt) -> str` callable the caller supplies, so the routing logic is testable
today and a real endpoint drops in later without touching any of it.

Two guards matter more than the prompt:

- **An unknown target is an abstention, not a crash and not a guess.** A router that
  hallucinates `Spark-Hermes-Database` must fall back to the generalist, because the
  alternative is dispatching to a worker that does not exist.
- **Low self-reported confidence is an abstention.** The model is asked for one, and it
  is enforced here rather than trusted downstream.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from hermes.router.base import (
    EMPTY_TASK,
    LOW_CONFIDENCE,
    MALFORMED_REPLY,
    ROUTER_ERROR,
    TIER_MODEL,
    UNKNOWN_TARGET,
    RoutingDecision,
    abstain,
)
from hermes.router.domains import DOMAINS, GENERAL, is_valid_target

DEFAULT_MIN_CONFIDENCE = 0.6

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(task: str) -> str:
    """Routing prompt listing every specialist and the abstention option."""
    lines = [f"- {key}: {domain.description}" for key, domain in DOMAINS.items()]
    lines.append(f"- {GENERAL}: anything else, or anything you are not sure about")
    catalogue = "\n".join(lines)
    return (
        "You route engineering tasks to the specialist best suited to them.\n\n"
        f"Specialists:\n{catalogue}\n\n"
        "Choose the single best target. If the task spans several domains, or you are "
        f"not confident, choose {GENERAL} -- a specialist working outside its domain is "
        "worse than a generalist working inside its limits.\n\n"
        'Reply with only a JSON object: {"target": "<key>", "confidence": <0..1>, '
        '"reason": "<one short sentence>"}\n\n'
        f"Task:\n{task}"
    )


def parse_decision(text: str, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> RoutingDecision:
    """Parse a router model's reply, abstaining on anything malformed or unsure."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return abstain("router reply was not JSON", reason_code=MALFORMED_REPLY, tier=TIER_MODEL)
    try:
        parsed: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return abstain("router reply was not valid JSON", reason_code=MALFORMED_REPLY, tier=TIER_MODEL)
    if not isinstance(parsed, dict):
        return abstain("router reply was not a JSON object", reason_code=MALFORMED_REPLY, tier=TIER_MODEL)

    target = str(parsed.get("target", "")).strip().lower()
    if not is_valid_target(target):
        # Hallucinated specialist: fall back rather than dispatch into the void.
        return abstain(f"router named unknown target {target!r}", reason_code=UNKNOWN_TARGET, tier=TIER_MODEL)

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        return abstain("router gave a non-numeric confidence", reason_code=MALFORMED_REPLY, tier=TIER_MODEL)
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason") or "").strip() or "no reason given"

    if target == GENERAL:
        # A deliberate `general` is a real decision, not an abstention -- it is scored
        # differently, so the distinction has to survive parsing.
        return RoutingDecision(target=GENERAL, confidence=confidence, reason=reason, tier=TIER_MODEL)

    if confidence < min_confidence:
        return abstain(
            f"router chose {target} but only at confidence {confidence:.2f} (needs {min_confidence:.2f})",
            confidence=confidence,
            reason_code=LOW_CONFIDENCE,
            tier=TIER_MODEL,
        )
    return RoutingDecision(target=target, confidence=confidence, reason=reason, tier=TIER_MODEL)


class ModelRouter:
    """Routes via a supplied completion callable.

    `complete` takes the prompt and returns the model's raw text. Anything that can do
    that works -- a served 3B checkpoint, an API call, or a stub in a test.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.complete = complete
        self.min_confidence = min_confidence

    def route(self, task: str) -> RoutingDecision:
        if not task or not task.strip():
            return abstain("empty task", reason_code=EMPTY_TASK, tier=TIER_MODEL)
        try:
            reply = self.complete(build_prompt(task))
        except Exception as exc:  # noqa: BLE001 - a dead router must degrade, not crash the run
            # The router is infrastructure in front of every request. If it falls over,
            # the generalist should still get the work.
            return abstain(
                f"router call failed: {type(exc).__name__}: {exc}", reason_code=ROUTER_ERROR, tier=TIER_MODEL
            )
        return parse_decision(reply, min_confidence=self.min_confidence)

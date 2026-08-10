"""Recover train-able CoT when a teacher answers with encrypted/empty reasoning.

GPT 5.6 over chat-completions frequently exposes no usable reasoning trace: the
`reasoning_content` field is absent, empty, or an encrypted `reasoning_details`
JSON dump. SparkDistill trains students to *reproduce reasoning*, so such a
trajectory becomes a bare-answer SFT row with no `<think>` block — the lowest
value kind of distillation data (see `teacher/format.py`).

Adapted from SparkProof's `training_cot` (PR #31). SparkDistill has no per-prompt
GPU validator and already runs both teachers on every prompt, so SparkProof's
kernel "re-solve + re-validate on hardware" path does not apply here. The honest,
general-purpose port keeps two of its ideas:

- **Normalize** the captured reasoning, dropping encrypted `reasoning_details`
  JSON so it never gets wrapped in `<think>` tags as if it were plaintext CoT.
- **Explain fallback**: when a non-Fable trajectory has no usable reasoning, ask
  Claude Fable 5 to explain how to reach that answer and attach Fable's plaintext
  rationale as the `<think>` trace, tagged in `metadata.cot_recovery` so the
  provenance stays honest.

This does **not** decrypt a teacher's private CoT — it produces a Fable-authored,
inspectable rationale suitable for student SFT.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from teacher.providers import Trajectory

# Fable 5 authors the recovered rationale; it already emits plaintext extended
# thinking, so its own trajectories never need (or trigger) recovery.
RECOVERY_PROVIDER = "anthropic"

_MIN_REASONING_CHARS = 32
_MIN_PLAINTEXT_CHARS = 8


def extract_plaintext_reasoning_details(details: Any) -> str | None:
    """Pull usable text from OpenRouter/gateway ``reasoning_details`` (skip encrypted)."""
    if not isinstance(details, list):
        return None
    chunks: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if "encrypted" in kind:
            continue
        if kind == "reasoning.text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        elif kind == "reasoning.summary":
            summary = item.get("summary") or item.get("text")
            if isinstance(summary, str) and summary.strip():
                chunks.append(summary.strip())
    joined = "\n\n".join(chunks).strip()
    return joined or None


def normalize_training_reasoning(reasoning: Any) -> str | None:
    """Return plaintext CoT usable for ``<think>`` SFT tags, or None.

    Drops encrypted ``reasoning_details`` JSON dumps and reasoning too short to be
    a meaningful trace.
    """
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    text = reasoning.strip()
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text if len(text) >= _MIN_PLAINTEXT_CHARS else None
        extracted = extract_plaintext_reasoning_details(parsed)
        if extracted and len(extracted) >= _MIN_REASONING_CHARS:
            return extracted
        return None
    return text if len(text) >= _MIN_PLAINTEXT_CHARS else None


def has_usable_training_reasoning(reasoning: Any) -> bool:
    """True when ``reasoning`` is plaintext suitable for ``<think>`` SFT tags."""
    return normalize_training_reasoning(reasoning) is not None


def explain_cot_prompt(*, task_prompt: str, answer: str) -> str:
    """Ask Fable for a step-by-step rationale that leads to an already-correct answer."""
    return (
        "The answer below is correct for the given problem. Write a clear, "
        "self-contained chain-of-thought that reasons step by step to exactly that "
        "answer — the way an expert would think it through before writing it down. "
        "Explain the key steps and why they hold; do not merely restate the answer. "
        "Return only the reasoning.\n\n"
        f"## Problem\n{task_prompt}\n\n"
        f"## Correct answer\n{answer}\n"
    )


def recover_trajectory_cot(
    trajectory: Trajectory,
    fable_teacher: Any,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> Trajectory:
    """Return a trajectory whose ``reasoning`` is plaintext CoT usable for ``<think>``.

    - If the trajectory already has usable reasoning, return it with that reasoning
      normalized (encrypted JSON dropped) — no extra teacher call.
    - Fable's own trajectories, and any trajectory with no answer to explain, are
      returned unchanged.
    - Otherwise ask ``fable_teacher`` to explain how to reach the answer and attach
      the plaintext rationale, stamping ``metadata.cot_recovery = "fable_explain"``
      (or ``"failed"`` when Fable also produced nothing usable).
    """
    existing = normalize_training_reasoning(trajectory.reasoning)
    if existing is not None:
        return trajectory if existing == trajectory.reasoning else replace(trajectory, reasoning=existing)

    # Fable already emits plaintext thinking; only non-Fable teachers need recovery,
    # and only when there is an actual answer to build a rationale from.
    if trajectory.provider == RECOVERY_PROVIDER or not (trajectory.response or "").strip():
        return trajectory

    explanation = fable_teacher.generate(
        explain_cot_prompt(task_prompt=trajectory.prompt, answer=trajectory.response),
        system=trajectory.system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Fable's visible response *is* the rationale here; fall back to any extended
    # thinking trace only if the response came back empty.
    rationale = normalize_training_reasoning(explanation.response) or normalize_training_reasoning(
        explanation.reasoning
    )

    metadata = dict(trajectory.metadata)
    if rationale is None:
        metadata["cot_recovery"] = "failed"
        return replace(trajectory, reasoning=None, metadata=metadata)
    metadata["cot_recovery"] = "fable_explain"
    metadata["cot_provider"] = fable_teacher.name
    metadata["cot_model"] = fable_teacher.model
    return replace(trajectory, reasoning=rationale, metadata=metadata)

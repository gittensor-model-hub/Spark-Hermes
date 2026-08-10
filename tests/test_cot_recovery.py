import json

from teacher.cot_recovery import (
    extract_plaintext_reasoning_details,
    has_usable_training_reasoning,
    normalize_training_reasoning,
    recover_trajectory_cot,
)
from teacher.providers import Trajectory

# --- normalization ---------------------------------------------------------


def test_normalize_keeps_plaintext_reasoning():
    assert normalize_training_reasoning("  Sum the digits, then divide.  ") == "Sum the digits, then divide."


def test_normalize_drops_too_short_plaintext():
    assert normalize_training_reasoning("ok") is None


def test_normalize_returns_none_for_non_string():
    assert normalize_training_reasoning(None) is None
    assert normalize_training_reasoning({"type": "reasoning.encrypted"}) is None


def test_normalize_drops_encrypted_reasoning_details_dump():
    encrypted = json.dumps([{"type": "reasoning.encrypted", "data": "b64blob=="}])
    assert normalize_training_reasoning(encrypted) is None


def test_normalize_extracts_plaintext_from_reasoning_details():
    details = json.dumps(
        [
            {"type": "reasoning.encrypted", "data": "b64blob=="},
            {"type": "reasoning.text", "text": "First factor out the common term, then simplify."},
        ]
    )
    assert normalize_training_reasoning(details) == "First factor out the common term, then simplify."


def test_normalize_ignores_short_extracted_details():
    details = json.dumps([{"type": "reasoning.text", "text": "hi"}])
    assert normalize_training_reasoning(details) is None


def test_normalize_malformed_json_falls_back_to_plaintext():
    text = "[incomplete but clearly a plaintext rationale of sufficient length"
    assert normalize_training_reasoning(text) == text


def test_extract_reasoning_details_prefers_text_and_summary():
    details = [
        {"type": "reasoning.encrypted", "data": "x"},
        {"type": "reasoning.summary", "summary": "Use symmetry."},
        {"type": "reasoning.text", "text": "Then integrate."},
    ]
    assert extract_plaintext_reasoning_details(details) == "Use symmetry.\n\nThen integrate."


def test_has_usable_training_reasoning():
    assert has_usable_training_reasoning("A sufficiently long plaintext rationale.") is True
    assert has_usable_training_reasoning("") is False
    assert has_usable_training_reasoning(json.dumps([{"type": "reasoning.encrypted"}])) is False


# --- recovery orchestration ------------------------------------------------


class _FableStub:
    """Stand-in for a Claude Fable 5 teacher; returns a canned explanation."""

    name = "anthropic"
    model = "claude-fable-5"

    def __init__(self, response: str = "", reasoning: str | None = None) -> None:
        self._response = response
        self._reasoning = reasoning
        self.calls: list[str] = []

    def generate(self, prompt: str, **_kwargs) -> Trajectory:
        self.calls.append(prompt)
        return Trajectory(
            prompt=prompt,
            response=self._response,
            provider=self.name,
            model=self.model,
            reasoning=self._reasoning,
        )


def _openai_traj(reasoning=None, response="The answer is 42.") -> Trajectory:
    return Trajectory(
        prompt="What is the answer?", response=response, provider="openai", model="gpt-5.6", reasoning=reasoning
    )


def test_recovery_attaches_fable_rationale_when_gpt_has_no_cot():
    fable = _FableStub(response="Start from the definition, then it follows that the answer is 42.")
    traj = _openai_traj(reasoning=None)

    out = recover_trajectory_cot(traj, fable)

    assert fable.calls, "Fable should be asked to explain when GPT has no usable CoT"
    assert out.reasoning == "Start from the definition, then it follows that the answer is 42."
    assert out.response == "The answer is 42."  # GPT's answer is preserved
    assert out.metadata["cot_recovery"] == "fable_explain"
    assert out.metadata["cot_provider"] == "anthropic"
    assert out.metadata["cot_model"] == "claude-fable-5"


def test_recovery_marks_failed_when_fable_also_yields_nothing_usable():
    fable = _FableStub(response="no", reasoning=None)
    traj = _openai_traj(reasoning=None)

    out = recover_trajectory_cot(traj, fable)

    assert out.reasoning is None
    assert out.metadata["cot_recovery"] == "failed"


def test_recovery_falls_back_to_fable_extended_thinking():
    fable = _FableStub(response="", reasoning="Deriving it: the invariant forces the result to be 42.")
    traj = _openai_traj(reasoning=None)

    out = recover_trajectory_cot(traj, fable)

    assert out.reasoning == "Deriving it: the invariant forces the result to be 42."
    assert out.metadata["cot_recovery"] == "fable_explain"


def test_recovery_normalizes_existing_reasoning_without_calling_fable():
    fable = _FableStub(response="unused")
    encrypted = json.dumps(
        [
            {"type": "reasoning.encrypted", "data": "x"},
            {"type": "reasoning.text", "text": "The exposed part of the rationale is here."},
        ]
    )
    traj = _openai_traj(reasoning=encrypted)

    out = recover_trajectory_cot(traj, fable)

    assert fable.calls == []  # already had usable (extractable) reasoning
    assert out.reasoning == "The exposed part of the rationale is here."


def test_recovery_leaves_usable_plaintext_reasoning_untouched():
    fable = _FableStub(response="unused")
    traj = _openai_traj(reasoning="A perfectly good plaintext chain of thought.")

    out = recover_trajectory_cot(traj, fable)

    assert fable.calls == []
    assert out is traj


def test_recovery_skips_fable_own_trajectories():
    fable = _FableStub(response="unused")
    traj = Trajectory(prompt="p", response="answer", provider="anthropic", model="claude-fable-5", reasoning=None)

    out = recover_trajectory_cot(traj, fable)

    assert fable.calls == []
    assert out is traj


def test_recovery_skips_when_no_answer_to_explain():
    fable = _FableStub(response="unused")
    traj = _openai_traj(reasoning=None, response="")

    out = recover_trajectory_cot(traj, fable)

    assert fable.calls == []
    assert out is traj

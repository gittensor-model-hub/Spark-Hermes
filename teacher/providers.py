"""Pluggable teacher-model clients for trajectory generation.

Production teachers:
- Anthropic Claude Fable 5 (`claude-fable-5`), direct
- Qwen 3.8 Max 2.4T (`qwen3.8-max`), through the yunwu gateway
- Kimi K3 (`kimi-k3`), through the OpenRouter gateway
- OpenAI GPT 5.6 (`gpt-5.6`), direct

`gpt-5.6-sol` is retired for generation but still verifiable -- see RETIRED_OPENAI_MODELS.

Every slug here is *pinned*: SparkProof commits the exact model string into a bundle's
`request_sha256` and re-checks the gateway's response model against it, so these
constants must stay identical to `sparkproof/policy.py`. Drift does not surface at
generation time -- it surfaces as datasets that fail verification after the GPU hours
have already been spent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

ANTHROPIC_TEACHER_MODEL = "claude-fable-5"
OPENAI_TEACHER_MODEL = "gpt-5.6"

# `gpt-5.6-sol` is retired as a *generation* teacher: no new trajectory should be built
# with it. It is deliberately NOT removed from SparkProof's verifier allowlist
# (`sparkproof/policy.py: ALLOWED_MODELS`), because 23 already-merged registry entries
# were proved with it and `mix_registry --all` re-reads every one of them when it
# rebuilds the canonical mining dataset. Dropping the slug from the verifier would make
# historical bundles -- including the data behind the current frontier -- fail to
# re-verify. Stopping generation and revoking verification are different actions; only
# the first is wanted here.
RETIRED_OPENAI_MODELS = frozenset({"gpt-5.6-sol"})
_ALLOWED_OPENAI_MODELS = frozenset({OPENAI_TEACHER_MODEL})

# Qwen 3.8 Max (2.4T), served through the yunwu gateway -- yunwu speaks the OpenAI
# chat-completions protocol, so it reuses OpenAICompatibleTeacher with a base_url.
#
# WARNING: this slug is pinned. `request_sha256` in a SparkProof bundle commits the exact
# model string, and the gateway's response model is checked against it, so a slug that
# does not match what yunwu actually serves makes every bundle fail verification rather
# than fail loudly at generation time. Confirm against yunwu's /v1/models before a
# production run, and keep it identical to SparkProof's `YUNWU_DEFAULT_QWEN`.
QWEN_TEACHER_MODEL = "qwen3.8-max"
_ALLOWED_QWEN_MODELS = frozenset({QWEN_TEACHER_MODEL})

# Kimi K3 from Moonshot, via OpenRouter. The provider key is the vendor, matching
# `anthropic`/`claude-fable-5` and SparkProof's `sparkproof/policy.py`.
#
# Two distinct slugs, and conflating them breaks verification:
#   OPENROUTER_MODEL_MOONSHOT  addresses the model on the wire
#   MOONSHOT_TEACHER_MODEL     is the logical id recorded on the trajectory
# SparkProof normalizes the former to the latter and checks the record, so a trajectory
# stamped `moonshotai/kimi-k3` fails verification even though the call itself was fine.
MOONSHOT_TEACHER_MODEL = "kimi-k3"
OPENROUTER_MODEL_MOONSHOT = "moonshotai/kimi-k3"
_ALLOWED_MOONSHOT_MODELS = frozenset({MOONSHOT_TEACHER_MODEL})

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai", "qwen", "moonshot"})


def openrouter_api_base() -> str:
    """OpenRouter gateway base URL. Mirrors SparkProof's `sparkproof.gateways`."""
    return os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")


def yunwu_api_base() -> str:
    """yunwu gateway base URL. Mirrors SparkProof's `sparkproof.gateways.yunwu_api_base`."""
    return os.environ.get("YUNWU_API_BASE", "https://yunwu.ai/v1").rstrip("/")


@dataclass(frozen=True)
class Trajectory:
    """A single prompt/response pair captured from a teacher model.

    `reasoning` is the teacher's captured chain-of-thought/thinking trace, kept
    separate from the final `response` — SparkDistill trains students to reproduce
    reasoning, not just answers, so the raw trajectory must preserve that distinction
    even before any training-format decision is made (see `teacher/format.py`).
    `reasoning` is `None` when a teacher provides no capturable trace (e.g. GPT 5.6
    over chat-completions, which may not expose reasoning tokens).
    """

    prompt: str
    response: str
    provider: str
    model: str
    system: str | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "provider": self.provider,
            "model": self.model,
            "system": self.system,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


class TeacherClient(Protocol):
    """Anything that can turn a prompt into a captured `Trajectory`."""

    name: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,
    ) -> Trajectory: ...


class AnthropicTeacher:
    """Teacher backed by the Anthropic API (Claude Fable 5 only)."""

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_TEACHER_MODEL, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,
    ) -> Trajectory:
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        if thinking_budget is not None:
            # Anthropic requires max_tokens > thinking.budget_tokens, and extended
            # thinking is incompatible with a fixed sampling temperature.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            max_tokens = max(max_tokens, thinking_budget + 1024)
            temperature = 1.0
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        reasoning = "".join(block.thinking for block in message.content if block.type == "thinking") or None
        response = "".join(block.text for block in message.content if block.type == "text")
        return Trajectory(
            prompt=prompt,
            response=response,
            provider=self.name,
            model=self.model,
            system=system,
            reasoning=reasoning,
            metadata={"stop_reason": message.stop_reason, "usage": message.usage.model_dump()},
        )


class OpenAICompatibleTeacher:
    """Teacher backed by any OpenAI-compatible chat-completions endpoint.

    Serves both the OpenAI teacher (direct to api.openai.com) and the Qwen teacher
    (through the yunwu gateway, which speaks the same protocol). `name` is an instance
    attribute rather than a class one because it is the `provider` recorded on every
    trajectory, and SparkProof verifies the provider/model pair against its allowlist --
    a Qwen trajectory labelled `openai` would fail that check, correctly.
    """

    def __init__(
        self,
        model: str = OPENAI_TEACHER_MODEL,
        api_key: str | None = None,
        *,
        name: str = "openai",
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        request_model: str | None = None,
    ) -> None:
        import openai

        self.name = name
        # `model` is the logical id recorded on every trajectory and verified by
        # SparkProof; `request_model` is what the gateway is actually called with. They
        # differ on OpenRouter, which addresses models as `provider/model`.
        self.model = model
        self.request_model = request_model or model
        # `base_url=None` is the client's own default (api.openai.com), so the gateway
        # case needs no branch here. Passing it through a conditional **kwargs unpack
        # instead would defeat type checking -- pyright cannot see the keys and spreads
        # the value across every parameter.
        self._client = openai.OpenAI(
            api_key=api_key or os.environ[api_key_env],
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,  # not used for GPT 5.6 chat completions
    ) -> Trajectory:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        completion = self._client.chat.completions.create(
            model=self.request_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        choice = completion.choices[0]
        reasoning = getattr(choice.message, "reasoning_content", None)
        return Trajectory(
            prompt=prompt,
            response=choice.message.content or "",
            provider=self.name,
            model=self.model,
            system=system,
            reasoning=reasoning,
            metadata={
                "finish_reason": choice.finish_reason,
                "usage": completion.usage.model_dump() if completion.usage else {},
            },
        )


def get_teacher(provider: str, model: str | None = None) -> TeacherClient:
    """Construct a configured teacher client by provider name.

    Each provider is pinned to a single model (Fable 5 or GPT 5.6). Reads credentials
    from the environment (see `.env.example`).
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}, expected one of {sorted(_SUPPORTED_PROVIDERS)}")

    if provider == "anthropic":
        if model is not None and model != ANTHROPIC_TEACHER_MODEL:
            raise ValueError(f"anthropic teacher is fixed to {ANTHROPIC_TEACHER_MODEL!r}; got {model!r}")
        return AnthropicTeacher(model=ANTHROPIC_TEACHER_MODEL)

    if provider == "moonshot":
        if model is not None and model not in _ALLOWED_MOONSHOT_MODELS:
            raise ValueError(f"moonshot teacher is fixed to {MOONSHOT_TEACHER_MODEL!r}; got {model!r}")
        return OpenAICompatibleTeacher(
            model=MOONSHOT_TEACHER_MODEL,
            name="moonshot",
            base_url=openrouter_api_base(),
            api_key_env="OPENROUTER_API_KEY",
            request_model=OPENROUTER_MODEL_MOONSHOT,
        )

    if provider == "qwen":
        if model is not None and model not in _ALLOWED_QWEN_MODELS:
            raise ValueError(f"qwen teacher is fixed to {QWEN_TEACHER_MODEL!r}; got {model!r}")
        return OpenAICompatibleTeacher(
            model=QWEN_TEACHER_MODEL,
            name="qwen",
            base_url=yunwu_api_base(),
            api_key_env="YUNWU_API_KEY",
        )

    if model is not None and model in RETIRED_OPENAI_MODELS:
        raise ValueError(
            f"{model!r} is retired as a generation teacher and must not produce new "
            f"trajectories; use {OPENAI_TEACHER_MODEL!r}. Bundles already proved with it "
            "still verify -- retiring generation does not revoke past verification."
        )
    if model is not None and model not in _ALLOWED_OPENAI_MODELS:
        raise ValueError(
            f"openai teacher is fixed to {OPENAI_TEACHER_MODEL!r} "
            f"(alias {sorted(_ALLOWED_OPENAI_MODELS)}); got {model!r}"
        )
    return OpenAICompatibleTeacher(model=OPENAI_TEACHER_MODEL)

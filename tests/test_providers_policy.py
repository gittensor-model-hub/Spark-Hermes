"""Teacher-selection policy.

Which models may *generate* new trajectories. This is deliberately narrower than which
models may be *verified*: the verifier's allowlist lives in SparkProof
(`sparkproof/policy.py: ALLOWED_MODELS`) and still accepts retired slugs, because
already-merged bundles must keep verifying. Retiring a generation teacher and revoking
past verification are different actions.
"""

import pytest

from teacher.providers import (
    ANTHROPIC_TEACHER_MODEL,
    MOONSHOT_TEACHER_MODEL,
    OPENAI_TEACHER_MODEL,
    OPENROUTER_MODEL_MOONSHOT,
    QWEN_TEACHER_MODEL,
    RETIRED_OPENAI_MODELS,
    get_teacher,
    openrouter_api_base,
    yunwu_api_base,
)


def test_retired_openai_model_cannot_generate():
    """gpt-5.6-sol must not produce new trajectories."""
    with pytest.raises(ValueError, match="retired as a generation teacher"):
        get_teacher("openai", "gpt-5.6-sol")


def test_retired_model_error_explains_that_old_bundles_still_verify():
    """The message must not imply past data was invalidated."""
    with pytest.raises(ValueError, match="still verify"):
        get_teacher("openai", "gpt-5.6-sol")


def test_every_retired_model_is_rejected():
    for model in RETIRED_OPENAI_MODELS:
        with pytest.raises(ValueError, match="retired"):
            get_teacher("openai", model)


def test_retired_models_are_not_in_the_generation_allowlist():
    from teacher.providers import _ALLOWED_OPENAI_MODELS

    assert not (RETIRED_OPENAI_MODELS & _ALLOWED_OPENAI_MODELS)


def test_unknown_openai_model_is_rejected():
    with pytest.raises(ValueError, match="openai teacher is fixed"):
        get_teacher("openai", "gpt-4o-mini")


def test_anthropic_teacher_is_pinned():
    with pytest.raises(ValueError, match="anthropic teacher is fixed"):
        get_teacher("anthropic", "claude-3-opus")


def test_unsupported_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        get_teacher("mistral")


def test_live_slugs_are_still_the_pinned_ones():
    """Guards against a rename drifting away from what SparkProof verifies.

    These strings are committed into a bundle's `request_sha256`. If they drift from
    `sparkproof/policy.py`, generation still succeeds and verification fails afterwards
    -- after the GPU hours are spent.
    """
    assert ANTHROPIC_TEACHER_MODEL == "claude-fable-5"
    assert OPENAI_TEACHER_MODEL == "gpt-5.6"
    assert QWEN_TEACHER_MODEL == "qwen3.8-max"


def test_qwen_teacher_goes_through_the_yunwu_gateway(monkeypatch):
    monkeypatch.setenv("YUNWU_API_KEY", "test-key")
    teacher = get_teacher("qwen")
    assert teacher.name == "qwen"
    assert teacher.model == QWEN_TEACHER_MODEL
    assert "yunwu" in str(teacher._client.base_url)


def test_qwen_provider_is_recorded_not_openai(monkeypatch):
    """A Qwen trajectory labelled `openai` would fail SparkProof's provider/model check."""
    monkeypatch.setenv("YUNWU_API_KEY", "test-key")
    assert get_teacher("qwen").name == "qwen"


def test_qwen_teacher_is_pinned_to_one_slug(monkeypatch):
    monkeypatch.setenv("YUNWU_API_KEY", "test-key")
    with pytest.raises(ValueError, match="qwen teacher is fixed"):
        get_teacher("qwen", "qwen-turbo")


def test_qwen_uses_the_yunwu_key_not_the_openai_one(monkeypatch):
    monkeypatch.delenv("YUNWU_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(KeyError, match="YUNWU_API_KEY"):
        get_teacher("qwen")


def test_yunwu_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("YUNWU_API_BASE", "https://proxy.example/v1/")
    assert yunwu_api_base() == "https://proxy.example/v1"


def test_openai_teacher_still_goes_direct(monkeypatch):
    """Adding a gateway path must not silently reroute the existing teacher."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    teacher = get_teacher("openai")
    assert teacher.name == "openai"
    assert "yunwu" not in str(teacher._client.base_url)


# --- Kimi K3 (Moonshot, via OpenRouter) -------------------------------------------


def test_moonshot_teacher_goes_through_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    teacher = get_teacher("moonshot")
    assert teacher.name == "moonshot"
    assert "openrouter" in str(teacher._client.base_url)


def test_recorded_slug_is_bare_but_the_request_slug_is_prefixed(monkeypatch):
    """SparkProof verifies the recorded id; OpenRouter needs the prefixed one.

    Conflating them fails at verification rather than at call time -- the request
    succeeds and the resulting bundle is rejected afterwards.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    teacher = get_teacher("moonshot")
    assert teacher.model == MOONSHOT_TEACHER_MODEL == "kimi-k3"
    assert teacher.request_model == OPENROUTER_MODEL_MOONSHOT == "moonshotai/kimi-k3"


def test_moonshot_is_pinned_to_one_slug(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="moonshot teacher is fixed"):
        get_teacher("moonshot", "kimi-k2")


def test_moonshot_uses_the_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
        get_teacher("moonshot")


def test_openrouter_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://proxy.example/v1/")
    assert openrouter_api_base() == "https://proxy.example/v1"


def test_adding_moonshot_left_the_other_teachers_alone(monkeypatch):
    """A new gateway path must not silently reroute an existing teacher."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("YUNWU_API_KEY", "k")
    openai_teacher = get_teacher("openai")
    qwen_teacher = get_teacher("qwen")
    assert openai_teacher.request_model == openai_teacher.model == "gpt-5.6"
    assert "openrouter" not in str(openai_teacher._client.base_url)
    assert "yunwu" in str(qwen_teacher._client.base_url)

import json
from pathlib import Path

import pytest

import teacher.generate as generate
from teacher.generate import _iter_prompts
from teacher.providers import Trajectory


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_iter_prompts_yields_all_records_without_limit(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n{"prompt": "c"}\n')

    got = list(_iter_prompts(path, limit=None))

    assert [r["prompt"] for r in got] == ["a", "b", "c"]


def test_iter_prompts_skips_blank_lines(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n\n\n{"prompt": "b"}\n')

    got = list(_iter_prompts(path, limit=None))

    assert [r["prompt"] for r in got] == ["a", "b"]


def test_limit_counts_prompts_not_file_lines(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "a"}\n{"prompt": "b"}\n\n{"prompt": "c"}\n{"prompt": "d"}\n{"prompt": "e"}\n',
    )

    got = list(_iter_prompts(path, limit=3))

    assert [r["prompt"] for r in got] == ["a", "b", "c"]


def test_limit_larger_than_available_yields_everything(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n')

    got = list(_iter_prompts(path, limit=10))

    assert [r["prompt"] for r in got] == ["a", "b"]


class _SpyTeacher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = "pinned"

    def generate(self, prompt: str, **_kwargs) -> Trajectory:
        return Trajectory(prompt=prompt, response="ok", provider=self.name, model=self.model)


def _install_spy(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_get_teacher(provider: str, model: str | None = None) -> _SpyTeacher:
        calls.append((provider, model))
        return _SpyTeacher(provider)

    monkeypatch.setattr(generate, "get_teacher", fake_get_teacher)
    return calls


def test_model_flag_is_not_forwarded_to_get_teacher(monkeypatch, tmp_path):
    calls = _install_spy(monkeypatch)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')

    list(
        generate.generate_trajectories(
            prompts,
            ["anthropic", "openai"],
            max_tokens=16,
            temperature=0.0,
            limit=None,
            concurrency=1,
            thinking_budget=None,
        )
    )

    assert calls == [("anthropic", None), ("openai", None)]


class _FlakyTeacher:
    """Teacher that raises on a chosen set of prompts, succeeds otherwise."""

    def __init__(self, name: str, fail_prompts: set[str]) -> None:
        self.name = name
        self.model = "pinned"
        self._fail_prompts = fail_prompts

    def generate(self, prompt: str, **_kwargs) -> Trajectory:
        if prompt in self._fail_prompts:
            raise RuntimeError(f"simulated teacher API error for {prompt!r}")
        return Trajectory(prompt=prompt, response="ok", provider=self.name, model=self.model)


def test_one_failed_teacher_call_does_not_abort_the_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(generate, "get_teacher", lambda provider, model=None: _FlakyTeacher(provider, {"b"}))
    prompts = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n{"prompt": "c"}\n{"prompt": "d"}\n')

    got = list(
        generate.generate_trajectories(
            prompts,
            ["anthropic"],
            max_tokens=16,
            temperature=0.0,
            limit=None,
            concurrency=1,
            thinking_budget=None,
        )
    )

    # The failing prompt "b" is skipped; the other three still come through.
    assert sorted(t.prompt for t in got) == ["a", "c", "d"]


def test_run_fails_loudly_when_every_teacher_call_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(generate, "get_teacher", lambda provider, model=None: _FlakyTeacher(provider, {"a", "b"}))
    prompts = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n')

    with pytest.raises(RuntimeError, match="no trajectories were generated"):
        list(
            generate.generate_trajectories(
                prompts,
                ["anthropic"],
                max_tokens=16,
                temperature=0.0,
                limit=None,
                concurrency=1,
                thinking_budget=None,
            )
        )


def test_main_with_model_flag_does_not_crash(monkeypatch, tmp_path):
    calls = _install_spy(monkeypatch)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    out = tmp_path / "out.jsonl"

    rc = generate.main(["--prompts", str(prompts), "--out", str(out), "--model", "gpt-5.6"])

    assert rc == 0
    assert all(model is None for _, model in calls)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["provider"] for r in records} == {"anthropic", "openai"}

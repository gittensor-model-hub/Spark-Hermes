"""Offline CPU checks. Temporary scripted fixtures are never a training corpus or model."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment

from admin.artifacts import StageError, checkpoint_files, content_digest, file_digest, write_record
from admin.pipeline import Workspace, build_corpus
from admin.split import eval_task_ids
from admin.training import RECIPES, ROOT, prepare_training, train, training_recipe
from hermes.base_model import load as load_base
from hermes.protocol import DIALECTS
from hermes.trajectory import FINAL, TOOL_CALL, Step
from hermesbench.runner import LocalToolExecutor, run_episode
from hermesbench.tasks import Task


def fixture_corpus_metadata(workspace: Workspace, tasks: list[Task]) -> None:
    """Explicit CPU-only policy and source declarations; never production rights."""
    from hermesbench.tasks import load_suite
    from validator.persistence import state_identity

    state_identity(workspace.root, mode="fixture", namespace="cpu-operator")
    directory = workspace.tasks / "generated"
    directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (directory / f"{task.task_id}.yaml").write_text(yaml.safe_dump(asdict(task)))
    parsed = load_suite("generated", root=workspace.tasks)
    memberships = [
        {
            "task_id": t.task_id,
            "repository": "fixture/operator",
            "version": content_digest(asdict(t)),
            "family_id": t.task_id,
            "partition": "public-development",
            "exposure": ["public"],
        }
        for t in parsed
    ]
    subjects = ["task:" + m["version"] for m in memberships]
    subjects.append("sha256:" + file_digest(workspace.rollouts / "episodes.jsonl"))
    write_record(
        workspace.root / "data-policy.json",
        {
            "schema": "spark-data-policy-v1",
            "version": "cpu-fixture-v1",
            "origin": workspace.identity,
            "family_aliases": {m["family_id"]: m["family_id"] for m in memberships},
            "memberships": memberships,
            "rights": [
                {
                    "subject": s,
                    "license": "CPU-FIXTURE-ONLY",
                    "attribution": "scripted software fixture",
                    "training": True,
                    "derivatives": True,
                }
                for s in subjects
            ],
        },
    )
    workspace.record("rollout", {"fixture": True, "origin": workspace.identity, "model": "scripted-cpu-fixture"})


class FixtureTokenizer:
    """Exercise the real template without downloading a tokenizer; counts are test-only."""

    chat_template = ""

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        environment = ImmutableSandboxedEnvironment()

        def fail(message: str) -> None:
            raise StageError(message)

        environment.globals["raise_exception"] = fail
        return environment.from_string(self.chat_template).render(messages=messages, **kwargs)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(text.encode())


class FixturePolicy:
    """A deterministic software fixture which reads actual tool results."""

    dialect = DIALECTS["qwen35"]
    system = "CPU software fixture: read source.txt, add one, and write answer.txt."
    tokens_used = 100  # A fixture value, never represented as model usage.

    def __init__(self, *, correct: bool) -> None:
        self.correct = correct
        self.turn = 0

    def next_steps(self, task: Task, history: list[Step]) -> list[Step]:
        self.turn += 1
        if self.turn == 1:
            return [Step(kind=TOOL_CALL, tool="file_read", args={"path": "source.txt"}, call_id="read")]
        if self.turn == 2:
            value = int(history[-1].content) + (1 if self.correct else 2)
            return [
                Step(
                    kind=TOOL_CALL,
                    tool="file_write",
                    args={"path": "answer.txt", "content": f"{value}\n"},
                    call_id="write",
                )
            ]
        return [Step(kind=FINAL, content="Fixture complete.")]


def software_status() -> dict[str, Any]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("yaml", "jinja2", "transformers", "datasets", "huggingface_hub", "openai", "anthropic")
    }
    blockers = [f"missing CPU dependency: {name}" for name, found in dependencies.items() if not found]
    assets = [
        "hermes/base_model.json",
        "hermes/templates/chat-template-qwen35.jinja",
        "hermes/templates/chat-template-qwen35-4b.jinja",
        "hermesbench/harness/tools.json",
        "hermesbench/harness/system_prompt.txt",
    ]
    blockers.extend(f"missing runtime asset: {name}" for name in assets if not (ROOT / name).is_file())
    try:
        load_base()
        tasks = eval_task_ids()
        for name in ("stage-c-tools.yaml", "stage-d-preference.yaml", "rtx5090-poc.yaml"):
            config = yaml.safe_load((RECIPES / name).read_text())
            if not isinstance(config, dict) or not config.get("base_model"):
                raise StageError(f"invalid recipe: {name}")
    except (OSError, ValueError, RuntimeError) as exc:
        blockers.append(str(exc))
        tasks = ()
    return {
        "ready": not blockers,
        "scope": "software",
        "blockers": blockers,
        "dependencies": dependencies,
        "held_out_tasks": len(tasks),
        "training_readiness_checked": False,
    }


def run() -> dict[str, Any]:
    report = software_status()
    if not report["ready"]:
        return report
    checks = []
    with TemporaryDirectory(prefix="spark-hermes-selfcheck-") as scratch:
        workspace = Workspace(Path(scratch))
        task = Task(
            task_id="cpu-fixture-increment",
            prompt="Add one to the integer in source.txt and write answer.txt.",
            setup="printf '3\\n' > source.txt",
            verify="test -f answer.txt",
            hidden_verify='test "$(cat answer.txt)" = 4',
            tools=("file_read", "file_write"),
            max_steps=6,
            timeout_s=10,
        )
        workspace.record("generate", {"accepted": [task.task_id], "fixture": True})
        rows = []
        for correct in (True, False):
            result = run_episode(
                task,
                FixturePolicy(correct=correct),
                LocalToolExecutor(allow_unsandboxed=True),
                workspace.root / f"fixture-{correct}",
            )
            if result.metrics.success != correct or result.metrics.hidden_passed != correct:
                raise StageError("CPU fixture verification did not distinguish correct and incorrect results")
            rows.append(
                {
                    "metrics": result.metrics.to_record(),
                    "trajectory": result.trajectory.to_record(),
                    "integrity": result.integrity.to_record(),
                    "evidence": result.evidence,
                    "disqualified": result.integrity.disqualified,
                    "setup_failed": result.setup_failed,
                }
            )
        workspace.record("rollout", {"fixture": True})
        (workspace.rollouts / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        fixture_corpus_metadata(workspace, [task])
        checks.append("real CPU tool execution and public/withheld verification")
        workspace.record("corpus", build_corpus(workspace))
        checks.append("verified SFT and preference corpus construction")
        prepared = prepare_training(workspace, tokenizer=FixtureTokenizer(), sequence_len=65536)
        dry_run = train(workspace, stage="sft", dry_run=True)
        if not dry_run["dry_run"] or workspace.manifest_of("train") is not None:
            raise StageError("dry run incorrectly marked training complete")
        checks.append("SFT preparation and training command without a GPU")
        # Test-only checkpoint bytes exercise reference provenance, never an optimizer.
        merged = workspace.models / "sft/adapter/merged"
        merged.mkdir(parents=True)
        write_record(merged / "config.json", {"model_type": "cpu_test_fixture", "not_a_model": True})
        (merged / "model.safetensors").write_bytes(b"CPU_TEST_FIXTURE_NOT_MODEL_WEIGHTS")
        write_record(
            workspace.models / "sft/merged.json",
            {
                "merged": str(merged),
                "profile": prepared["profile"],
                "base_model": prepared["base_model"],
                "revision": prepared["revision"],
                "recipe_sha256": file_digest(Path(prepared["prepared_recipe"])),
                "files": checkpoint_files(merged),
            },
        )
        preference = prepare_training(workspace, stage="dpo", tokenizer=FixtureTokenizer(), sequence_len=65536)
        if not training_recipe(workspace, "dpo").is_file():
            raise StageError("DPO recipe was not created")
        pair = json.loads(Path(preference["data"]).read_text())
        if "<function=file_write>" not in pair["chosen"] or "<tool_response>" not in pair["chosen"]:
            raise StageError("DPO lost tool calls or results")
        checks.append("whole-trajectory DPO rendering and merged-reference binding")
        Path(prepared["data"]).write_text("tampered fixture")
        try:
            training_recipe(workspace, "sft")
        except StageError:
            checks.append("modified training data rejected")
        else:
            raise StageError("modified training data was accepted")
    return {
        **report,
        "checks": checks,
        "fixtures_removed": True,
        "model_training_executed": False,
        "token_counts": "test fixture values; real tokenizer validation occurs during prepare",
    }

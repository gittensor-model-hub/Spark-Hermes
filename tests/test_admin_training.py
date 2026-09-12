"""Exercise the operator handoff using the real corpus renderer and pinned Jinja template."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment

from admin.artifacts import checkpoint_files, file_digest, write_record
from admin.cli import main
from admin.pipeline import StageError, Workspace, build_corpus, evaluate_command, run_rollouts
from admin.training import prepare_training, training_recipe
from hermes.harness import derive_task_salt, salted_digest
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.tasks import Task


class Tokenizer:
    chat_template = ""

    def apply_chat_template(self, messages, **kwargs):
        env = ImmutableSandboxedEnvironment()

        def fail(message):
            raise ValueError(message)

        env.globals["raise_exception"] = fail
        return env.from_string(self.chat_template).render(messages=messages, **kwargs)

    def encode(self, text, **kwargs):
        # Deterministic CPU stand-in; production uses the pinned AutoTokenizer.
        return text.split()


def episode(success=True, hidden=True, task_id="gen-test", answer="done"):
    trajectory = AgentTrajectory(
        task="Repair the counter",
        system="Inspect the files and verify the result.",
        task_id=task_id,
        success=success,
        tools_available=("terminal",),
        metadata={"executed": True},
        steps=(
            Step(kind=THINKING, content="Inspect the counter first."),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "cat counter.txt"}, call_id="c0"),
            Step(kind=TOOL_RESULT, content="41", call_id="c0", ok=True),
            Step(kind=FINAL, content=answer),
        ),
    )
    return {
        "metrics": {
            "task_id": task_id,
            "public_passed": success,
            "hidden_passed": hidden,
            "tokens_used": 100,
            "dialect": "qwen35",
            "max_steps_hit": False,
            "success": success,
            "setup_failed": False,
            "disqualified": False,
            "integrity_clean": True,
            "integrity_fully_checked": True,
            "protocol_clean": True,
            "malformed_turns": 0,
            "tool_calls": 1,
            "steps": 4,
            "wall_time_s": 1.0,
            "verify_digest": "sha256:" + hashlib.sha256(b"true").hexdigest(),
        },
        "trajectory": trajectory.to_record(),
    }


def workspace(tmp_path, rows=None):
    ws = Workspace(tmp_path)
    ws.record("generate", {"accepted": ["gen-test"]})
    ws.record("rollout", {})
    (ws.rollouts / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (rows or [episode()])))
    from admin.selfcheck import fixture_corpus_metadata

    fixture_corpus_metadata(
        ws,
        [
            Task(
                task_id="gen-test",
                prompt="Repair the counter",
                verify="true",
                hidden_verify="true",
                tools=("terminal",),
            )
        ],
    )
    return ws


def test_corpus_to_training_preserves_calls_and_pin(tmp_path):
    ws = workspace(tmp_path)
    result = build_corpus(ws)
    ws.record("corpus", result)
    report = prepare_training(ws, tokenizer=Tokenizer())
    cfg = yaml.safe_load(Path(report["prepared_recipe"]).read_text())
    assert cfg["base_model"] == "Qwen/Qwen3.8-27B"
    assert len(cfg["base_model_revision"]) == 40
    assert cfg["chat_template"] == "jinja"
    assert "<function=" in cfg["chat_template_jinja"]
    assert "lora_model_dir" not in cfg
    assert cfg["sample_packing"] is False
    assert cfg["val_set_size"] == 0
    assert cfg["datasets"][0]["roles_to_train"] == ["assistant"]
    assert report["rows"] == 1
    assert training_recipe(ws, "sft").is_file()
    assert ws.manifest_of("train") is None


@pytest.mark.parametrize("hidden", [None, "true", 1])
def test_missing_or_malformed_withheld_result_is_refused(tmp_path, hidden):
    ws = workspace(tmp_path, [episode(hidden=hidden)])
    with pytest.raises(StageError, match="withheld-check"):
        build_corpus(ws)


def test_integrity_disqualified_success_is_not_imitation(tmp_path):
    row = episode()
    row["trajectory"]["success"] = False
    ws = workspace(tmp_path, [row])
    with pytest.raises((StageError, ValueError, RuntimeError), match="conflict|contradict"):
        build_corpus(ws)


@pytest.mark.parametrize("change", ["simulated", "unpriced", "eval", "dialect"])
def test_bad_training_evidence_is_refused(tmp_path, change):
    row = episode()
    if change == "simulated":
        row["trajectory"]["metadata"]["executed"] = False
    elif change == "unpriced":
        row["metrics"]["tokens_used"] = 0
    elif change == "eval":
        row["metrics"]["task_id"] = "fix-failing-test"
    else:
        row["metrics"]["dialect"] = "atem"
    with pytest.raises((StageError, ValueError, RuntimeError)):
        build_corpus(workspace(tmp_path, [row]))


def test_long_episode_is_refused_before_writing_recipe(tmp_path):
    ws = workspace(tmp_path)
    ws.record("corpus", build_corpus(ws))
    with pytest.raises(StageError, match="exceeds sequence_len"):
        prepare_training(ws, tokenizer=Tokenizer(), sequence_len=4)
    assert not (ws.models / "sft/train.yaml").exists()


@pytest.mark.parametrize("key", ["source", "data", "prepared_recipe"])
def test_modified_training_inputs_require_preparation_again(tmp_path, key):
    ws = workspace(tmp_path)
    ws.record("corpus", build_corpus(ws))
    report = prepare_training(ws, tokenizer=Tokenizer())
    with Path(report[key]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(StageError, match="changed"):
        training_recipe(ws, "sft")


def test_whole_preference_episode_keeps_tools_reasoning_and_results(tmp_path):
    ws = workspace(tmp_path, [episode(), episode(success=False, hidden=False, answer="wrong")])
    ws.record("corpus", build_corpus(ws))
    prepared = prepare_training(ws, tokenizer=Tokenizer())
    merged = ws.models / "sft/adapter/merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type":"test_fixture"}')
    (merged / "model.safetensors").write_bytes(b"test fixture")
    write_record(
        ws.models / "sft/merged.json",
        {
            "merged": str(merged),
            "files": checkpoint_files(merged),
            "recipe_sha256": file_digest(Path(prepared["prepared_recipe"])),
            **{key: prepared[key] for key in ("profile", "base_model", "revision")},
        },
    )
    report = prepare_training(ws, stage="dpo", tokenizer=Tokenizer())
    row = json.loads(Path(report["data"]).read_text())
    assert "<tools>" in row["prompt"]
    assert '"name": "terminal"' in row["prompt"]
    for side in ("chosen", "rejected"):
        assert "Inspect the counter first." in row[side]
        assert "<function=terminal>" in row[side]
        assert "<tool_response>\n41" in row[side]
    assert "wrong" in row["rejected"]
    cfg = yaml.safe_load(Path(report["prepared_recipe"]).read_text())
    assert cfg["base_model"] == str(merged)
    assert "base_model_revision" not in cfg
    assert cfg["datasets"][0]["type"]["chosen_format"] == "{chosen}"


def test_dpo_refuses_missing_merged_policy(tmp_path):
    ws = workspace(tmp_path, [episode(), episode(success=False, hidden=False, answer="wrong")])
    ws.record("corpus", build_corpus(ws))
    with pytest.raises(StageError, match="merged SFT"):
        prepare_training(ws, stage="dpo", tokenizer=Tokenizer())


def test_rollout_passes_generated_root_and_committed_checks(tmp_path, monkeypatch):
    salt = "a-private-test-salt"
    monkeypatch.setenv("HERMESBENCH_WITHHELD_SALT", salt)
    ws = Workspace(tmp_path)
    ws.record("generate", {"accepted": ["gen-test"]})
    task_dir = ws.tasks / "generated"
    task_dir.mkdir()
    (ws.tasks / "withheld").mkdir()
    hidden = "test -f answer.txt"
    (ws.tasks / "withheld/gen-test.sh").write_text(hidden)
    (task_dir / "gen-test.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "gen-test",
                "prompt": "Repair the counter",
                "tools": ["terminal"],
                "verify": "true",
                "hidden_verify_commitment": salted_digest(hidden, derive_task_salt(salt, "gen-test")),
            }
        )
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    run_rollouts(ws, repeats=2, base_url="http://localhost:8001/v1", model="test", runner=runner)
    command, kwargs = calls[0]
    assert command[command.index("--task-root") + 1] == str(ws.tasks)
    assert command[command.index("--dialect") + 1] == "qwen35"
    assert "--allow-unsandboxed" not in command
    assert kwargs["env"]["SPARKDISTILL_WITHHELD_ROOT"] == str(ws.tasks / "withheld")


def test_record_invalidates_dependent_completion(tmp_path):
    ws = Workspace(tmp_path)
    ws.record("corpus", {})
    ws.record("train", {})
    ws.record("evaluate", {})
    ws.record("corpus", {"summary": "rebuilt"})
    assert ws.manifest_of("train") is None
    assert ws.manifest_of("evaluate") is None


def test_cli_failure_does_not_mark_training_complete(tmp_path, capsys):
    assert main(["train", "--root", str(tmp_path)]) == 2
    assert "corpus" in capsys.readouterr().err
    assert Workspace(tmp_path).manifest_of("train") is None


def test_evaluation_command_uses_pin(tmp_path):
    command = evaluate_command(Workspace(tmp_path), base_url="http://localhost:8001/v1", model="candidate")
    assert command[command.index("--dialect") + 1] == "qwen35"
    assert command[command.index("--repeats") + 1] == "10"


def test_rtx5090_profile_uses_pinned_4b_and_records_smoke_steps(tmp_path):
    ws = workspace(tmp_path)
    ws.record("corpus", build_corpus(ws))
    report = prepare_training(ws, tokenizer=Tokenizer(), profile="rtx5090-poc", max_steps=10)
    cfg = yaml.safe_load(Path(report["prepared_recipe"]).read_text())
    assert cfg["adapter"] == "lora"
    assert cfg["base_model"] == "Qwen/Qwen3.5-4B"
    assert cfg["base_model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert cfg["load_in_4bit"] is False
    assert cfg["sequence_len"] == 2048
    assert cfg["micro_batch_size"] == 1
    assert cfg["max_steps"] == report["max_steps"] == 10
    assert report["profile"] == "rtx5090-poc"


def test_dry_run_needs_no_gpu_and_records_no_completion(tmp_path):
    ws = workspace(tmp_path)
    ws.record("corpus", build_corpus(ws))
    prepare_training(ws, tokenizer=Tokenizer(), profile="rtx5090-poc", max_steps=10)
    assert main(["train", "--root", str(tmp_path), "--dry-run"]) == 0
    assert ws.manifest_of("train") is None


def test_preprocessing_cache_changes_with_sequence_length(tmp_path):
    ws = workspace(tmp_path)
    ws.record("corpus", build_corpus(ws))
    first = prepare_training(ws, tokenizer=Tokenizer(), sequence_len=8192)
    first_cache = yaml.safe_load(Path(first["prepared_recipe"]).read_text())["dataset_prepared_path"]
    second = prepare_training(ws, tokenizer=Tokenizer(), sequence_len=4096)
    second_cache = yaml.safe_load(Path(second["prepared_recipe"]).read_text())["dataset_prepared_path"]
    assert first_cache != second_cache


@pytest.mark.parametrize("change", ["profile", "weights"])
def test_dpo_refuses_changed_merged_reference(tmp_path, change):
    # Establish the complete SFT -> merge -> DPO fixture, then alter its provenance.
    test_whole_preference_episode_keeps_tools_reasoning_and_results(tmp_path)
    ws = Workspace(tmp_path)
    if change == "weights":
        (ws.models / "sft/adapter/merged/model.safetensors").write_bytes(b"different CPU fixture")
    else:
        path = ws.models / "sft/merged.json"
        record = json.loads(path.read_text())
        record["profile"] = "rtx5090-poc"
        write_record(path, record)
    with pytest.raises(StageError, match="merged SFT checkpoint"):
        training_recipe(ws, "dpo")

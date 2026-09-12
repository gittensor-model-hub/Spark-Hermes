"""CPU-only project checks and the evaluation-to-promotion artifact contract."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from admin.artifacts import StageError, checkpoint_files, write_record
from admin.cli import main
from admin.evaluation import evaluate, read_serving
from admin.pipeline import Workspace
from admin.selfcheck import run, software_status
from admin.split import eval_task_ids
from hermes.harness import RunManifest, SuiteDigest, TaskResult
from hermes.promotion import PromotionError, load_run


def test_software_selfcheck_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert software_status()["ready"]
    result = run()
    assert result["ready"]
    assert result["model_training_executed"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", ["--dry-run", "--print-only", "--software-only", "--offline"])
def test_action_flags_cannot_be_silently_ignored(flag):
    with pytest.raises(SystemExit) as exc:
        main(["generate", flag])
    assert exc.value.code == 2


def test_heldout_id_is_read_from_yaml_content(tmp_path):
    (tmp_path / "different-filename.yml").write_text(
        "task_id: protected-id\nprompt: Solve the task\ntools: [terminal]\nverify: 'true'\n"
    )
    assert eval_task_ids(tmp_path) == ("protected-id",)


def test_checkpoint_missing_shard_is_refused(tmp_path):
    write_record(tmp_path / "config.json", {"model_type": "test_fixture"})
    (tmp_path / "model-1.safetensors").write_bytes(b"CPU test fixture")
    write_record(tmp_path / "model.safetensors.index.json", {"weight_map": {"layer": "model-2.safetensors"}})
    with pytest.raises(StageError, match="missing or unsafe weight shard"):
        checkpoint_files(tmp_path)


def serving_record():
    return {
        "precision": "bf16",
        "device": "CPU test fixture",
        "engine": "scripted test fixture",
        "temperature": 0,
        "top_p": 1,
        "max_model_len": 8192,
        "confidential_computing": False,
    }


def test_serving_zero_temperature_is_stated(tmp_path):
    path = tmp_path / "serving.json"
    write_record(path, serving_record())
    assert not read_serving(path).unstated()


@pytest.mark.parametrize("key,value", [("temperature", True), ("top_p", 0), ("max_model_len", 0)])
def test_invalid_serving_config_refused(tmp_path, key, value):
    path = tmp_path / "serving.json"
    write_record(path, {**serving_record(), key: value})
    with pytest.raises(StageError):
        read_serving(path)


def evaluation_fixture(tmp_path, monkeypatch, *, mismatch=False):
    from hermes.promotion import Serving

    ws = Workspace(tmp_path)
    monkeypatch.setattr("admin.evaluation.load_suite", lambda _: [SimpleNamespace(task_id="cpu-fixture")])
    monkeypatch.setattr("admin.evaluation.withheld_environment", lambda _: {})
    result = TaskResult(task_id="cpu-fixture", passed=True, hidden_passed=True, steps=3, tokens_used=100, tool_calls=1)
    manifest = RunManifest(
        model="test-model",
        suite=SuiteDigest("test", "test-suite-digest", 1, withheld_tasks=1),
        harness="test-harness-digest",
        results=(replace(result, tokens_used=101) if mismatch else result,) * 10,
    )

    def runner(command, **kwargs):
        assert command[command.index("--temperature") + 1] == "0.0"
        assert command[command.index("--top-p") + 1] == "1.0"
        write_record(ws.reports / "manifest.json", manifest.to_record())
        row = {"metrics": {**result.to_record(), "success": True}, "disqualified": False}
        (ws.reports / "episodes.jsonl").write_text((json.dumps(row) + "\n") * 10)
        return SimpleNamespace(returncode=0)

    return ws, runner, Serving.from_record(serving_record())


def test_evaluation_export_can_be_loaded_by_promotion(tmp_path, monkeypatch):
    ws, runner, serving = evaluation_fixture(tmp_path, monkeypatch)
    evaluate(ws, serving=serving, runner=runner, model="test-model", base_url="http://unused.invalid/v1")
    loaded = load_run(ws.reports / "run.json")
    assert loaded.manifest is not None
    assert loaded.manifest.harness == "test-harness-digest"
    assert len(loaded.episodes) == 10
    assert RunManifest.from_record(loaded.manifest.to_record()).to_record() == loaded.manifest.to_record()


@pytest.mark.parametrize("artifact", ["episodes.jsonl", "manifest.json"])
def test_promotion_refuses_changed_evaluation_artifacts(tmp_path, monkeypatch, artifact):
    ws, runner, serving = evaluation_fixture(tmp_path, monkeypatch)
    evaluate(ws, serving=serving, runner=runner, model="test-model", base_url="http://unused.invalid/v1")
    with (ws.reports / artifact).open("a") as stream:
        stream.write("\n")
    with pytest.raises(PromotionError, match="changed after evaluation"):
        load_run(ws.reports / "run.json")


def test_evaluation_refuses_manifest_log_disagreement(tmp_path, monkeypatch):
    ws, runner, serving = evaluation_fixture(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(StageError, match="disagree"):
        evaluate(ws, serving=serving, runner=runner, model="test-model", base_url="http://unused.invalid/v1")
    assert not (ws.reports / "run.json").exists()


def test_evaluation_requires_serving_record(tmp_path, capsys):
    assert main(["evaluate", "--root", str(tmp_path)]) == 2
    assert "--serving-config" in capsys.readouterr().err


def test_unpinnable_checkout_fails_before_model_requests(tmp_path, monkeypatch, capsys):
    from hermes.harness import HarnessError
    from hermesbench.runner import main as run_benchmark

    monkeypatch.delenv("SPARKDISTILL_WITHHELD_ROOT", raising=False)
    monkeypatch.setenv("HERMESBENCH_WITHHELD_SALT", "cpu-test-private-salt")

    def unpinnable(**kwargs):
        raise HarnessError("working tree is dirty")

    def unexpected(**kwargs):
        pytest.fail("model client was created before pin validation")

    monkeypatch.setattr("hermes.pin.build_pin", unpinnable)
    monkeypatch.setattr("hermesbench.policy.openai_completion", unexpected)
    assert (
        run_benchmark(
            [
                "--model",
                "test",
                "--base-url",
                "http://unused.invalid/v1",
                "--workspace-root",
                str(tmp_path / "workspace"),
                "--out",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 2
    )
    assert "before execution" in capsys.readouterr().err
    assert not (tmp_path / "manifest.json").exists()

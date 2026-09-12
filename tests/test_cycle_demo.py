"""Connected installed-command semantics; CPU fixtures are only external inputs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from admin.artifacts import StageError, read_record
from admin.cycle_demo import main
from admin.cycles import CycleController
from admin.replay import ReplayStore
from admin.task_feedback import verify_feedback_task
from validator.settlement import SettlementStore
from validator.store import RoundStore


def test_demo_requires_explicit_fixture_before_creating_state(tmp_path):
    for extra in ([], ["--mode", "production"]):
        with pytest.raises(SystemExit) as exc:
            main(["--root", str(tmp_path / "absent"), *extra])
        assert exc.value.code == 2
        assert not (tmp_path / "absent").exists()


def test_connected_cycle_demo_and_resume(tmp_path):
    root = tmp_path / "demo"
    command = [sys.executable, "-m", "admin.cli", "cycle", "demo", "--root", str(root), "--mode", "fixture"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["first"]["status"] == "complete"
    assert summary["second"]["status"] == "refused"
    assert summary["second"]["incumbent"] == summary["first"]["incumbent"]
    assert summary["second"]["incumbent"]["generation"] == 1
    assert not summary["trained"] and not summary["measured_learning"] and not summary["live_rewards"]
    assert summary["second_parent"]["decision"]["strict"] is True
    controller = CycleController(root / "releases")
    states = [controller.status(summary[name]["id"]) for name in ("first", "second")]
    for state, count in zip(states, (2, 3)):
        jobs = {j["stage"]: j for j in state["jobs"]}
        assert len(jobs["replay"]["output"]["value"]["inputs"]) == count
        assert jobs["train"]["output"]["value"]["fixture_only"] is True
        assert jobs["train"]["output"]["value"]["returncode"] == 0
        assert jobs["prepare"]["output"]["value"]["base_model"] == "Qwen/Qwen3.5-4B"
        assert all(j["attempts"] == 1 for j in state["jobs"])
    assert summary["second_parent"]["decision"]["candidate"] == controller.active_pair()["model"]
    source = RoundStore(root / "source/rounds")
    second = source.load("round-two")
    assert second.challenge.epoch["incumbent"] == controller.epoch_binding(controller.active_pair())
    assert second.challenge.task_pins["feedback"]["request"] == summary["feedback"]["payload"]["request"]
    replay = ReplayStore(root / "replay-1")
    verify_feedback_task(replay, summary["feedback"]["id"])
    actions = SettlementStore(root / "source/settlement").actions()
    assert actions and all(a["status"] == "pending" for a in actions)
    for round_id in ("round-bootstrap", "round-one", "round-two"):
        outcome = SettlementStore(root / "source/settlement").record(round_id)["outcome"]
        assert outcome["crowned"] and outcome["winner"]["miner_id"] == "alice"
    assert summary["feedback"]["payload"]["request"]["category"] == "withheld_generalization"
    before = [[(j["id"], j["attempts"]) for j in s["jobs"]] for s in states]
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["first"] == summary["first"]
    assert before == [[(j["id"], j["attempts"]) for j in controller.status(s["id"])["jobs"]] for s in states]
    commands = [json.loads(line) for line in (root / "commands.jsonl").read_text().splitlines()]
    assert any("prepare" in r["argv"] and r["exit_code"] == 0 for r in commands)
    assert sum(r["exit_code"] == 3 for r in commands) == 2
    assert len({r["pid"] for r in commands}) == len(commands)
    # Altering a generated task's original bytes invalidates the authoritative handoff.
    generated = Path(summary["feedback"]["payload"]["files"][0]["path"])
    generated.write_text(generated.read_text() + "\n# changed\n")
    with pytest.raises(StageError, match="changed"):
        verify_feedback_task(replay, summary["feedback"]["id"])
    assert read_record(root / "summary.json")["fixture_only"] is True

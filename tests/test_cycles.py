"""Real controller recovery, immutable inputs, namespace isolation and epoch storage."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cycle_support import run_cycle, setup_cycle
from release_support import successor

from admin.artifacts import StageError, read_record, write_record
from admin.cycles import CycleController
from admin.evaluation import execute_crossed
from admin.release import ReleaseAuthority


@pytest.fixture
def cycle_setup(tmp_path):
    return setup_cycle(tmp_path)


def crash_at(boundary):
    def crash(name):
        if name == boundary:
            raise RuntimeError("injected crash: " + boundary)

    return crash


def job(state, stage):
    return next(j for j in state["jobs"] if j["stage"] == stage)


def test_real_cycle_prepared_dry_run_restart_success(cycle_setup):
    controller, spec, initial = cycle_setup
    first = controller.start("first", spec)
    state = controller.resume(first["id"])
    assert state["status"] == "pending" and state["next_stage"] == "train"
    assert not state["trained"] and controller.dry_run(first["id"])["dry_run"]
    with controller.store.connect() as db:
        assert db.execute("SELECT count(*) FROM cycle_external_jobs").fetchone()[0] == 0
    restarted = CycleController(controller.root)
    final = restarted.resume(first["id"], execute_training=True, allow_unsandboxed=True)
    assert final["status"] == "complete" and not final["trained"] and final["fixture_only"]
    assert job(final, "train")["output"]["value"]["execution_status"] == "fixture-complete"
    assert final["incumbent"]["generation"] == 1
    assert final["incumbent"]["candidate"] == job(final, "candidate")["output"]["value"]["id"]
    assert final["incumbent"]["candidate"] != initial[0]["id"]
    assert job(final, "decide")["output"]["value"]["payload"]["report"]["eligible"]
    assert restarted.start("first", spec)["id"] == first["id"]
    assert restarted.resume(first["id"]) == final
    ids = job(final, "experience")["output"]["value"]["ids"]
    assert job(final, "curriculum")["output"]["value"]["settled_inputs"] == ids
    assert job(final, "replay")["output"]["value"]["inputs"] == ids
    assert len({j["id"] for j in final["jobs"]}) == 13
    assert restarted.active_pair()["agent_record"]["system"] == "CPU fixture agent 1"


@pytest.mark.parametrize(
    "stage",
    [
        "admission",
        "settlement",
        "experience",
        "replay",
        "curriculum",
        "prepare",
        "train",
        "merge",
        "candidate",
        "plan",
        "evaluate",
        "decide",
        "activate",
    ],
)
def test_after_producer_crash_reconciles_same_job(cycle_setup, stage):
    controller, spec, _ = cycle_setup
    cycle = controller.start("crash", spec)
    with pytest.raises(RuntimeError, match="injected crash"):
        controller.resume(
            cycle["id"],
            through=stage,
            execute_training=True,
            allow_unsandboxed=True,
            hook=crash_at(stage + ":after_producer"),
        )
    before = controller._jobs(cycle["id"])
    state = CycleController(controller.root).resume(
        cycle["id"], through=stage, execute_training=True, allow_unsandboxed=True
    )
    assert job(state, stage)["status"] == "complete"
    assert job(state, stage)["id"] == before[stage]["id"]
    with controller.store.connect() as db:
        assert db.execute("SELECT count(*) FROM cycle_external_jobs WHERE status='complete'").fetchone()[0] <= 2
        assert db.execute("SELECT count(*) FROM experiment_runs").fetchone()[0] <= 1
    assert state["incumbent"]["generation"] == (1 if stage == "activate" else 0)


@pytest.mark.parametrize(
    "boundary",
    [
        "train:after_submit",
        "train:after_launch",
        "activate:after_pointer",
        "activate:before_commit",
        "activate:after_commit",
    ],
)
def test_launch_and_atomic_activation_crashes(cycle_setup, boundary):
    controller, spec, _ = cycle_setup
    cycle = controller.start("launch", spec)
    with pytest.raises(RuntimeError, match="injected crash"):
        controller.resume(cycle["id"], execute_training=True, allow_unsandboxed=True, hook=crash_at(boundary))
    if boundary == "train:after_launch":
        from admin.cycle_jobs import run_job

        run_job(controller.root, controller._jobs(cycle["id"])["train"]["id"])
    current = controller.release.status()
    assert len(current["history"]) == current["generation"] + 1
    if boundary in {"activate:after_pointer", "activate:before_commit"}:
        assert current["generation"] == 0
    final = CycleController(controller.root).resume(cycle["id"], execute_training=True, allow_unsandboxed=True)
    assert final["status"] == "complete" and final["incumbent"]["generation"] == 1


def test_changed_input_and_missing_completed_artifact_refuse(cycle_setup):
    controller, spec, _ = cycle_setup
    cycle = controller.start("immutable", spec)
    different = json.loads(json.dumps(spec))
    different["training"]["max_steps"] = 2
    with pytest.raises(StageError, match="different immutable inputs"):
        controller.start("immutable", different)
    controller.resume(cycle["id"], through="prepare")
    artifact = controller.workspace(cycle["id"]).corpus / "sft.jsonl"
    original = artifact.read_bytes()
    artifact.unlink()
    with pytest.raises((StageError, OSError)):
        controller.resume(cycle["id"])
    artifact.write_bytes(original)
    Path(spec["agent"]).write_text("changed")
    with pytest.raises(StageError, match="artifact"):
        controller.resume(cycle["id"])
    assert controller.release.status()["generation"] == 0


def test_concurrent_resume_runs_producers_once(cycle_setup):
    controller, spec, _ = cycle_setup
    identifier = controller.start("concurrent", spec)["id"]

    def resume(_):
        return CycleController(controller.root).resume(identifier, through="train", execute_training=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        states = list(pool.map(resume, range(3)))
    assert all(job(s, "train")["status"] == "complete" for s in states)
    assert all(j["attempts"] == 1 for j in states[0]["jobs"])
    with controller.store.connect() as db:
        assert db.execute("SELECT count(*) FROM cycle_external_jobs").fetchone()[0] == 1


def test_production_rejects_fixture_sources_inputs_and_activation(cycle_setup, tmp_path):
    controller, spec, _ = cycle_setup
    production = CycleController(tmp_path / "production")
    assert production.identity["mode"] == "production"
    with pytest.raises(StageError, match="trust mode/namespace"):
        production.release.configure(
            candidates=controller.release.candidates().store.root,
            incumbent=controller.release.status()["candidate"],
            data_policy=tmp_path / "confirmation.json",
            policy=controller.release.configuration()["policy"],
        )
    with pytest.raises(ValueError, match="immutable"):
        CycleController(controller.root, mode="production")
    final = run_cycle(controller, spec)
    decision = job(final, "decide")["output"]["value"]
    with pytest.raises(StageError, match="no committed authority"):
        production.release.activate(decision["id"])
    copied = production.store.put("release-decision", decision["payload"])
    with pytest.raises(StageError, match="accepted strict"):
        production.release.activate(copied["id"])


def test_refused_cycle_keeps_incumbent(cycle_setup):
    controller, spec, initial = cycle_setup
    script = read_record(Path(spec["evaluation"]["fixture"]))
    script["cells"]["Q11"] = script["cells"]["Q00"]
    write_record(Path(spec["evaluation"]["fixture"]), script)
    final = run_cycle(controller, spec)
    assert final["status"] == "refused"
    assert final["incumbent"]["candidate"] == initial[0]["id"] and final["incumbent"]["generation"] == 0
    assert "activate" not in [j["stage"] for j in final["jobs"]]
    assert controller.resume(final["id"]) == final


def test_failed_supervisor_adapter_is_not_completion(cycle_setup, monkeypatch):
    from admin import cycle_jobs

    controller, spec, _ = cycle_setup
    cycle = controller.start("uncertain-training", spec)
    with pytest.raises(RuntimeError, match="injected crash"):
        controller.resume(cycle["id"], through="train", execute_training=True, hook=crash_at("train:after_submit"))
    original = cycle_jobs.train

    def lost_completion(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("supervisor died after writing the adapter")

    monkeypatch.setattr(cycle_jobs, "train", lost_completion)
    job_id = controller._jobs(cycle["id"])["train"]["id"]
    with pytest.raises(RuntimeError, match="supervisor died"):
        cycle_jobs.run_job(controller.root, job_id)
    state = controller.resume(cycle["id"], execute_training=True)
    assert state["status"] == "pending" and not state["trained"]
    assert (controller.workspace(cycle["id"]).models / "sft/adapter/adapter_model.safetensors").is_file()
    with controller.store.connect() as db:
        row = db.execute("SELECT status,completion FROM cycle_external_jobs WHERE id=?", (job_id,)).fetchone()
    assert row == ("failed", None)


def test_additional_imports_do_not_expand_explicit_replay(tmp_path):
    from learning_support import NAMESPACE, policy_config, produce_round

    from admin.curriculum import build_curriculum
    from admin.pipeline import Workspace
    from admin.replay import ReplayStore
    from validator.persistence import state_identity

    a = produce_round(tmp_path / "source", round_id="r-a")
    b = produce_round(tmp_path / "source", round_id="r-b", task_id="gen-second")
    config, _ = policy_config(tmp_path, [a, b])
    replay = ReplayStore(tmp_path / "replay", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    ids = [replay.import_round("source", "r-a")["id"]]
    replay.import_round("source", "r-b")
    state_identity(tmp_path / "workspace", mode="fixture", namespace=NAMESPACE)
    ws = Workspace(tmp_path / "workspace")
    with pytest.raises(StageError, match="distinct"):
        replay.freeze(ws, identifiers=ids + ids)
    assert replay.freeze(ws, identifiers=ids)["inputs"] == ids
    curriculum = build_curriculum(
        replay,
        identifiers=ids,
        config={"version": "spark-curriculum-v1", "count": 1, "min_families": 1, "max_per_family": 1},
    )
    assert curriculum["settled_inputs"] == ids and set(curriculum["families"]) == {"family-0"}


@pytest.mark.parametrize("through", ["open", "graded"])
def test_controller_calls_admission_and_settlement_producers(cycle_setup, tmp_path, through):
    from learning_support import NAMESPACE, policy_config, produce_round

    from admin.replay import ReplayStore

    old, spec, initial = cycle_setup
    source_root = tmp_path / "fresh"
    produced = produce_round(source_root, round_id="fresh", through=through)
    if through == "graded":
        config, _ = policy_config(source_root, [produced])
    else:
        config = {
            "policy": str(tmp_path / "policy.json"),
            "sources": [
                {
                    "name": "fresh",
                    "rounds": str(produced["store"].root),
                    "settlement": str(produced["settlement"].root),
                    "intake": str(produced["intake"].root),
                    "receipts": str(produced["intake"].receipts),
                }
            ],
        }
    replay = ReplayStore(tmp_path / "fresh-replay", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    controller = CycleController(tmp_path / "fresh-release", mode="fixture", namespace=NAMESPACE)
    release_config = old.release.configuration()
    controller.release.configure(
        candidates=old.release.candidates().store.root,
        incumbent=initial[0]["id"],
        data_policy=Path(release_config["data_policy"]["path"]),
        policy=release_config["policy"],
    )
    bootstrap = old.configuration()["bootstrap_parent"]
    controller.configure(replay=replay.root, bootstrap_parent=bootstrap, github=old.configuration()["github"])
    fixture = read_record(Path(spec["evaluation"]["fixture"]))
    fixture["origin"] = controller.identity
    write_record(tmp_path / "fresh-serving.json", fixture)
    spec["evaluation"]["fixture"] = str(tmp_path / "fresh-serving.json")
    pr = spec["rounds"][0]["prs"][0]
    spec["rounds"] = [
        {
            "source": "fresh",
            "round_id": "fresh",
            "prs": [pr],
            "scorecards": str(source_root / "cards"),
            "episodes": str(source_root / "episodes"),
        }
    ]
    cycle = controller.start("producer-call", spec)
    result = controller.resume(cycle["id"], through="settlement", github_transport=produced.get("transport"))
    assert (
        produced["store"].load("fresh").admissions["alice"]["admission_id"]
        == job(result, "admission")["output"]["value"]["admissions"][0]["record"]["admission_id"]
    )
    if through == "open":
        assert result["status"] == "pending" and result["next_stage"] == "settlement"
    else:
        assert job(result, "settlement")["status"] == "complete"
        assert produced["settlement"].record("fresh")["entries"]


def test_concurrent_activations_stale_epoch_and_rollback_replay(cycle_setup, tmp_path):
    controller, spec, _ = cycle_setup
    final = run_cycle(controller, spec)
    first_approval = job(final, "decide")["output"]["value"]["id"]
    prior = controller.release.candidates().resolve(final["incumbent"]["candidate"])
    candidate, freeze = successor(tmp_path, controller.release, prior, {"id": first_approval}, index=2, profile="bf16")
    plan = controller.release.freeze(**freeze)
    evaluated = execute_crossed(controller.release, plan["id"], allow_unsandboxed=True)
    approval = controller.release.decide(evaluated["id"])["id"]

    def activate(_):
        return ReleaseAuthority(controller.root).activate(approval)

    with ThreadPoolExecutor(max_workers=3) as pool:
        states = list(pool.map(activate, range(3)))
    assert all(s["candidate"] == candidate["id"] and s["generation"] == 2 for s in states)
    current = controller.release.status()
    kwargs = {
        "operation_id": "operator-rollback-1",
        "expected_generation": current["generation"],
        "expected_candidate": current["candidate"],
    }
    rollback = controller.rollback(first_approval, **kwargs)
    assert rollback["generation"] == 3 and len(rollback["history"]) == 4
    controller.release.activate(
        approval,
        rollback=True,
        operation_id="operator-rollback-2",
        expected_generation=3,
        expected_candidate=prior["id"],
    )
    replayed = controller.rollback(first_approval, **kwargs)
    assert replayed["generation"] == 4 and replayed["candidate"] == candidate["id"]
    with pytest.raises(StageError, match="stale"):
        controller.rollback(first_approval, **{**kwargs, "operation_id": "stale-rollback"})
    assert [h["generation"] for h in replayed["history"]] == list(range(5))


def test_second_cycle_uses_first_strict_parent_and_refusal_keeps_pair(cycle_setup, tmp_path):
    from learning_support import NAMESPACE, policy_config, produce_round

    from admin.replay import ReplayStore

    controller, spec, _ = cycle_setup
    first = run_cycle(controller, spec)
    active = controller.active_pair()
    original_epoch = controller.epoch_binding(active)
    new_root = tmp_path / "round-two"
    produced = produce_round(new_root, round_id="r-2", incumbent=original_epoch)
    config, _ = policy_config(new_root, [produced])
    replay = ReplayStore(tmp_path / "replay-two", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    workload = read_record(Path(spec["workload"]))
    fixture = read_record(Path(spec["evaluation"]["fixture"]))
    for index, item in enumerate(workload["tasks"]):
        previous = item["task"]["task_id"]
        item["task"]["task_id"] = f"confirm-{index + 6}"
        for cell in fixture["cells"].values():
            cell[item["task"]["task_id"]] = cell.pop(previous)
    fixture["cells"]["Q11"] = fixture["cells"]["Q00"]
    agent = read_record(Path(spec["agent"]))
    agent["system"] = "CPU fixture second candidate"
    for name, value in (("workload-two", workload), ("serving-two", fixture), ("agent-two", agent)):
        write_record(tmp_path / (name + ".json"), value)
    admission = produced["admission"]
    second_spec = {
        **spec,
        "replay": str(replay.root),
        "workload": str(tmp_path / "workload-two.json"),
        "agent": str(tmp_path / "agent-two.json"),
        "rounds": [
            {
                "source": "round-two",
                "round_id": "r-2",
                "prs": [{"number": admission["pr_number"], "head": admission["head_sha"], "author": "alice"}],
                "scorecards": str(new_root / "cards"),
                "episodes": str(new_root / "episodes"),
            }
        ],
        "evaluation": {**spec["evaluation"], "fixture": str(tmp_path / "serving-two.json")},
    }
    restarted = CycleController(controller.root)
    second = run_cycle(restarted, second_spec, "cycle-two")
    assert second["status"] == "refused"
    assert restarted.epoch_binding(restarted.active_pair()) == original_epoch
    prepared = job(second, "prepare")["output"]["value"]
    assert prepared["parent_approval"] == active["approval"]
    assert prepared["parent"]["decision"]["candidate"] == active["model"]
    assert prepared["parent"]["decision"]["strict"] is True
    assert restarted.status(first["id"])["status"] == "complete"


def test_installed_shape_cli_start_resume_status(cycle_setup, tmp_path):
    controller, _, _ = cycle_setup
    commands = []

    def cli(*args, expected=0):
        command = [sys.executable, "-m", "admin.cli", "cycle", *map(str, args), "--root", str(controller.root)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        commands.append({"argv": command, "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        write_record(tmp_path / "cycle-commands.json", {"commands": commands})
        assert result.returncode == expected, result.stderr
        return json.loads(result.stdout)

    cli("init", "--config", tmp_path / "cycle-config.json")
    cycle = cli("start", "--name", "cli-cycle", "--spec", tmp_path / "cycle-spec.json")
    cli("resume", "--id", cycle["id"], expected=4)
    assert cli("dry-run", "--id", cycle["id"])["trained"] is False
    final = cli("resume", "--id", cycle["id"], "--execute-training", "--allow-unsandboxed")
    assert final["status"] == "complete"
    assert cli("status", "--id", cycle["id"])["incumbent"]["generation"] == 1
    assert cli("active")["candidate"] == final["incumbent"]["candidate"]
    assert (controller.root / "authority.sqlite3").is_file()

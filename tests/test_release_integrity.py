"""Real release producers and installed-compatible CPU boundary regressions."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from release_support import setup_release

from admin.artifacts import StageError, content_digest, read_record, write_record
from admin.candidates import file_identity
from admin.cycle_jobs import FixtureTraining
from admin.evaluation import execute_crossed
from admin.parents import ParentAuthority
from admin.pipeline import Workspace
from admin.release import ReleaseAuthority
from admin.replay import ReplayStore
from admin.selfcheck import FixtureTokenizer
from admin.serving_identity import TrustedServing, validate_completion_usage
from admin.training import merge, prepare_training, train, training_recipe
from hermesbench.tasks import Task
from validator.persistence import state_identity


def producer_candidate(root, *, layout="indexed"):
    authority, candidates, _, freeze = setup_release(root)
    bootstrap = ParentAuthority(root / "bootstrap-parent")
    approval = bootstrap.store.records(kind="release-decision")[0]
    state_identity(root / "child", mode="fixture", namespace=authority.identity["namespace"])
    ws = Workspace(root / "child")
    ReplayStore(root / "replay").freeze(ws)
    prepare_training(
        ws,
        profile="rtx5090-poc",
        tokenizer=FixtureTokenizer(),
        sequence_len=65536,
        parent_approval=approval["id"],
        release_root=bootstrap.store.root,
    )
    executor = FixtureTraining(ws, "CPU checkpoint layout boundary")
    train(ws, stage="sft", fixture_executor=executor)

    def external_merge(argv, **kwargs):
        result = executor(argv, **kwargs)
        directory = ws.models / "sft/adapter/merged"
        if layout in {"partial", "indexed"}:
            name = "model-00001-of-00002.safetensors"
            (directory / "model.safetensors").rename(directory / name)
            if layout == "indexed":
                other = "model-00002-of-00002.safetensors"
                (directory / other).write_text("CPU fixture second shard")
                write_record(directory / "model.safetensors.index.json", {"weight_map": {"a": name, "b": other}})
        elif layout in {"nonstandard", "ambiguous"}:
            (directory / "untracked.bin").write_text("CPU fixture referenced shard")
            write_record(directory / "pytorch_model.bin.index.json", {"weight_map": {"a": "untracked.bin"}})
            if layout == "nonstandard":
                (directory / "model.safetensors").unlink()
        return result

    merge(ws, stage="sft", fixture_executor=external_merge)
    new = authority.candidates().register(
        workspace=ws,
        merged_record=ws.models / "sft/merged.json",
        agent=root / "agent-1.json",
        workload=root / "workload.json",
        parent=Path(candidates[0]["payload"]["model"]["merged"]),
    )
    config = freeze["serving"].pop(candidates[1]["payload"]["model_id"])
    path = Path(config["fixture"]["path"])
    script = read_record(path)
    script["model_id"] = new["payload"]["model_id"]
    write_record(path, script)
    freeze["serving"][new["payload"]["model_id"]] = {"fixture": file_identity(path)}
    freeze["new"] = new["id"]
    return authority, [candidates[0], new], ws, freeze


@pytest.mark.parametrize("layout", ["partial", "ambiguous"])
def test_actual_merge_refuses_original_incomplete_layouts_before_publication(tmp_path, layout):
    with pytest.raises(StageError, match="unindexed|ambiguous"):
        producer_candidate(tmp_path, layout=layout)
    assert not (tmp_path / "child/models/sft/merged.json").exists()
    # The failed external output remains available for diagnostics.
    assert (tmp_path / "child/models/sft/adapter/merged/config.json").exists()


@pytest.mark.parametrize("layout", ["indexed", "nonstandard"])
def test_complete_checkpoint_release_and_changed_shard_activation(tmp_path, layout):
    authority, candidates, ws, freeze = producer_candidate(tmp_path, layout=layout)
    plan = authority.freeze(**freeze)
    evidence = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    decision = authority.decide(evidence["id"])
    assert decision["payload"]["result"] == "accepted"
    model = candidates[1]["payload"]["model"]
    name = "untracked.bin" if layout == "nonstandard" else "model-00002-of-00002.safetensors"
    assert name in model["files"]
    path = Path(model["merged"]) / name
    original = path.read_bytes()
    before = authority.status()
    path.write_bytes(b"CPU fixture replaced referenced shard")
    try:
        for resolve in [
            lambda: authority.candidates().resolve(candidates[1]["id"]),
            lambda: authority.resolve_decision(decision["id"]),
            lambda: authority.activate(decision["id"]),
        ]:
            with pytest.raises(StageError, match="checkpoint|artifacts"):
                resolve()
        assert authority.status() == before
    finally:
        path.write_bytes(original)
    assert authority.activate(decision["id"])["candidate"] == candidates[1]["id"]
    second = Workspace(tmp_path / "next")
    state_identity(second.root, mode="fixture", namespace=authority.identity["namespace"])
    ReplayStore(tmp_path / "replay").freeze(second)
    prepare_training(
        second,
        profile="rtx5090-poc",
        tokenizer=FixtureTokenizer(),
        sequence_len=65536,
        parent_approval=decision["id"],
        release_root=authority.store.root,
    )
    path.unlink()
    try:
        with pytest.raises(StageError):
            training_recipe(second, "sft")
    finally:
        path.write_bytes(original)
    assert training_recipe(second, "sft").is_file()


def rebind_serving(freeze, mutate):
    for model_index, config in enumerate(freeze["serving"].values()):
        path = Path(config["fixture"]["path"])
        script = read_record(path)
        agents = ["sha256:" + file_identity(path.parent / f"agent-{i}.json")["sha256"] for i in range(2)]
        for agent_id, tasks in script["agents"].items():
            agent_index = agents.index(agent_id)
            for task, attempts in tasks.items():
                for attempt, row in attempts.items():
                    mutate(f"Q{agent_index}{model_index}", task, attempt, row)
        write_record(path, script)
        config["fixture"] = file_identity(path)


@pytest.mark.parametrize("cell", ["Q00", "Q10", "Q01", "Q11"])
def test_incomplete_evidence_in_any_cell_refuses_without_censoring_success(tmp_path, cell):
    authority, candidates, _, freeze = setup_release(tmp_path)

    def truncated(name, task, attempt, row):
        if name == cell:
            row["responses"] = [row["responses"][0]] * 5
            row["prompt_tokens"] = row["completion_tokens"] = 10

    rebind_serving(freeze, truncated)
    plan = authority.freeze(**freeze)
    evaluation = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    decision = authority.decide(evaluation["id"])
    report = decision["payload"]["report"]
    assert decision["payload"]["result"] == "refused"
    assert any(cell in r and "harness_final" in r for r in report["reasons"])
    assert report["joint_gain"] == pytest.approx(0.7)
    assert report["cells"][cell]["tokens"] == 100
    assert report["cells"][cell]["execution"]["complete_families"] == []
    assert len(report["cells"][cell]["execution"]["ineligible_attempts"]) == 60
    assert all(len(c["rows"]) == 60 for c in evaluation["payload"]["matrix"]["cells"].values())
    with pytest.raises(StageError, match="accepted strict"):
        authority.activate(decision["id"])
    assert authority.status()["candidate"] == candidates[0]["id"]
    assert authority.status()["generation"] == 0
    with pytest.raises(StageError, match="already started"):
        execute_crossed(authority, plan["id"], allow_unsandboxed=True)


def test_setup_failed_family_does_not_count_as_confirmation(tmp_path):
    authority, candidates, workspaces, freeze = setup_release(tmp_path)
    workload = read_record(tmp_path / "workload.json")
    workload["tasks"][-1]["task"]["setup"] = "exit 19"
    path = tmp_path / "setup-workload.json"
    write_record(path, workload)
    catalog = read_record(tmp_path / "confirmation.json")
    catalog["memberships"][-1]["version"] = content_digest(asdict(Task.from_record(workload["tasks"][-1]["task"])))
    write_record(tmp_path / "setup-catalog.json", catalog)
    candidate = authority.candidates().register(
        workspace=workspaces[1],
        merged_record=Path(candidates[1]["payload"]["model"]["record"]),
        agent=tmp_path / "agent-1.json",
        workload=path,
        parent=Path(candidates[0]["payload"]["model"]["merged"]),
    )
    new = ReleaseAuthority(tmp_path / "setup-release", mode="fixture", namespace=authority.identity["namespace"])
    new.configure(
        candidates=authority.candidates().store.root,
        incumbent=candidates[0]["id"],
        data_policy=tmp_path / "setup-catalog.json",
        policy=authority.configuration()["policy"],
    )
    for config in freeze["serving"].values():
        path = Path(config["fixture"]["path"])
        script = read_record(path)
        script["origin"] = new.identity
        write_record(path, script)
        config["fixture"] = file_identity(path)
    freeze["new"] = candidate["id"]
    plan = new.freeze(**freeze)
    evaluation = execute_crossed(new, plan["id"], allow_unsandboxed=True)
    decision = new.decide(evaluation["id"])
    assert decision["payload"]["result"] == "refused"
    report = decision["payload"]["report"]
    assert report["joint_gain"] == pytest.approx(7 / 12)
    assert report["joint_interval"] == pytest.approx([0.35, 0.7])
    for cell in report["cells"].values():
        assert len(cell["families"]) == 6
        assert len(cell["execution"]["complete_families"]) == 5
        assert cell["execution"]["eligible_attempts"] == 50
        assert cell["tokens"] == pytest.approx(5000 / 60)
    with pytest.raises(StageError):
        new.activate(decision["id"])
    assert new.status()["generation"] == 0


@pytest.mark.parametrize("variant", ["recovered", "large_usage"])
def test_complete_controls_still_reach_strict_release(tmp_path, variant):
    authority, _, _, freeze = setup_release(tmp_path)

    def control(cell, task, attempt, row):
        if variant == "large_usage":
            row["prompt_tokens"] = 10000
            row["completion_tokens"] = 1024
        elif cell == "Q11":
            row["responses"].insert(0, "<tool_call>garbage</tool_call>")
            row["prompt_tokens"] = row["completion_tokens"] = 16

    rebind_serving(freeze, control)
    plan = authority.freeze(**freeze)
    evidence = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    decision = authority.decide(evidence["id"])
    assert decision["payload"]["result"] == "accepted", decision["payload"]["reasons"]
    assert authority.activate(decision["id"])["generation"] == 1
    assert evidence["payload"]["report"]["cells"]["Q11"]["tokens"] == (22048 if variant == "large_usage" else 96)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        [],
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": "1"},
        {"prompt_tokens": 1, "completion_tokens": 1.0},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 1025},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
        {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": True}},
        {"prompt_tokens": 10000, "completion_tokens": 1024},
        {"prompt_tokens": 0, "completion_tokens": 0},
    ],
)
def test_controlled_production_usage_and_output_limit(monkeypatch, usage):
    monkeypatch.setenv("CPU_USAGE_CREDENTIAL", "fixture-not-a-secret")
    adapter = TrustedServing(
        {
            "url": "https://fixture.invalid",
            "api_key_env": "CPU_USAGE_CREDENTIAL",
            "alias": "same-alias",
            "deployment_id": "deployment",
            "engine": "fixture",
            "precision": "bf16",
            "device": "cpu",
            "environment": "controlled",
        },
        model_id="sha256:" + "a" * 64,
        origin={"mode": "production", "namespace": "controlled"},
        sampling={},
        budget={"max_tokens": 1024},
    )

    def request(route, payload):
        nonce = payload["nonce"] if route == "/identity" else payload["spark_identity"]["nonce"]
        identity = {**adapter.expected, "nonce": nonce}
        return (
            identity
            if route == "/identity"
            else {
                "spark_identity": identity,
                "usage": usage,
                "choices": [{"message": {"content": "controlled response"}}],
            }
        )

    monkeypatch.setattr(adapter, "_request", request)
    observed = []
    complete = adapter.completion(0, usage_observer=observed.append)
    valid = usage in ({"prompt_tokens": 10000, "completion_tokens": 1024}, {"prompt_tokens": 0, "completion_tokens": 0})
    if valid:
        for _ in range(3):
            assert complete([])[1] == usage
        assert len(observed) == 3
    else:
        with pytest.raises(StageError):
            complete([])
        assert observed == [usage]
        with pytest.raises(StageError):
            validate_completion_usage(usage, max_tokens=1024)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completion_tokens", 1025),
        ("completion_tokens", True),
        ("completion_tokens", 1.5),
        ("prompt_tokens", "100"),
        ("prompt_tokens", None),
    ],
)
def test_fixture_usage_refusal_preserves_spent_history_and_response_journal(tmp_path, field, value):
    authority, _, _, freeze = setup_release(tmp_path)

    def bad_usage(cell, task, attempt, row):
        if cell == "Q00" and task == "confirm-0" and attempt == "1":
            row[field] = value

    rebind_serving(freeze, bad_usage)
    plan = authority.freeze(**freeze)
    before = authority.status()
    with pytest.raises(StageError, match="token"):
        execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    directory = authority.experiments.root / plan["id"].removeprefix("sha256:")
    journal = [json.loads(line) for line in (directory / "completion-usage.jsonl").read_text().splitlines()]
    assert len(journal) == 3 and journal[-1]["usage"][field] == value
    assert len((directory / "Q00.jsonl").read_text().splitlines()) == 1
    with authority.store.connect() as db:
        assert db.execute("SELECT status FROM experiment_runs WHERE id=?", (plan["id"],)).fetchone()[0] == "failed"
        assert db.execute("SELECT count(*) FROM confirmation_use WHERE experiment=?", (plan["id"],)).fetchone()[0] == 6
    with pytest.raises(StageError, match="already started"):
        execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    assert authority.status() == before


@pytest.mark.parametrize("last_reason,accepted", [("length", False), ("stop", True)])
def test_fixture_provider_status_controls_strict_release_without_changing_outcomes(tmp_path, last_reason, accepted):
    authority, _, _, freeze = setup_release(tmp_path)

    def status(cell, task, attempt, row):
        if cell == "Q11":
            row["finish_reasons"] = ["tool_calls", last_reason]

    rebind_serving(freeze, status)
    plan = authority.freeze(**freeze)
    before = authority.status()
    evidence = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    decision = authority.decide(evidence["id"])
    report = decision["payload"]["report"]
    assert (decision["payload"]["result"] == "accepted") is accepted
    assert report["joint_gain"] == pytest.approx(0.7)
    assert report["cells"]["Q11"]["execution"]["eligible_attempts"] == (60 if accepted else 0)
    rows = evidence["payload"]["matrix"]["cells"]["Q11"]["rows"]
    assert len(rows) == 60 and sum(r["success"] for r in rows) == 54
    directory = authority.experiments.root / plan["id"].removeprefix("sha256:")
    journal = [json.loads(line) for line in (directory / "completion-status.jsonl").read_text().splitlines()]
    assert sum(r["cell"] == "Q11" and r["finish_reason"] == last_reason for r in journal) == 60
    if accepted:
        assert authority.activate(decision["id"])["generation"] == 1
    else:
        assert any("finish reason: 'length'" in r for r in report["reasons"])
        with pytest.raises(StageError, match="accepted strict"):
            authority.activate(decision["id"])
        assert authority.status() == before

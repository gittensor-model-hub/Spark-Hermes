"""Labelled CPU artifact/serving fixtures; actual replay, preparation and episode tools."""

from dataclasses import asdict
from pathlib import Path

from learning_support import NAMESPACE, policy_config, produce_round

from admin.artifacts import checkpoint_files, content_digest, read_record, write_record
from admin.candidates import CandidateStore, file_identity
from admin.parents import ParentAuthority
from admin.pipeline import Workspace
from admin.release import ReleaseAuthority
from admin.replay import ReplayStore
from admin.selfcheck import FixtureTokenizer
from admin.training import prepare_training
from hermes.cotraining import DEFAULT_POLICY
from hermesbench.tasks import Task
from validator.persistence import state_identity


def checkpoint(root, label, *, revision=None, repository=None, template="CPU template"):
    root.mkdir(parents=True)
    write_record(
        root / "config.json",
        {"model_type": "qwen3_5", "fixture_only": True, "_commit_hash": revision, "_name_or_path": repository},
    )
    (root / "model.safetensors").write_text("CPU_FIXTURE_NOT_MODEL_WEIGHTS_" + label)
    write_record(root / "tokenizer_config.json", {"fixture_only": True, "chat_template": template})
    write_record(root / "tokenizer.json", {"fixture_only": True})
    return root


def setup_release(root, *, families=6, policy=None, pool_families=None, profile="rtx5090-poc"):
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    p = produce_round(root / "source")
    config, training_policy = policy_config(root, [p])
    replay = ReplayStore(root / "replay", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    replay.import_round("source", "r-1")
    workload = {"schema": "spark-crossed-workload-v1", "tasks": []}
    members = []
    for i in range(pool_families or families):
        task = Task(
            task_id=f"confirm-{i}",
            prompt="Write four into answer.txt.",
            tools=("terminal",),
            verify='test "$(cat answer.txt)" = 4',
            hidden_verify='test "$(cat answer.txt)" = 4',
            max_steps=4,
        )
        if i < families:
            workload["tasks"].append({"task": asdict(task), "repository": "fixture/confirmation"})
        members.append(
            {
                "task_id": task.task_id,
                "repository": "fixture/confirmation",
                "version": content_digest(asdict(task)),
                "family_id": f"sealed-{i}",
                "partition": "sealed-release",
                "exposure": [],
            }
        )
    workload_path = root / "workload.json"
    write_record(workload_path, workload)
    confirmation = root / "confirmation.json"
    write_record(
        confirmation,
        {
            "schema": "spark-data-policy-v1",
            "version": "cpu-sealed-v1",
            "origin": {"mode": "fixture", "namespace": NAMESPACE},
            "family_aliases": {
                **training_policy["family_aliases"],
                **{m["family_id"]: m["family_id"] for m in members},
            },
            "memberships": training_policy["memberships"] + members,
            "rights": [],
        },
    )
    source = CandidateStore(root / "candidates", mode="fixture", namespace=NAMESPACE)
    parent_store = ParentAuthority(root / "bootstrap-parent", mode="fixture", namespace=NAMESPACE)
    candidates, workspaces, parent = [], [], None
    for number in range(2):
        ws_root = root / f"workspace-{number}"
        state_identity(ws_root, mode="fixture", namespace=NAMESPACE)
        ws = Workspace(ws_root)
        replay.freeze(ws)
        options = {"parent_approval": parent["id"], "release_root": parent_store.store.root} if parent else {}
        prepared = prepare_training(ws, profile=profile, tokenizer=FixtureTokenizer(), sequence_len=65536, **options)
        template = yaml.safe_load(Path(prepared["prepared_recipe"]).read_text())["chat_template_jinja"]
        merged = checkpoint(ws.models / "sft/merged", str(number), template=template)
        record = ws.models / "sft/merged.json"
        write_record(
            record,
            {
                "merged": str(merged),
                "files": checkpoint_files(merged),
                "origin": ws.identity,
                "stage": "sft",
                "recipe": prepared["prepared_recipe"],
                "recipe_sha256": prepared["recipe_sha256"],
                "corpus_authority": prepared["corpus_authority"],
                **{k: prepared[k] for k in ("profile", "base_model", "revision")},
            },
        )
        agent = root / f"agent-{number}.json"
        write_record(
            agent,
            {
                "schema": "spark-agent-v1",
                "system": f"CPU fixture agent {number}",
                "dialect": "qwen35",
                "tool_schemas": {
                    "terminal": {
                        "name": "terminal",
                        "description": "Run terminal commands",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    }
                },
                "native_tool_messages": False,
            },
        )
        actual_parent = (
            Path(candidates[0]["payload"]["model"]["merged"])
            if number
            else checkpoint(root / "base", "base", revision=prepared["revision"], repository=prepared["base_model"])
        )
        candidate = source.register(
            workspace=ws, merged_record=record, agent=agent, workload=workload_path, parent=actual_parent
        )
        candidates.append(candidate)
        workspaces.append(ws)
        if number == 0:
            evaluation = root / "bootstrap-evaluation.json"
            write_record(evaluation, {"fixture_only": True, "purpose": "initial fixture parent handoff"})
            parent = parent_store.fixture_approve(
                merged_record=record, workspace=ws, agent=candidate["payload"]["agent_id"], evaluation=evaluation
            )
    authority = ReleaseAuthority(root / "release", mode="fixture", namespace=NAMESPACE)
    authority.configure(
        candidates=source.store.root,
        incumbent=candidates[0]["id"],
        data_policy=confirmation,
        policy=policy or DEFAULT_POLICY,
    )
    serving = {}
    for model_number, candidate in enumerate(candidates):
        script = {
            "schema": "spark-serving-fixture-v1",
            "origin": authority.identity,
            "model_id": candidate["payload"]["model_id"],
            "agents": {},
        }
        for agent_number, agent in enumerate(candidates):
            tasks = {}
            # Q00=.2, Q10=.4, Q01=.5, Q11=.9, interaction=.2, joint=.7.
            successes = ((2, 5), (4, 9))[agent_number][model_number]
            for item in workload["tasks"]:
                tasks[item["task"]["task_id"]] = {
                    str(i): {
                        "responses": [
                            "<tool_call>\n<function=terminal>\n<parameter=command>\n"
                            + ("printf '4' > answer.txt" if i < successes else "printf '5' > answer.txt")
                            + "\n</parameter>\n</function>\n</tool_call>",
                            "CPU fixture complete.",
                        ],
                        "prompt_tokens": 25,
                        "completion_tokens": 25,
                        "latency": 1.0,
                    }
                    for i in range(10)
                }
            script["agents"][agent["payload"]["agent_id"]] = tasks
        path = root / f"serving-{model_number}.json"
        write_record(path, script)
        serving[candidate["payload"]["model_id"]] = {"fixture": file_identity(path)}
    freeze = {
        "old": candidates[0]["id"],
        "new": candidates[1]["id"],
        "schedule": [{"attempt_id": str(i), "seed": i} for i in range(10)],
        "budget": {"max_steps": 4, "max_tokens": 1024, "tool_timeout_s": 30},
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "serving": serving,
    }
    write_record(root / "freeze.json", freeze)
    return authority, candidates, workspaces, freeze


def successor(root, authority, prior, approval, index=2, profile="rtx5090-poc"):
    """Prepare a real next SFT recipe from a strict accepted parent; checkpoint stays a labelled fixture."""
    import yaml

    previous = prior["payload"]
    replay = ReplayStore(Path(previous["corpus"]["authority"]["root"]))
    state_identity(root / f"workspace-{index}", mode="fixture", namespace=NAMESPACE)
    ws = Workspace(root / f"workspace-{index}")
    replay.freeze(ws)
    prepared = prepare_training(
        ws,
        profile=profile,
        tokenizer=FixtureTokenizer(),
        sequence_len=65536,
        parent_approval=approval["id"],
        release_root=authority.store.root,
    )
    template = yaml.safe_load(Path(prepared["prepared_recipe"]).read_text())["chat_template_jinja"]
    merged = checkpoint(ws.models / "sft/merged", str(index), template=template)
    record = ws.models / "sft/merged.json"
    write_record(
        record,
        {
            "merged": str(merged),
            "files": checkpoint_files(merged),
            "origin": ws.identity,
            "stage": "sft",
            "recipe": prepared["prepared_recipe"],
            "recipe_sha256": prepared["recipe_sha256"],
            "corpus_authority": prepared["corpus_authority"],
            **{k: prepared[k] for k in ("profile", "base_model", "revision")},
        },
    )
    agent = read_record(Path(previous["agent"]["path"]))
    agent["system"] = f"CPU fixture successor {index}"
    agent_path = root / f"agent-{index}.json"
    write_record(agent_path, agent)
    workload = read_record(Path(previous["workload"]["path"]))
    for i, item in enumerate(workload["tasks"]):
        item["task"]["task_id"] = f"confirm-{(index - 1) * 6 + i}"
    workload_path = root / f"workload-{index}.json"
    write_record(workload_path, workload)
    candidate = authority.candidates().register(
        workspace=ws,
        merged_record=record,
        agent=agent_path,
        workload=workload_path,
        parent=Path(previous["model"]["merged"]),
    )
    serving = {}
    for model_index, model in enumerate((prior, candidate)):
        scripts = {}
        for agent_index, a in enumerate((prior, candidate)):
            successes = ((2, 5), (4, 9))[agent_index][model_index]
            scripts[a["payload"]["agent_id"]] = {
                item["task"]["task_id"]: {
                    str(i): {
                        "responses": [
                            "<tool_call>\n<function=terminal>\n<parameter=command>\n"
                            + ("printf '4' > answer.txt" if i < successes else "printf '5' > answer.txt")
                            + "\n</parameter>\n</function>\n</tool_call>",
                            "CPU fixture complete.",
                        ],
                        "prompt_tokens": 25,
                        "completion_tokens": 25,
                        "latency": 1,
                    }
                    for i in range(10)
                }
                for item in workload["tasks"]
            }
        path = root / f"cycle-{index}-serving-{model_index}.json"
        write_record(
            path,
            {
                "schema": "spark-serving-fixture-v1",
                "origin": authority.identity,
                "model_id": model["payload"]["model_id"],
                "agents": scripts,
            },
        )
        serving[model["payload"]["model_id"]] = {"fixture": file_identity(path)}
    freeze = {
        "old": prior["id"],
        "new": candidate["id"],
        "schedule": [{"attempt_id": str(i), "seed": i} for i in range(10)],
        "budget": {"max_steps": 4, "max_tokens": 1024, "tool_timeout_s": 30},
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "serving": serving,
    }
    return candidate, freeze

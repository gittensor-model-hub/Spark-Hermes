"""Installed, CPU-only connected cycle demonstration with explicit external fixtures.

All authoritative outputs come from the normal admission, judge, settlement,
replay, preparation, training, merge, candidate, matrix and controller producers.
The fixture adapters cannot confer production approval or deliver rewards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, content_digest, read_record, write_record
from admin.candidates import CandidateStore
from admin.cycle_fixtures import GitHubFixture, SynthesisFixture, competition_serving, matrix_serving, workloads
from admin.cycle_jobs import FixtureTraining
from admin.cycles import CycleController
from admin.parents import ParentAuthority
from admin.pipeline import Workspace
from admin.release import ReleaseAuthority
from admin.replay import ReplayStore
from admin.selfcheck import FixtureTokenizer
from admin.training import merge, prepare_training, train
from hermes.cotraining import DEFAULT_POLICY
from hermes.taskgen.cli import _write_accepted
from hermes.taskgen.dna import TaskDNA
from hermes.taskgen.gate import gate
from hermes.taskgen.synth import synthesise
from hermesbench.tasks import load_suite
from validator.persistence import locked, state_identity

NAMESPACE = "installed-cpu-cycle"
PROFILE = "rtx5090-poc"
SALT = "CPU-FIXTURE-WITHHELD-SALT-NOT-A-PRODUCTION-SECRET"


def _workspace(path: Path) -> Workspace:
    state_identity(path, mode="fixture", namespace=NAMESPACE)
    return Workspace(path)


def _generated(root: Path, label: str) -> Workspace:
    ws = _workspace(root / label)
    dna = TaskDNA(
        domain="repository_engineering",
        skills=("self_verification",),
        environment="shell",
        difficulty=2,
        horizon=(2, 4),
        failure_mode="wrong_ordering",
        required_tools=("terminal",),
        verification="public and withheld",
        source={"fixture_only": True},
    )
    result = synthesise(
        dna, task_id="gen-" + label, complete=SynthesisFixture(ws.identity, root / (label + "-synthesis.json"), label)
    )
    verdict = gate(result.candidate, timeout_s=30)
    if not verdict.accepted:
        raise StageError("CPU generated task failed actual gate: " + verdict.failed_check)
    _write_accepted(ws.tasks / "generated", ws.tasks / "withheld", result, salt=SALT, max_steps=12)
    ws.record("generate", {"origin": ws.identity, "accepted": [result.candidate.task_id], "gate": verdict.to_record()})
    return ws


def _round(root: Path, round_id: str, ws: Workspace, *, release_root: Path | None = None, feedback=None):
    from admin.competition_pair import base_agent_id, base_profile, build_epoch
    from eval.strategy_track import Commitment
    from hermes.challenge import from_episode_log
    from hermes.round import open_round
    from hermes.seed import OPEN, Round
    from hermesbench.runner import verify_digest
    from hermesbench.sink import read_episodes
    from hermesbench.withheld import overlay
    from validator.intake import Intake
    from validator.judge import judge_round, runner_for
    from validator.pr_admission import GitHubSource, admit
    from validator.settlement import SettlementStore
    from validator.store import RoundStore

    source = root / "source"
    store = RoundStore(source / "rounds", mode="fixture", namespace=NAMESPACE)
    intake = Intake(source / "bundles", source / "receipts.jsonl", mode="fixture", namespace=NAMESPACE)
    settlement = SettlementStore(source / "settlement", mode="fixture", namespace=NAMESPACE)
    os.environ["SPARKDISTILL_WITHHELD_ROOT"] = str(ws.tasks / "withheld")
    tasks = overlay(load_suite("generated", root=ws.tasks), root=ws.tasks / "withheld", salt=SALT)
    task = tasks[0]
    epoch = build_epoch(
        tasks,
        attempt_ids=[str(i) for i in range(10)],
        epoch_id=round_id if release_root is None else None,
        release_root=release_root,
        profile=PROFILE,
        salt=SALT,
    )
    agent_id = ReleaseAuthority(release_root).active_pair()["agent_id"] if release_root else base_agent_id(PROFILE)
    fixture_root = release_root or store.root
    fixture_identity = state_identity(fixture_root)
    baseline_script = ws.root / "baseline-serving.json"
    scripts = {miner: ws.root / (miner + "-serving.json") for miner in ("alice", "bob")}
    for path, successes, tokens in ((baseline_script, 3, 100), (scripts["alice"], 10, 25), (scripts["bob"], 8, 25)):
        competition_serving(
            path,
            identity=fixture_identity,
            model_id=epoch["model_revision"],
            agent_id=agent_id,
            task_id=task.task_id,
            successes=successes,
            tokens=tokens,
        )
    common = {
        "round_id": round_id,
        "base_url": "http://fixture.invalid",
        "model": epoch["model_revision"] if release_root else base_profile(PROFILE)["repository"],
        "api_key_env": "SPARK_DEMO_UNUSED_KEY",
        "task_id": task.task_id,
        "repeats": 10,
        "repo_root": None,
        "allow_unsandboxed": True,
        "dialect": "qwen35",
        "task_root": ws.tasks,
        "release_root": release_root,
        "fixture_root": fixture_root,
        "profile": PROFILE,
    }
    context = {"epoch": epoch, "round_id": round_id, "origin": store.identity, "bundle_sha256": ""}
    baseline_dir = ws.root / "baseline"
    baseline_dir.mkdir()
    with (ws.root / "runner.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
        baseline = runner_for(**common, fixture_serving=baseline_script, evaluation_context=context)(
            "baseline", None, baseline_dir
        )
    pins = {
        "task_id": task.task_id,
        "verify_digest": verify_digest(task),
        "private_check_required": True,
        "hidden_verify_commitment": task.hidden_verify_commitment,
        "task_version": content_digest(asdict(task)),
        "feedback": feedback,
    }
    challenges, refused = from_episode_log(read_episodes(baseline), epoch=epoch, task_pins={task.task_id: pins})
    if len(challenges) != 1 or refused:
        raise StageError("actual baseline did not create a challenge: " + str(refused))
    assignment = Round(
        round_id=round_id, seed="a" * 64, task_ids=(task.task_id,), miner_ids=("alice", "bob"), replicas=2, state=OPEN
    )
    window = open_round(challenges[0], round_id=round_id, opened_at=0, deadline=1e12, assignment=assignment)
    store.save(window)
    admissions = []
    previous = os.environ.get("SPARK_DEMO_GITHUB_TOKEN")
    os.environ["SPARK_DEMO_GITHUB_TOKEN"] = "CPU-FIXTURE-CREDENTIAL"
    try:
        for miner in ("alice", "bob"):
            receipt = intake.accept(
                round_id=round_id,
                miner_id=miner,
                files={"SOUL.md": f"CPU fixture contribution by {miner}: copy inputs exactly and check output."},
                now=10,
            )
            commitment = Commitment(round_id, miner, receipt.bundle_sha256, (task.task_id,)).to_record()
            transport = GitHubFixture(
                store.identity, assignment.to_record(reveal_seed=True), commitment, ws.root / "github.jsonl"
            )
            metadata = GitHubSource(
                "fixture/spark", credential_env="SPARK_DEMO_GITHUB_TOKEN", transport=transport
            ).collect(transport.number, round_id=round_id)
            admissions.append(admit(metadata=metadata, round_id=round_id, store=store, intake=intake, now=20))
    finally:
        if previous is None:
            os.environ.pop("SPARK_DEMO_GITHUB_TOKEN", None)
        else:
            os.environ["SPARK_DEMO_GITHUB_TOKEN"] = previous
    window = store.load(round_id)
    window.freeze(now=30, reason="CPU fixture contribution window")
    store.save(window)
    runners = {miner: runner_for(**common, fixture_serving=script) for miner, script in scripts.items()}

    def execute(miner, bundle, workspace):
        return runners[miner](miner, bundle, workspace)

    with (ws.root / "runner.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
        result = judge_round(
            round_id=round_id,
            run=execute,
            model_revision=epoch["model_revision"],
            harness_digest=epoch["harness_digest"],
            store=store,
            intake=intake,
            workspace=source / "episodes" / round_id,
            scorecard_dir=source / "cards",
            settle=False,
        )
    if len(result) != 2 or not all(r.ok for r in result):
        raise StageError("actual admitted runner/judge failed: " + str(result))
    # Bootstrap needs a corpus before a controller exists. Later rounds remain
    # GRADED: the controller owns their actual settlement and outbox producers.
    if release_root is None:
        settlement.activate(store, round_id, "fixture/spark")
        settlement.settle_round(store, scorecards=source / "cards", episodes=source / "episodes", round_id=round_id)
    record = {
        "round_id": round_id,
        "task": asdict(task),
        "version": content_digest(pins),
        "admissions": admissions,
        "workspace": str(ws.root),
        "epoch": epoch,
        "baseline": str(baseline),
    }
    write_record(ws.root / "round.json", record)
    return record


def _replay(root: Path, number: int, rounds, sealed):
    policy = {
        "schema": "spark-data-policy-v1",
        "version": f"cpu-demo-policy-{number}",
        "origin": {"mode": "fixture", "namespace": NAMESPACE},
        "family_aliases": {"copy-provenance": "copy-provenance", **{m["family_id"]: m["family_id"] for m in sealed}},
        "memberships": [
            {
                "task_id": r["task"]["task_id"],
                "repository": "fixture/spark",
                "version": r["version"],
                "family_id": "copy-provenance",
                "partition": "private-competition",
                "exposure": ["selection"],
            }
            for r in rounds
        ]
        + sealed,
        "rights": [
            {
                "subject": subject,
                "license": "CPU-FIXTURE-ONLY",
                "attribution": "scripted CPU software contribution fixture",
                "training": True,
                "derivatives": True,
            }
            for r in rounds
            for subject in ("task:" + r["version"], *(a["admission_id"] for a in r["admissions"]))
        ],
    }
    path = root / f"policy-{number}.json"
    write_record(path, policy)
    source = root / "source"
    config = {
        "policy": str(path),
        "sources": [
            {
                "name": "competition",
                "rounds": str(source / "rounds"),
                "settlement": str(source / "settlement"),
                "intake": str(source / "bundles"),
                "receipts": str(source / "receipts.jsonl"),
            }
        ],
    }
    replay = ReplayStore(root / f"replay-{number}", mode="fixture", namespace=NAMESPACE)
    replay.configure(config)
    return replay


def _agent(root: Path, number: int) -> Path:
    from hermes.pin import load_tool_schemas
    from hermesbench.runner import HARNESS_DIR

    path = root / f"agent-{number}.json"
    write_record(
        path,
        {
            "schema": "spark-agent-v1",
            "system": f"CPU fixture incumbent agent {number}: follow the task and verify its output.",
            "dialect": "qwen35",
            "tool_schemas": {
                name: {"name": name, **schema} for name, schema in load_tool_schemas(HARNESS_DIR / "tools.json").items()
            },
            "native_tool_messages": False,
        },
    )
    return path


def _bootstrap(root: Path, replay: ReplayStore):
    ws = _workspace(root / "bootstrap-model")
    replay.import_round("competition", "round-bootstrap")
    replay.freeze(ws)
    prepared = prepare_training(
        ws, profile=PROFILE, sequence_len=65536, max_steps=1, tokenizer=FixtureTokenizer(), local_files_only=True
    )
    executor = FixtureTraining(ws, "installed-demo-bootstrap")
    training = train(ws, stage="sft", fixture_executor=executor)
    ws.record("train", {**training, "fixture_only": True, "trained": False})
    merge(ws, stage="sft", fixture_executor=executor)
    base = root / "fixture-pinned-base"
    base.mkdir()
    write_record(
        base / "config.json",
        {
            "model_type": "qwen3_5",
            "fixture_only": True,
            "_name_or_path": prepared["base_model"],
            "_commit_hash": prepared["revision"],
        },
    )
    (base / "model.safetensors").write_text("CPU_FIXTURE_NOT_MODEL_WEIGHTS:PINNED_BASE")
    write_record(base / "tokenizer.json", {"fixture_only": True})
    write_record(
        base / "tokenizer_config.json", {"fixture_only": True, "chat_template": "CPU fixture pinned base tokenizer"}
    )
    candidates = CandidateStore(root / "candidates", mode="fixture", namespace=NAMESPACE)
    candidate = candidates.register(
        workspace=ws,
        merged_record=ws.models / "sft/merged.json",
        agent=_agent(root, 0),
        workload=root / "workload-1.json",
        parent=base,
    )
    initial = ParentAuthority(root / "bootstrap-parent", mode="fixture", namespace=NAMESPACE)
    declaration = root / "bootstrap-fixture-declaration.json"
    write_record(
        declaration,
        {
            "fixture_only": True,
            "purpose": "Initial configured CPU pair handoff; not a measured release",
            "training_workspace": str(ws.root),
        },
    )
    parent = initial.fixture_approve(
        merged_record=ws.models / "sft/merged.json",
        workspace=ws,
        agent=candidate["payload"]["agent_id"],
        evaluation=declaration,
    )
    return candidate, parent


def _command(root: Path, *args: str, expected=0):
    command = [sys.executable, "-m", "admin.cli", *map(str, args)]
    with (root / "commands.jsonl").open("a") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=root) as process:
            try:
                stdout, stderr = process.communicate(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                stream.write(
                    json.dumps(
                        {
                            "argv": command,
                            "pid": process.pid,
                            "exit_code": process.returncode,
                            "stdout": stdout,
                            "stderr": stderr,
                            "timeout": True,
                        }
                    )
                    + "\n"
                )
                raise StageError("cycle client timed out; resume reconciles its original supervised jobs")
        record = {
            "argv": command,
            "pid": process.pid,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        stream.write(json.dumps(record) + "\n")
    if process.returncode != expected:
        raise StageError(f"installed cycle subprocess exited {process.returncode}; see {root / 'commands.jsonl'}")
    return json.loads(stdout)


def _spec(root, controller, replay, round_record, number, *, history_ids):
    serving = root / f"matrix-serving-{number}.json"
    matrix_serving(serving, controller.identity, cycle=number)
    source = root / "source"
    return {
        "rounds": [
            {
                "source": "competition",
                "round_id": round_record["round_id"],
                "prs": [
                    {"number": a["pr_number"], "head": a["head_sha"], "author": a["author"]}
                    for a in round_record["admissions"]
                ],
                "scorecards": str(source / "cards"),
                "episodes": str(source / "episodes"),
            }
        ],
        "replay": str(replay.root),
        "history_ids": history_ids,
        "confirmation_policy": replay.configuration()["policy"],
        "agent": str(_agent(root, number)),
        "workload": str(root / f"workload-{number}.json"),
        "training": {"profile": PROFILE, "sequence_len": 65536, "max_steps": 1, "execution": "fixture"},
        "curriculum": {"version": "spark-curriculum-v1", "count": 1, "min_families": 1, "max_per_family": 1},
        "evaluation": {
            "schedule": [{"attempt_id": str(i), "seed": i} for i in range(10)],
            "budget": {"max_steps": 4, "max_tokens": 1024, "tool_timeout_s": 30},
            "sampling": {"temperature": 0.0, "top_p": 1.0},
            "fixture": str(serving),
        },
    }


def _first(root: Path):
    sealed = workloads(root, state_identity(root))
    boot = _round(root, "round-bootstrap", _generated(root, "bootstrap-task"))
    replay0 = _replay(root, 0, [boot], sealed)
    candidate, parent = _bootstrap(root, replay0)
    controller = CycleController(root / "releases", mode="fixture", namespace=NAMESPACE)
    controller.release.configure(
        candidates=root / "candidates",
        incumbent=candidate["id"],
        data_policy=root / "policy-0.json",
        policy=DEFAULT_POLICY,
    )
    round1 = _round(root, "round-one", _generated(root, "first-task"), release_root=controller.root)
    replay1 = _replay(root, 1, [boot, round1], sealed)
    history = [replay1.import_round("competition", boot["round_id"])["id"]]
    controller.configure(
        replay=replay1.root, bootstrap_parent={"root": str(root / "bootstrap-parent"), "id": parent["id"]}
    )
    spec = _spec(root, controller, replay1, round1, 1, history_ids=history)
    write_record(root / "first-spec.json", spec)
    first = _command(
        root,
        "cycle",
        "start",
        "--root",
        str(controller.root),
        "--name",
        "first",
        "--spec",
        str(root / "first-spec.json"),
    )
    write_record(root / "demo-progress.json", {"first": first["id"], "rounds": [boot, round1], "sealed": sealed})


def run(root: Path) -> dict[str, Any]:
    from admin.task_feedback import generate_from_feedback, verify_feedback_task

    root = root.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "demo-config.json").is_file():
        raise StageError("demo requires a fresh root or its original retained demo root")
    state_identity(root, mode="fixture", namespace=NAMESPACE)
    config = {
        "schema": "spark-installed-cycle-demo-v1",
        "origin": state_identity(root),
        "fixture_only": True,
        "profile": PROFILE,
    }
    if (root / "demo-config.json").exists() and read_record(root / "demo-config.json") != config:
        raise StageError("demo configuration differs from original fixture root")
    write_record(root / "demo-config.json", config)
    prior_salt, prior_withheld = (
        os.environ.get("HERMESBENCH_WITHHELD_SALT"),
        os.environ.get("SPARKDISTILL_WITHHELD_ROOT"),
    )
    os.environ["HERMESBENCH_WITHHELD_SALT"] = SALT
    try:
        with locked(root / ".demo.lock"):
            if not (root / "demo-progress.json").exists():
                _first(root)
            progress = read_record(root / "demo-progress.json")
            controller = CycleController(root / "releases")
            first_id = progress["first"]
            if "second" not in progress:
                # Distinct installed processes execute/resume the same controller job set.
                _command(
                    root, "cycle", "resume", "--root", str(controller.root), "--id", first_id, "--through", "prepare"
                )
                first = _command(
                    root,
                    "cycle",
                    "resume",
                    "--root",
                    str(controller.root),
                    "--id",
                    first_id,
                    "--execute-training",
                    "--allow-unsandboxed",
                )
                if first["status"] != "complete" or first["incumbent"]["generation"] != 1:
                    raise StageError("first actual crossed release did not activate")
                replay1 = controller.replay(first_id)
                curriculum = next(j["output"]["value"] for j in first["jobs"] if j["stage"] == "curriculum")
                ws = _workspace(root / "feedback-task")
                generation = generate_from_feedback(
                    replay1,
                    curriculum_id=curriculum["authority_id"],
                    request_id=curriculum["requests"][0]["request_id"],
                    workspace=ws,
                    complete=SynthesisFixture(ws.identity, root / "feedback-synthesis.json", "measured feedback"),
                    salt=SALT,
                    allow_unsandboxed=True,
                )
                verify_feedback_task(replay1, generation["id"])
                second_round = _round(
                    root,
                    "round-two",
                    ws,
                    release_root=controller.root,
                    feedback={
                        "root": str(replay1.root),
                        "id": generation["id"],
                        "request": generation["payload"]["request"],
                    },
                )
                replay2 = _replay(root, 2, [*progress["rounds"], second_round], progress["sealed"])
                history = [replay2.import_round("competition", r["round_id"])["id"] for r in progress["rounds"]]
                spec = _spec(root, controller, replay2, second_round, 2, history_ids=history)
                write_record(root / "second-spec.json", spec)
                second = _command(
                    root,
                    "cycle",
                    "start",
                    "--root",
                    str(controller.root),
                    "--name",
                    "second",
                    "--spec",
                    str(root / "second-spec.json"),
                )
                progress.update(
                    second=second["id"],
                    feedback=generation,
                    first_incumbent=first["incumbent"],
                    second_round=second_round,
                )
                write_record(root / "demo-progress.json", progress)
            second = _command(
                root,
                "cycle",
                "resume",
                "--root",
                str(controller.root),
                "--id",
                progress["second"],
                "--execute-training",
                "--allow-unsandboxed",
                expected=3,
            )
            first = _command(root, "cycle", "status", "--root", str(controller.root), "--id", first_id)
            if second["status"] != "refused" or second["incumbent"] != progress["first_incumbent"]:
                raise StageError("second numerical refusal did not preserve the first exact incumbent/history")
            from validator.settlement import SettlementStore

            settlement = SettlementStore(root / "source/settlement")
            for round_id in ("round-bootstrap", "round-one", "round-two"):
                outcome = settlement.record(round_id)["outcome"]
                if not outcome["crowned"] or outcome["winner"]["miner_id"] != "alice":
                    raise StageError("demonstration requires an actually accepted and settled contribution")
            if progress["feedback"]["payload"]["request"]["category"] != "withheld_generalization":
                raise StageError("next task must consume genuine recorded private-check failures")
            prepared = read_record(Path(second["workspace"]) / "models/sft/prepared.json")
            summary = {
                **config,
                "ready": True,
                "trained": False,
                "measured_learning": False,
                "live_rewards": False,
                "first": {k: first[k] for k in ("id", "status", "workspace", "incumbent")},
                "second": {k: second[k] for k in ("id", "status", "workspace", "incumbent")},
                "second_parent": prepared["parent"],
                "feedback": progress["feedback"],
                "commands": str(root / "commands.jsonl"),
                "module_origin": __file__,
                "external_prerequisites": [
                    "licensed real corpus and sealed independent checks",
                    "real 4B then 27B training and measured gains",
                    "trusted serving identity and hardware validation/optional attestation",
                    "SN74 repository onboarding and subnet-controlled eligibility/payout",
                ],
            }
            write_record(root / "summary.json", summary)
            return summary
    finally:
        for key, value in (("HERMESBENCH_WITHHELD_SALT", prior_salt), ("SPARKDISTILL_WITHHELD_ROOT", prior_withheld)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, type=Path, help="fresh disposable directory; retain it for resume/evidence"
    )
    parser.add_argument(
        "--mode", required=True, choices=("fixture",), help="required: CPU fixtures never authorize production"
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.root), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f"cycle demo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

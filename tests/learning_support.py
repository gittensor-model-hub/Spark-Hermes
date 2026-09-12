"""Real CPU execution -> admission/judge -> settlement fixtures for learning tests.

Policy actions/token counts and authenticated GitHub responses are labelled fixtures.
The tools, verifiers, trajectories, scorecards and durable records are real producers.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from competition_support import GitHubTransport

from admin.artifacts import content_digest, write_record
from eval.strategy_track import Commitment
from hermes.challenge import from_episode_log
from hermes.protocol import DIALECTS
from hermes.round import open_round
from hermes.seed import OPEN, Round
from hermes.trajectory import FINAL, TOOL_CALL, Step
from hermesbench.runner import LocalToolExecutor, run_suite
from hermesbench.sink import JsonlEpisodeSink, read_episodes
from hermesbench.tasks import Task
from validator.intake import Intake, capture_surface
from validator.judge import execution_context, judge_round
from validator.pr_admission import GitHubSource, admit
from validator.score import policy_record
from validator.settlement import SettlementStore
from validator.store import RoundStore

NAMESPACE = "cpu-learning"


class Policy:
    dialect = DIALECTS["qwen35"]
    tokens_used = 100

    def __init__(self, success, system):
        self.success, self.system = success, system

    def next_steps(self, task, history):
        if history:
            return [Step(kind=FINAL, content="CPU fixture complete.")]
        return [
            Step(
                kind=TOOL_CALL,
                tool="terminal",
                call_id="write",
                args={"command": "printf '4' > answer.txt" if self.success else "printf '5' > answer.txt"},
            )
        ]


def produce_round(
    root: Path,
    round_id="r-1",
    *,
    task_id="gen-learning",
    successes=5,
    variant="",
    poison=None,
    through="settlement",
    incumbent=None,
):
    root.mkdir(parents=True, exist_ok=True)
    store = RoundStore(root / "rounds", mode="fixture", namespace=NAMESPACE)
    intake = Intake(root / "bundles", root / "receipts.jsonl", mode="fixture", namespace=NAMESPACE)
    settlement = SettlementStore(root / "settlement", mode="fixture", namespace=NAMESPACE)
    task = Task(
        task_id=task_id,
        prompt="Write four into answer.txt." + variant,
        tools=("terminal",),
        verify='test "$(cat answer.txt)" = 4',
        max_steps=4,
    )
    epoch = {
        "epoch_id": round_id,
        "model_revision": "a" * 40,
        "harness_digest": "b" * 64,
        "attempt_ids": [str(i) for i in range(10)],
        "score_policy": policy_record(),
    }
    if incumbent is not None:
        epoch["incumbent"] = incumbent
    base_path = root / f"{round_id}-baseline.jsonl"
    count = []

    def baseline(_):
        count.append(1)
        return Policy(len(count) <= 3, "CPU baseline fixture")

    with JsonlEpisodeSink(base_path, keep_trajectories=True) as sink:
        run_suite(
            [task],
            baseline,
            LocalToolExecutor(allow_unsandboxed=True),
            root / f"{round_id}-baseline-work",
            repeats=10,
            sink=sink,
            evaluation_context={"epoch": epoch, "round_id": round_id, "origin": store.identity, "bundle_sha256": ""},
        )
    pins = {"task_id": task_id, "verify": task.verify, "private_check_required": False}
    challenges, refused = from_episode_log(read_episodes(base_path), epoch=epoch, task_pins={task_id: pins})
    assert len(challenges) == 1 and not refused
    assignment = Round(
        round_id=round_id, seed="a" * 64, task_ids=(task_id,), miner_ids=("alice",), replicas=1, state=OPEN
    )
    win = open_round(challenges[0], round_id=round_id, opened_at=0, deadline=1e12, assignment=assignment)
    store.save(win)
    receipt = intake.accept(round_id=round_id, miner_id="alice", files={"SOUL.md": "CPU learning fixture"}, now=10)
    commitment = Commitment(round_id, "alice", receipt.bundle_sha256, (task_id,)).to_record()
    transport = GitHubTransport(assignment.to_record(reveal_seed=True), commitment)
    os.environ["GH_TOKEN"] = "fixture-credential"
    if through == "open":
        return {
            "store": store,
            "intake": intake,
            "settlement": settlement,
            "task": task,
            "version": content_digest(pins),
            "transport": transport,
        }
    metadata = GitHubSource("example/spark", transport=transport).collect(7, round_id=round_id)
    admission = admit(metadata=metadata, round_id=round_id, store=store, intake=intake, now=20)
    win = store.load(round_id)
    win.freeze(now=30, reason="CPU learning fixture")
    store.save(win)
    count.clear()

    def execute(miner, bundle, workspace):
        files = capture_surface(bundle)
        assert files == {"SOUL.md": "CPU learning fixture"}
        log = workspace / f"{miner}.jsonl"

        def candidate(_):
            count.append(1)
            return Policy(len(count) <= successes, files["SOUL.md"])

        with JsonlEpisodeSink(log, keep_trajectories=True) as sink:
            run_suite(
                [task],
                candidate,
                LocalToolExecutor(allow_unsandboxed=True),
                workspace / "work",
                repeats=10,
                sink=sink,
                evaluation_context=execution_context(miner, bundle),
            )
        if poison:
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            poison(rows)
            log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return log

    result = judge_round(
        round_id=round_id,
        run=execute,
        model_revision=epoch["model_revision"],
        harness_digest=epoch["harness_digest"],
        store=store,
        intake=intake,
        workspace=root / "episodes" / round_id,
        scorecard_dir=root / "cards",
        settle=False,
    )
    assert result[0].ok, result[0].problem
    record = None
    if through != "graded":
        settlement.activate(store, round_id, "example/spark")
        record = settlement.settle_round(
            store, scorecards=root / "cards", episodes=root / "episodes", round_id=round_id
        )
    return {
        "store": store,
        "intake": intake,
        "settlement": settlement,
        "record": record,
        "admission": admission,
        "task": task,
        "version": content_digest(pins),
    }


def policy_config(root, produced, *, mixture=None):
    members = []
    rights = {}
    for index, p in enumerate(produced):
        member = {
            "task_id": p["task"].task_id,
            "repository": "example/spark",
            "version": p["version"],
            "family_id": p.get("family", f"family-{index}"),
            "partition": "private-competition",
            "exposure": ["selection"],
        }
        if member not in members:
            members.append(member)
        for subject in ("task:" + p["version"], p["admission"]["admission_id"]):
            rights[subject] = {
                "subject": subject,
                "license": "CPU-FIXTURE-ONLY",
                "attribution": "scripted CPU fixture",
                "training": True,
                "derivatives": True,
            }
    policy = {
        "schema": "spark-data-policy-v1",
        "version": "fixture-policy-v1",
        "origin": {"mode": "fixture", "namespace": NAMESPACE},
        "family_aliases": {m["family_id"]: m["family_id"] for m in members},
        "memberships": members,
        "rights": list(rights.values()),
    }
    path = root / "policy.json"
    write_record(path, policy)
    sources = {}
    for p in produced:
        base = p["store"].root.parent
        sources[str(base)] = {
            "name": base.name,
            "rounds": str(p["store"].root),
            "settlement": str(p["settlement"].root),
            "intake": str(p["intake"].root),
            "receipts": str(p["intake"].receipts),
        }
    config = {"policy": str(path), "sources": list(sources.values())}
    if mixture:
        config["mixture"] = mixture
    write_record(root / "config.json", config)
    return config, policy


def cli(root, module, *args, expected=0):
    command = [sys.executable, "-m", module, *map(str, args)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    with (root / "commands.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"argv": command, "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr})
            + "\n"
        )
    assert done.returncode == expected, done.stdout + done.stderr
    return done

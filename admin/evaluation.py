"""Execute held-out evaluation and export the record consumed by the promotion gate."""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, file_digest, read_record, write_record
from admin.pipeline import Workspace, evaluate_command, withheld_environment
from hermes.harness import HarnessError, RunManifest
from hermes.promotion import Serving
from hermesbench.sink import read_episodes
from hermesbench.tasks import load_suite


def read_serving(path: Path) -> Serving:
    record = read_record(path)
    required = {"precision", "device", "engine", "temperature", "top_p", "max_model_len", "confidential_computing"}
    if required - record.keys():
        raise StageError("serving config is missing: " + ", ".join(sorted(required - record.keys())))
    for key in ("precision", "device", "engine"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise StageError(f"serving config requires a nonempty {key}")
    for key in ("temperature", "top_p"):
        if type(record[key]) not in (int, float) or not math.isfinite(record[key]):
            raise StageError(f"serving config requires a finite {key}")
    if record["temperature"] < 0 or not 0 < record["top_p"] <= 1:
        raise StageError("serving config has invalid sampling values")
    if type(record["max_model_len"]) is not int or record["max_model_len"] < 1:
        raise StageError("serving config requires a positive max_model_len")
    if type(record["confidential_computing"]) is not bool:
        raise StageError("serving config must state confidential_computing as true or false")
    return Serving.from_record(record)


def evaluate(workspace: Workspace, *, serving: Serving, runner: Any = subprocess.run, **options: Any) -> dict[str, Any]:
    tasks = load_suite("v0,v1")
    environment = withheld_environment(tasks)
    episodes = workspace.reports / "episodes.jsonl"
    if episodes.exists() and episodes.stat().st_size:
        raise StageError("evaluation episodes already exist; use a separate --root per model")
    command = evaluate_command(
        workspace, sampling={"temperature": serving.temperature, "top_p": serving.top_p}, **options
    )
    done = runner(command, env=environment)
    if done.returncode:
        raise StageError(f"evaluation exited {done.returncode}")
    manifest_path = workspace.reports / "manifest.json"
    try:
        manifest = RunManifest.from_record(read_record(manifest_path))
    except HarnessError as exc:
        raise StageError(f"invalid evaluation manifest: {exc}") from exc
    if manifest.model != options["model"]:
        raise StageError("evaluation manifest names a different model")
    rows = list(read_episodes(episodes))
    counts: Counter[str] = Counter()
    for row in rows:
        metrics = row.get("metrics", {})
        if type(metrics.get("hidden_passed")) is not bool:
            raise StageError("evaluation is missing withheld-check results")
        counts[metrics.get("task_id", "")] += 1
    if counts != Counter({t.task_id: 10 for t in tasks}):
        raise StageError("evaluation is incomplete or includes unexpected tasks; no completion recorded")
    fields = ("task_id", "hidden_passed", "steps", "tokens_used", "tool_calls")
    logged = Counter(
        tuple(row["metrics"].get(key) for key in fields)
        + (row["metrics"].get("success"), row.get("disqualified", False))
        for row in rows
    )
    published = Counter(
        tuple(getattr(result, key) for key in fields) + (result.passed, result.disqualified)
        for result in manifest.results
    )
    if logged != published:
        raise StageError("evaluation manifest results disagree with the episode log")
    record = {
        "model": options["model"],
        "serving": serving.to_record(),
        "episodes_path": episodes.name,
        "manifest_path": manifest_path.name,
        "episodes_sha256": file_digest(episodes),
        "manifest_sha256": file_digest(manifest_path),
        "suite": manifest.suite.to_record(),
        "harness": manifest.harness,
        "serving_evidence": "operator-declared; sampling values were sent with every completion request",
    }
    path = workspace.reports / "run.json"
    write_record(path, record)
    return {"summary": f"held-out evaluation: {options['model']}", "manifest": str(manifest_path), "run": str(path)}


def crossed_rows(artifact: dict[str, Any], *, plan: dict[str, Any], cell: str) -> list[dict[str, Any]]:
    """Export exact attempt identities and costs from the original complete episode log."""
    rows = []
    expected = {tuple(r[k] for k in ("task_id", "attempt_id")): r for r in plan["schedule"]}
    from admin.artifacts import canonical, content_digest
    from admin.candidates import bound_bytes
    from admin.episode_evidence import assess_episode
    from hermesbench.sink import decode_episodes

    for episode in decode_episodes(bound_bytes(artifact), source=artifact["path"]):
        evidence = episode.get("evidence", {})
        if not isinstance(evidence, dict) or any(
            not isinstance(value, str) for value in (episode.get("task_id"), evidence.get("attempt_id"))
        ):
            raise StageError("episode lacks original task/attempt identity evidence")
        key = (episode.get("task_id"), evidence.get("attempt_id"))
        if key not in expected:
            raise StageError("episode is outside the frozen paired schedule")
        binding = {
            **expected[key],
            "cell": cell,
            "plan_hash": content_digest(plan),
            "agent": plan["factors"]["agents"][int(cell[1])],
            "model": plan["factors"]["models"][int(cell[2])],
            "origin": plan["origin"],
        }
        # Fixture latency is explicitly separate from the actual runner's elapsed time.
        extra = {"fixture_latency"} if plan["origin"]["mode"] == "fixture" else set()
        if set(evidence) - {"completion_usage", "completion_status"} != set(binding) | extra or any(
            canonical(evidence.get(k)) != canonical(v) for k, v in binding.items()
        ):
            raise StageError("episode candidate/schedule/producer identity mismatch")
        metrics = episode.get("metrics")
        if not isinstance(metrics, dict) or type(metrics.get("success")) is not bool:
            raise StageError("episode lacks original metrics/explicit verified success")
        if not {"tokens_used", "wall_time_s"} <= metrics.keys():
            raise StageError("episode lacks measured resource costs")
        task = next(t["task"] for t in plan["tasks"] if t["task"]["task_id"] == key[0])
        execution = assess_episode(episode, task=task, budget=plan["budget"])
        rows.append(
            {
                **expected[key],
                "success": metrics["success"],
                "tokens": metrics["tokens_used"],
                "latency": evidence["fixture_latency"] if extra else metrics["wall_time_s"],
                "execution": execution,
            }
        )
    return rows


def execute_crossed(authority: Any, plan_id: str, *, allow_unsandboxed: bool = False) -> dict[str, Any]:
    from admin.runtime_protocol import campaign_lock, writable

    with campaign_lock(authority.store.root):
        writable(authority)
        return _execute_crossed(authority, plan_id, allow_unsandboxed=allow_unsandboxed)


def _execute_crossed(authority: Any, plan_id: str, *, allow_unsandboxed: bool = False) -> dict[str, Any]:
    """Run the frozen four-cell experiment through the existing policy/episode executor.

    Only a fixture authority can use scripted completion artifacts. Production uses the
    authenticated representation-bound serving adapter on every completion.
    """
    from dataclasses import replace

    from admin.artifacts import content_digest
    from admin.candidates import bound_record, file_identity
    from admin.serving_identity import TrustedServing, validate_completion_usage
    from hermes.cotraining import CELLS, crossed_report
    from hermes.protocol import DIALECTS
    from hermesbench.policy import ServedModelPolicy
    from hermesbench.runner import LocalToolExecutor, run_episode
    from hermesbench.sink import JsonlEpisodeSink
    from hermesbench.tasks import Task

    plan = authority.plan(plan_id)
    with authority.store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM experiment_runs WHERE id=?", (plan_id,)).fetchone():
            raise StageError("experiment already started; failed/partial evidence cannot be retried on spent families")
        db.execute("INSERT INTO experiment_runs VALUES (?,'running')", (plan_id,))
    directory = authority.experiments.root / plan_id.removeprefix("sha256:")
    directory.mkdir()
    source = authority.candidates()
    candidates = [source.resolve(plan[k])["payload"] for k in ("old", "new")]
    matrix = {
        k: plan[k]
        for k in (
            "policy",
            "policy_hash",
            "factors",
            "schedule",
            "evaluator",
            "workload",
            "environment",
            "budget",
            "sampling",
        )
    }
    matrix["cells"] = {}
    logs, attestations = {}, {}
    try:
        for cell in CELLS:
            # Revalidate the original authority, then execute captured in-memory artifacts.
            authority.plan(plan_id)
            agent = candidates[int(cell[1])]["agent_record"]
            model = candidates[int(cell[2])]
            config = plan["serving"][model["model_id"]]
            remote = None
            script = None
            if "fixture" in config:
                if authority.identity["mode"] != "fixture" or set(config) != {"fixture"}:
                    raise StageError("fixture execution cannot confer production evidence")
                script = bound_record(config["fixture"])
                if (
                    script.get("schema") != "spark-serving-fixture-v1"
                    or script.get("origin") != authority.identity
                    or script.get("model_id") != model["model_id"]
                ):
                    raise StageError("fixture serving issuer/model mismatch")
            else:
                remote = TrustedServing(
                    config,
                    model_id=model["model_id"],
                    origin=authority.identity,
                    sampling=plan["sampling"],
                    budget=plan["budget"],
                )
            log = directory / (cell + ".jsonl")
            tasks = {item["task"]["task_id"]: item["task"] for item in plan["tasks"]}
            with JsonlEpisodeSink(log, keep_trajectories=True) as sink:
                for index, attempt in enumerate(plan["schedule"]):
                    task = replace(Task.from_record(tasks[attempt["task_id"]]), max_steps=plan["budget"]["max_steps"])
                    if any(t not in agent["tool_schemas"] for t in task.tools):
                        raise StageError("executed agent lacks an actual task tool schema")
                    fixture_row = None
                    completion_usage = []
                    completion_status = []

                    def observe_usage(usage):
                        # Preserve spent responses even when validation aborts before
                        # an EpisodeResult exists. No prompts/credentials are logged.
                        snapshot = json.loads(json.dumps(usage, allow_nan=False))
                        completion_usage.append(snapshot)
                        with (directory / "completion-usage.jsonl").open("a") as journal:
                            journal.write(
                                json.dumps({"cell": cell, **attempt, "usage": snapshot}, allow_nan=False) + "\n"
                            )
                            journal.flush()

                    def observe_finish(reason):
                        snapshot = json.loads(json.dumps(reason, allow_nan=False))
                        completion_status.append(snapshot)
                        with (directory / "completion-status.jsonl").open("a") as journal:
                            journal.write(
                                json.dumps({"cell": cell, **attempt, "finish_reason": snapshot}, allow_nan=False) + "\n"
                            )
                            journal.flush()

                    if script is not None:
                        fixture_row = script["agents"][candidates[int(cell[1])]["agent_id"]][attempt["task_id"]][
                            attempt["attempt_id"]
                        ]
                        # v1 script entries are complete returned responses. An
                        # explicit status list models provider cutoffs/continuations.
                        statuses = fixture_row.get("finish_reasons", ["stop"] * len(fixture_row["responses"]))
                        if not isinstance(statuses, list) or len(statuses) != len(fixture_row["responses"]):
                            raise StageError("fixture completion statuses do not match scripted responses")
                        responses = iter(zip(fixture_row["responses"], statuses, strict=True))

                        def complete(messages, *, tools=None):
                            assert fixture_row is not None
                            try:
                                text, reason = next(responses)
                            except StopIteration as exc:
                                raise StageError("fixture exhausted exact attempt responses") from exc
                            usage = {
                                k: fixture_row[k] for k in ("prompt_tokens", "completion_tokens") if k in fixture_row
                            }
                            observe_usage(usage)
                            observe_finish(reason)
                            validate_completion_usage(usage, max_tokens=plan["budget"]["max_tokens"])
                            return text, usage
                    else:
                        assert remote is not None
                        complete = remote.completion(
                            attempt["seed"], usage_observer=observe_usage, finish_observer=observe_finish
                        )
                    dialect = DIALECTS[agent["dialect"]]
                    policy = ServedModelPolicy(
                        complete=complete,
                        dialect=dialect,
                        tool_schemas=agent["tool_schemas"],
                        system=agent["system"],
                        scratch_pad=dialect.supports_scratch_pad,
                        native_tool_messages=agent["native_tool_messages"],
                    )
                    result = run_episode(
                        task,
                        policy,
                        LocalToolExecutor(
                            allow_unsandboxed=allow_unsandboxed, timeout_s=plan["budget"]["tool_timeout_s"]
                        ),
                        directory / "work" / cell / str(index),
                    )
                    result.evidence = {
                        **attempt,
                        "cell": cell,
                        "plan_hash": content_digest(plan),
                        "agent": candidates[int(cell[1])]["agent_id"],
                        "model": model["model_id"],
                        "origin": plan["origin"],
                        "completion_usage": completion_usage,
                        "completion_status": completion_status,
                    }
                    if authority.identity["mode"] == "fixture":
                        result.evidence["fixture_latency"] = (
                            fixture_row["latency"] if fixture_row else result.metrics.wall_time_s
                        )
                    sink.append(result)
            if remote is not None:
                remote.attest()
                attestations[cell] = remote.observations
            else:
                attestations[cell] = {"fixture_only": True, "artifact": config["fixture"]}
            logs[cell] = file_identity(log)
            matrix["cells"][cell] = {
                **{k: plan[k] for k in ("evaluator", "environment", "workload", "budget", "sampling", "policy_hash")},
                "agent": plan["factors"]["agents"][int(cell[1])],
                "model": model["model_id"],
                "rows": crossed_rows(logs[cell], plan=plan, cell=cell),
            }
        authority.plan(plan_id)
        report = crossed_report(matrix, require_execution=True)
        artifact = directory / "matrix.json"
        product = {
            "schema": "spark-crossed-evaluation-v1",
            "origin": authority.experiments.identity,
            "plan_id": plan_id,
            "matrix": matrix,
            "report": report,
            "logs": logs,
            "serving": attestations,
            "fixture_only": authority.identity["mode"] == "fixture",
        }
        write_record(artifact, product)
        issued = authority.experiments.put("crossed-evaluation", {**product, "artifact": file_identity(artifact)})
        with authority.store.connect() as db:
            db.execute("UPDATE experiment_runs SET status='complete' WHERE id=?", (plan_id,))
        return issued
    except BaseException:
        with authority.store.connect() as db:
            db.execute("UPDATE experiment_runs SET status='failed' WHERE id=?", (plan_id,))
        raise


def verify_crossed_evaluation(authority: Any, identifier: str) -> dict[str, Any]:
    from admin.artifacts import canonical
    from admin.candidates import bound_record, checked_file
    from hermes.cotraining import CELLS, crossed_report

    evidence = authority.experiments.get(identifier, kind="crossed-evaluation")["payload"]
    plan = authority.plan(evidence["plan_id"])
    with authority.store.connect() as db:
        status = db.execute("SELECT status FROM experiment_runs WHERE id=?", (evidence["plan_id"],)).fetchone()
    if status is None or status[0] != "complete":
        raise StageError("evaluation has no completed producer execution")
    if evidence["origin"] != authority.experiments.identity or evidence["fixture_only"] != (
        authority.identity["mode"] == "fixture"
    ):
        raise StageError("evaluation fixture/issuer mismatch")
    original = bound_record(evidence["artifact"])
    if canonical(original) != canonical({k: v for k, v in evidence.items() if k != "artifact"}):
        raise StageError("evaluation artifact differs from original committed producer")
    matrix = evidence["matrix"]
    if any(
        matrix[k] != plan[k]
        for k in (
            "policy",
            "policy_hash",
            "factors",
            "schedule",
            "evaluator",
            "workload",
            "environment",
            "budget",
            "sampling",
        )
    ):
        raise StageError("evaluation differs from frozen experiment")
    if set(evidence["logs"]) != set(CELLS) or set(evidence["serving"]) != set(CELLS):
        raise StageError("evaluation is missing four-cell execution/serving evidence")
    for cell in CELLS:
        if crossed_rows(evidence["logs"][cell], plan=plan, cell=cell) != matrix["cells"][cell]["rows"]:
            raise StageError("exported attempts disagree with original executed episodes")
        serving = evidence["serving"][cell]
        if authority.identity["mode"] == "production" and (not isinstance(serving, list) or not serving):
            raise StageError("production evaluation lacks verified serving identity")
        if isinstance(serving, dict) and "artifact" in serving:
            checked_file(serving["artifact"])
    if crossed_report(matrix, require_execution=True) != evidence["report"]:
        raise StageError("evaluation report does not match strict numerical computation")
    return evidence

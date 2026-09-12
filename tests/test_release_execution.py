"""Real CPU episode exports: completeness is separate from success and costs."""

import copy
from dataclasses import asdict

import pytest

from admin.artifacts import content_digest
from admin.candidates import file_identity
from admin.episode_evidence import assess_episode
from admin.evaluation import crossed_rows
from admin.serving_identity import validate_completion_usage
from hermes.protocol import DIALECTS
from hermesbench.policy import ServedModelPolicy
from hermesbench.runner import LocalToolExecutor, run_episode
from hermesbench.sink import JsonlEpisodeSink, read_episodes
from hermesbench.tasks import Task

CALL = "<tool_call>\n<function=terminal>\n<parameter=command>\nprintf 4 > answer.txt\n</parameter>\n</function>\n</tool_call>"
SCHEMAS = {
    "terminal": {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}
}


def produced_episode(root, variant="valid", *, prompt=5000, output=1024):
    task = Task(
        task_id="t",
        prompt="Write four into answer.txt.",
        tools=("terminal",),
        verify='test "$(cat answer.txt)" = 4',
        hidden_verify='test "$(cat answer.txt)" = 4',
        max_steps=4,
    )
    responses = [CALL, "Finished."]
    if variant == "failed":
        responses[0] = CALL.replace("printf 4", "printf 5")
    elif variant == "recovered":
        responses.insert(0, "<tool_call>garbage</tool_call>")
    elif variant == "truncated":
        responses = [CALL] * 5
    elif variant == "stalled":
        responses = [CALL, ""]
    elif variant == "malformed_incomplete":
        responses = ["<tool_call>garbage</tool_call>"] * (task.max_reasoning_steps + 1)
    elif variant == "setup":
        from dataclasses import replace

        task = replace(task, setup="exit 19")
    elif variant == "no_tools":
        responses = ["Unable to solve this task."]
    scripted = iter(responses)
    usage = []
    statuses = []

    def complete(messages, *, tools=None):
        text = next(scripted)
        raw = {"prompt_tokens": prompt, "completion_tokens": output}
        usage.append(raw)
        statuses.append("stop")
        validate_completion_usage(raw, max_tokens=1024)
        return text, raw

    policy = ServedModelPolicy(complete, DIALECTS["qwen35"], SCHEMAS, system="CPU response fixture")
    result = run_episode(task, policy, LocalToolExecutor(allow_unsandboxed=True), root / "work")
    schedule = [{"task_id": "t", "family_id": "f", "attempt_id": "0", "seed": 0}]
    plan = {
        "schedule": schedule,
        "tasks": [{"task": asdict(task)}],
        "factors": {"agents": ["a0", "a1"], "models": ["m0", "m1"]},
        "origin": {"mode": "fixture", "namespace": "release-episode"},
        "budget": {"max_tokens": 1024},
    }
    result.evidence = {
        **schedule[0],
        "cell": "Q00",
        "plan_hash": content_digest(plan),
        "agent": "a0",
        "model": "m0",
        "origin": plan["origin"],
        "fixture_latency": 1.0,
        "completion_usage": usage,
        "completion_status": statuses,
    }
    log = root / "episode.jsonl"
    with JsonlEpisodeSink(log, keep_trajectories=True) as sink:
        sink.append(result)
    return next(read_episodes(log)), plan, log


@pytest.mark.parametrize(
    "variant,eligible,success",
    [
        ("valid", True, True),
        ("failed", True, False),
        ("recovered", True, True),
        ("no_tools", True, False),
        ("truncated", False, True),
        ("stalled", False, True),
        ("malformed_incomplete", False, False),
        ("setup", False, False),
    ],
)
def test_actual_original_envelopes_preserve_outcomes_and_full_usage(tmp_path, variant, eligible, success):
    episode, plan, log = produced_episode(tmp_path, variant)
    original = log.read_bytes()
    row = crossed_rows(file_identity(log), plan=plan, cell="Q00")[0]
    assert row["success"] is success is episode["metrics"]["success"]
    assert row["tokens"] == episode["metrics"]["tokens_used"]
    assert row["execution"]["eligible"] is eligible
    assert bool(row["execution"]["reasons"]) is not eligible
    assert log.read_bytes() == original
    if variant == "setup":
        assert row["tokens"] == 0
    else:
        # Each output is exactly at its limit, despite prompts and totals above it.
        assert row["tokens"] == len(episode["evidence"]["completion_usage"]) * 6024
    if variant == "recovered":
        assert episode["metrics"]["malformed_turns"] > 0
        assert episode["metrics"]["protocol_clean"] is False
        assert row["tokens"] == 18072


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    return produced_episode(tmp_path_factory.mktemp("complete-envelope"))


@pytest.mark.parametrize(
    "variant",
    [
        "missing_integrity",
        "missing_verifier",
        "missing_trajectory",
        "missing_hidden",
        "missing_completion",
        "summary_conflict",
        "unassessed",
        "invalid_integrity",
        "outer_truncated",
        "outer_success",
        "trajectory_success",
        "missing_result",
        "bad_result_type",
        "duplicate_call",
        "wrong_task",
        "wrong_count",
        "missing_final_flag",
        "not_executed",
        "nonfinal",
        "empty_final",
        "continued_after_final",
        "bad_step",
        "bad_args",
        "bad_content",
        "bad_state",
        "empty_state",
        "bad_state_goal",
        "bad_state_facts",
        "bad_abstention",
        "contradictory_abstention",
        "bad_source",
        "protocol_contradiction",
        "usage_contradiction",
        "usage_limit",
        "usage_bool",
        "usage_missing",
        "missing_status",
        "status_count",
        "status_length",
        "status_unknown",
        "status_type",
        "unfinished_tool_calls",
    ],
)
def test_missing_contradictory_and_structurally_invalid_envelopes_refuse(original, tmp_path, variant):
    import json

    episode, plan, _ = original
    row = copy.deepcopy(episode)
    metrics, trajectory = row["metrics"], row["trajectory"]
    if variant.startswith("missing_") and variant.split("missing_")[1] in {
        "integrity",
        "verifier",
        "trajectory",
        "hidden",
        "completion",
    }:
        name = variant.split("missing_")[1]
        if name == "hidden":
            del metrics["hidden_passed"]
            del trajectory["metadata"]["hidden_passed"]
        elif name == "completion":
            del row["evidence"]["completion_usage"]
        else:
            del row[{"verifier": "verification"}.get(name, name)]
    elif variant == "summary_conflict":
        row["integrity"]["signals"] = [
            {"code": "protected_path_modified", "severity": "disqualifying", "detail": "changed"}
        ]
    elif variant == "unassessed":
        row["integrity"] = {
            "clean": False,
            "fully_checked": False,
            "disqualified": False,
            "unassessed": ["not checked"],
        }
    elif variant == "invalid_integrity":
        row["integrity"] = {"clean": False, "fully_checked": True, "disqualified": False}
    elif variant == "outer_truncated":
        row["truncated"] = True
    elif variant == "outer_success":
        row["success"] = False
    elif variant == "trajectory_success":
        trajectory["success"] = False
    elif variant == "missing_result":
        trajectory["steps"].pop(1)
    elif variant == "bad_result_type":
        trajectory["steps"][1]["ok"] = 1
    elif variant == "duplicate_call":
        trajectory["steps"].insert(0, copy.deepcopy(trajectory["steps"][0]))
    elif variant == "wrong_task":
        trajectory["task"] = "another prompt"
    elif variant == "wrong_count":
        metrics["tool_calls"] += 1
    elif variant == "missing_final_flag":
        del trajectory["metadata"]["harness_final"]
    elif variant == "not_executed":
        trajectory["metadata"]["executed"] = False
    elif variant == "nonfinal":
        trajectory["steps"][-1]["kind"] = "thinking"
    elif variant == "empty_final":
        trajectory["steps"][-1]["content"] = ""
    elif variant == "continued_after_final":
        trajectory["steps"].insert(0, trajectory["steps"][-1])
    elif variant == "bad_step":
        trajectory["steps"][0]["kind"] = "not-a-step"
    elif variant == "bad_args":
        trajectory["steps"][0]["args"] = []
    elif variant == "bad_content":
        trajectory["steps"][-1]["content"] = True
    elif variant in ("bad_state", "empty_state", "bad_state_goal", "bad_state_facts"):
        trajectory["steps"][0]["state"] = {
            "bad_state": 42,
            "empty_state": {},
            "bad_state_goal": {"goal": 42, "action": "execute"},
            "bad_state_facts": {"goal": "solve", "action": "execute", "known": "not a list"},
        }[variant]
    elif variant == "bad_abstention":
        trajectory["abstention"] = "false"
    elif variant == "contradictory_abstention":
        trajectory["abstention"] = True
    elif variant == "bad_source":
        trajectory["source"] = 42
    elif variant == "protocol_contradiction":
        metrics["protocol_clean"] = False
    elif variant == "usage_contradiction":
        metrics["tokens_used"] += 1
    elif variant == "usage_limit":
        row["evidence"]["completion_usage"][0]["completion_tokens"] = 1025
    elif variant == "usage_bool":
        row["evidence"]["completion_usage"][0]["prompt_tokens"] = True
    elif variant == "usage_missing":
        row["evidence"]["completion_usage"] = []
    elif variant == "missing_status":
        del row["evidence"]["completion_status"]
    elif variant == "status_count":
        row["evidence"]["completion_status"].pop()
    elif variant in ("status_length", "status_unknown", "status_type", "unfinished_tool_calls"):
        row["evidence"]["completion_status"][-1] = {
            "status_length": "length",
            "status_unknown": "unknown",
            "status_type": True,
            "unfinished_tool_calls": "tool_calls",
        }[variant]
    log = tmp_path / "input.jsonl"
    log.write_text(json.dumps(row) + "\n")
    exported = crossed_rows(file_identity(log), plan=plan, cell="Q00")[0]
    assert not exported["execution"]["eligible"]
    assert exported["execution"]["reasons"]
    assert exported["success"] is metrics["success"]
    assert exported["tokens"] == metrics["tokens_used"]


def test_opaque_transcript_strings_do_not_become_authority(original):
    episode, plan, _ = original
    row = copy.deepcopy(episode)
    row["trajectory"]["steps"][1]["content"] = '{"truncated": true, "integrity": null}'
    assert assess_episode(row, task=plan["tasks"][0]["task"], budget=plan["budget"])["eligible"]


def test_supported_structured_state_remains_eligible(original, tmp_path):
    import json

    from hermes.state import ReasoningState
    from hermes.trajectory import THINKING, Step

    episode, plan, _ = original
    row = copy.deepcopy(episode)
    state = ReasoningState(goal="Write the requested result", action="Run the terminal command", known=("four",))
    row["trajectory"]["steps"].insert(0, Step(THINKING, state=state).to_record())
    row["metrics"]["steps"] += 1
    log = tmp_path / "structured.jsonl"
    log.write_text(json.dumps(row) + "\n")
    exported = crossed_rows(file_identity(log), plan=plan, cell="Q00")[0]
    assert exported["execution"] == {"eligible": True, "reasons": []}
    assert exported["success"] is row["metrics"]["success"]
    assert exported["tokens"] == row["metrics"]["tokens_used"]


@pytest.mark.parametrize(
    "variant,eligible,success",
    [
        ("stop", True, True),
        ("length", False, True),
        ("missing", False, True),
        ("unknown", False, True),
        ("content_filter", False, True),
        ("invalid_type", False, True),
        ("unfinished_tool_calls", False, True),
        ("tool_continuation", True, True),
        ("complete_failure", True, False),
        ("public_timeout", True, False),
        ("hidden_timeout", True, False),
    ],
)
def test_actual_serving_status_survives_policy_runner_and_original_sink(
    tmp_path, monkeypatch, variant, eligible, success
):
    from admin.serving_identity import TrustedServing

    monkeypatch.setenv("CPU_COMPLETION_STATUS_KEY", "explicit-fixture-credential")
    verify = 'test "$(cat answer.txt)" = 4'
    task = Task(
        task_id="status-control",
        prompt="Write four into answer.txt.",
        tools=("terminal",),
        verify="sleep 0.2" if variant == "public_timeout" else "false" if variant == "complete_failure" else verify,
        hidden_verify="sleep 0.2" if variant == "hidden_timeout" else verify,
        timeout_s=0.05 if variant.endswith("timeout") else 5,
        max_steps=4,
    )
    adapter = TrustedServing(
        {
            "url": "https://fixture.invalid",
            "api_key_env": "CPU_COMPLETION_STATUS_KEY",
            "alias": "controlled",
            "deployment_id": "controlled",
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
    final_reason = {
        "missing": None,
        "invalid_type": True,
        "unfinished_tool_calls": "tool_calls",
    }.get(variant, variant if variant in {"length", "unknown", "content_filter"} else "stop")
    choices = [{"finish_reason": "stop", "message": {"content": CALL}}, {"message": {"content": "Finished."}}]
    if variant != "missing":
        choices[-1]["finish_reason"] = final_reason
    if variant == "tool_continuation":
        choices[0] = {
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command":"printf 4 > answer.txt"}'}}]
            },
        }
    scripted = iter(choices)
    usage, statuses, responses = [], [], []

    def request(route, payload):
        nonce = payload["nonce"] if route == "/identity" else payload["spark_identity"]["nonce"]
        identity = {**adapter.expected, "nonce": nonce}
        if route == "/identity":
            return identity
        response = {
            "spark_identity": identity,
            "usage": {"prompt_tokens": 5000, "completion_tokens": 1024},
            "choices": [next(scripted)],
        }
        responses.append(response)
        return response

    monkeypatch.setattr(adapter, "_request", request)
    complete = adapter.completion(0, usage_observer=usage.append, finish_observer=statuses.append)
    policy = ServedModelPolicy(complete, DIALECTS["qwen35"], SCHEMAS, system="CPU provider boundary")
    result = run_episode(task, policy, LocalToolExecutor(allow_unsandboxed=True), tmp_path / "work")
    result.evidence = {"completion_usage": usage, "completion_status": statuses}
    log = tmp_path / "provider.jsonl"
    with JsonlEpisodeSink(log, keep_trajectories=True) as sink:
        sink.append(result)
    original_bytes = log.read_bytes()
    episode = next(read_episodes(log))
    assert episode["evidence"]["completion_status"] == [r["choices"][0].get("finish_reason") for r in responses]
    assessment = assess_episode(episode, task=asdict(task), budget={"max_tokens": 1024})
    assert assessment["eligible"] is eligible
    assert bool(assessment["reasons"]) is not eligible
    assert episode["metrics"]["success"] is success
    assert episode["metrics"]["tokens_used"] == 12048
    assert episode["trajectory"]["metadata"]["harness_final"] is False
    assert log.read_bytes() == original_bytes

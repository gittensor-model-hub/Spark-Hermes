"""CPU regression cases at each episode producer/consumer boundary."""

import copy
import dataclasses
import json

import pytest
from competition_support import challenge, rows
from settlement_support import prepare, settle

from hermes.challenge import ChallengeError, episode_metrics_of, from_episode_log
from validator.score import ScoreError, baseline_arm, candidate_arm, read_metrics, validate_execution
from validator.settlement import SettlementStore


def nested(row):
    return {
        "task_id": row["task_id"],
        "setup_failed": row["setup_failed"],
        "disqualified": row["disqualified"],
        "metrics": copy.deepcopy(row),
        "integrity": {
            "clean": row["integrity_clean"],
            "fully_checked": row["integrity_fully_checked"],
            "disqualified": row["disqualified"],
        },
    }


@pytest.mark.parametrize("wrapper", ["metrics", "evidence"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("setup_failed", True),
        ("disqualified", True),
        ("integrity_clean", False),
        ("integrity_fully_checked", False),
        ("max_steps_hit", True),
        ("protocol_clean", False),
        ("public_passed", False),
        ("hidden_passed", False),
        ("tokens_used", 0),
        ("setup_failed", 0),  # Equality to False must not erase the wrong type.
    ],
)
def test_duplicate_assertions_cannot_override_execution(wrapper, key, value, tmp_path):
    row = rows(n=1)[0]
    record = {**row, wrapper: {key: value}}
    with pytest.raises(ChallengeError, match="conflicting"):
        episode_metrics_of(record)
    with pytest.raises(ScoreError):
        validate_execution(record, private_required=True)
    with pytest.raises(ScoreError):
        candidate_arm([record])
    log = tmp_path / "episode.jsonl"
    log.write_text(json.dumps(record) + "\n")
    with pytest.raises(ScoreError):
        read_metrics(log)


@pytest.mark.parametrize("flag", ["truncated", "harness_final", "integrity_disqualified", "max_steps_hit"])
@pytest.mark.parametrize("location", ["outer", "metrics", "evidence", "recursive"])
def test_negative_flags_survive_every_supported_envelope(flag, location):
    record = nested(rows(n=1)[0])
    if location == "outer":
        record[flag] = True
    elif location == "metrics":
        record["metrics"][flag] = True
    elif location == "evidence":
        record["evidence"] = {flag: True}
    else:
        record["metrics"]["evidence"] = {"metrics": {flag: True}}
    with pytest.raises(ScoreError):
        candidate_arm([record])


@pytest.mark.parametrize("field", ["metrics", "evidence", "integrity"])
@pytest.mark.parametrize("value", [None, False, [], "clean"])
def test_malformed_wrappers_cannot_hide_behind_valid_flat_flags(field, value):
    record = {**rows(n=1)[0], field: value}
    with pytest.raises(ScoreError):
        candidate_arm([record])


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("wrapped", [True, False])
def test_baseline_retains_original_input_and_valid_controls_remain_usable(private, wrapped):
    source = challenge(private=private)
    originals = [copy.deepcopy(a.evidence) for a in source.baseline.attempts]
    if wrapped:
        originals = [nested(r) for r in originals]
    opened, refused = from_episode_log(originals, epoch=source.epoch, task_pins={source.task_id: source.task_pins})
    assert len(opened) == 1 and not refused
    baseline = opened[0]
    assert baseline.baseline.attempts[0].evidence == originals[0]
    originals[0]["setup_failed"] = True
    assert baseline.baseline.attempts[0].evidence["setup_failed"] is False
    assert baseline_arm(baseline).passes == 4
    candidates = rows(hidden=True if private else None)
    arm, _, _ = candidate_arm([nested(r) for r in candidates] if wrapped else candidates, private_required=private)
    assert arm.passes == 10


def test_original_baseline_conflict_is_refused_before_challenge_creation():
    source = challenge()
    originals = [nested(a.evidence) for a in source.baseline.attempts]
    originals[3]["metrics"]["setup_failed"] = True
    with pytest.raises(ChallengeError, match="conflicting episode evidence: setup_failed"):
        from_episode_log(originals, epoch=source.epoch, task_pins={source.task_id: source.task_pins})


def test_stored_baseline_consumer_rechecks_raw_envelope():
    source = challenge()
    attempts = list(source.baseline.attempts)
    bad = nested(attempts[3].evidence)
    bad["truncated"] = True
    attempts[3] = dataclasses.replace(attempts[3], evidence=bad)
    source = dataclasses.replace(source, baseline=dataclasses.replace(source.baseline, attempts=tuple(attempts)))
    with pytest.raises(ScoreError, match="truncated"):
        baseline_arm(source)


@pytest.mark.parametrize("attack", ["truncated", "conflicting-integrity"])
def test_settlement_rechecks_nested_execution_before_committing(tmp_path, attack):
    prepare(tmp_path, tokens={"alice": 60000})
    path = tmp_path / "episodes/r-1/alice.jsonl"
    records = [nested(json.loads(line)) for line in path.read_text().splitlines()]
    if attack == "truncated":
        records[3]["truncated"] = True
    else:
        records[3]["metrics"]["integrity_clean"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ScoreError):
        settle(tmp_path)
    assert SettlementStore(tmp_path / "settlement").actions() == []


@pytest.mark.parametrize("case", ["success", "setup", "truncated", "integrity", "protocol"])
def test_real_runner_and_sink_keep_execution_failures_visible(tmp_path, case):
    from hermes.trajectory import FINAL, TOOL_CALL, Step
    from hermesbench.runner import LocalToolExecutor, run_suite
    from hermesbench.sink import JsonlEpisodeSink
    from hermesbench.tasks import Task

    task = Task.from_record(
        {
            "task_id": "cpu-envelope-control",
            "prompt": "Create done.txt without changing protected.txt",
            "tools": ["terminal"],
            "setup": "false" if case == "setup" else "echo original > protected.txt",
            "verify": "test -f done.txt",
            "protected_paths": ["protected.txt"],
            "max_steps": 4,
            "timeout_s": 5,
        }
    )

    class Policy:
        # Explicitly synthetic CPU fixture tokens; no model inference.
        tokens_used = 42
        parse_failures = 1 if case == "protocol" else 0

        def next_steps(self, task, history):
            if not history or case == "truncated":
                command = "touch done.txt && test -f done.txt"
                if case == "integrity":
                    command += " && echo changed > protected.txt"
                return [Step(kind=TOOL_CALL, tool="terminal", args={"command": command}, call_id="fixture")]
            return [Step(kind=FINAL, content="done")]

    log = tmp_path / "runner.jsonl"
    with JsonlEpisodeSink(log) as sink:
        _, results = run_suite(
            [task], lambda _: Policy(), LocalToolExecutor(allow_unsandboxed=True), tmp_path / "ws", sink=sink
        )
    # Exercise both the runner export and actual sink output, which differ in shape.
    for records in ([results[0].to_record()], read_metrics(log)):
        if case == "success":
            assert candidate_arm(records, private_required=False)[0].passes == 1
        else:
            with pytest.raises(ScoreError):
                candidate_arm(records, private_required=False)

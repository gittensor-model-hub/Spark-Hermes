"""Complete evidence, producer semantics, and durable competition refusal boundaries.

All tokens and policies used here are labelled CPU fixtures, without model inference.
"""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from competition_support import EPOCH, admit_receipt, challenge, rows, window
from settlement_support import prepare, settle

from hermes.challenge import episode_metrics_of, from_episode_log, unverifiable_tasks
from hermes.evidence_json import evidence_object
from hermesbench.integrity import IntegrityReport, IntegritySignal
from hermesbench.sink import SinkError, decode_episodes, read_episode_prefix, read_episodes
from validator.crown import CrownError
from validator.intake import Intake
from validator.judge import judge_round
from validator.round_loop import LoopError, open_from_packet
from validator.score import ScoreError, baseline_arm, candidate_arm, read_metrics
from validator.settlement import SettlementError, SettlementStore
from validator.store import RoundStore, StoreError

BAD_RECORDS = [
    b'{"setup_failed":true,"setup_failed":false}\n',
    b'{"metrics":{"setup_failed":true,"setup_failed":false}}\n',
    b'{"evidence":{"metrics":{"success":false,"success":true}}}\n',
    b'{"integrity":{"signals":[{"severity":"disqualifying","severity":"warning"}]}}\n',
    b'{"same":false,"same":false}\n',
    b'{"cost":NaN}\n',
    b'{"payload":{"value":Infinity}}\n',
    b'{"cost":-Infinity}\n',
    b'{"payload":{"value":1e999}}\n',
    b"[]\n",
    b"null\n",
    b"true\n",
    b'"record"\n',
    b"{broken}\n",
    b'{"task_id":"extra"}',  # Even syntactically complete without the sink terminator.
    b'{"task_id":"extra","metrics":',
    b'{"payload":"\xff"}\n',
]


@pytest.mark.parametrize("bad", BAD_RECORDS)
def test_complete_decoders_never_expose_a_valid_prefix(tmp_path, bad):
    log = tmp_path / "episodes.jsonl"
    raw = json.dumps(rows(n=1)[0]).encode() + b"\n" + bad
    log.write_bytes(raw)
    for read in (lambda: next(read_episodes(log)), lambda: read_metrics(log), lambda: decode_episodes(raw)):
        with pytest.raises((SinkError, ScoreError)):
            read()
    if not bad.endswith(b"\n"):
        assert len(list(read_episode_prefix(log))) == 1
    else:
        with pytest.raises(SinkError):
            list(read_episode_prefix(log))
    assert log.read_bytes() == raw


@pytest.mark.parametrize("bad", BAD_RECORDS)
def test_invalid_raw_candidate_leaves_no_card_verdict_or_outbox(tmp_path, monkeypatch, bad):
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture", namespace="byte-boundaries")
    win = window(store)
    intake = Intake(tmp_path / "bundles", tmp_path / "receipts", mode="fixture", namespace=store.identity["namespace"])
    receipt = intake.accept(round_id=win.round_id, miner_id="alice", files={"SOUL.md": "CPU fixture"}, now=10)
    admit_receipt(store, intake, receipt, monkeypatch)
    win = store.load(win.round_id)
    win.freeze(now=30, reason="CPU fixture")
    store.save(win)
    evidence = rows(origin=store.identity, digest=receipt.bundle_sha256)
    log = tmp_path / "input.jsonl"
    log.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in evidence) + bad)
    results = judge_round(
        round_id=win.round_id,
        run=lambda *_: log,
        model_revision=EPOCH["model_revision"],
        store=store,
        intake=intake,
        workspace=tmp_path / "work",
        scorecard_dir=tmp_path / "cards",
        settle=False,
    )
    assert len(results) == 1 and not results[0].ok and results[0].scorecard is None
    assert not list((tmp_path / "cards").glob("*.json"))
    assert not store.load(win.round_id).verdicts
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace=store.identity["namespace"])
    assert settlement.actions() == []
    with pytest.raises(SettlementError, match="no committed settlement"):
        settlement.record(win.round_id)


@pytest.mark.parametrize("bad", BAD_RECORDS)
def test_round_reconstruction_refuses_bad_original_bytes_before_persistence(tmp_path, bad):
    source = challenge()
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(source.to_record()))
    log = tmp_path / "baseline.jsonl"
    log.write_bytes(b"".join(json.dumps(a.evidence).encode() + b"\n" for a in source.baseline.attempts) + bad)
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture")
    with pytest.raises(LoopError, match="complete challenge evidence"):
        open_from_packet(packet_path=packet, episodes_path=log, round_id="r-1", hours=1, store=store)
    assert not store.path_for("r-1").exists()


def reports():
    clean = IntegrityReport().to_record()
    cases = []
    for code, severity in (
        ("new_detector", "disqualifying"),
        ("ordinary_warning", "warning"),
        ("integrity_partial", "warning"),
    ):
        report = IntegrityReport((IntegritySignal(code, severity, "CPU finding"),)).to_record()
        cases.append(report)  # Consistent failures must still refuse credit.
        cases.append({**report, "clean": True, "fully_checked": True, "disqualified": False})
    cases.extend(
        [
            {**clean, "unassessed": ["missing workspace"]},
            {**clean, "unassessed": "missing workspace"},
            {**clean, "unassessed": [None]},
            {**clean, "fully_checked": False},
            {**clean, "disqualified": True},
            {**clean, "clean": False},
            {"clean": True, "fully_checked": False, "disqualified": False},
            {"clean": True, "fully_checked": True, "disqualified": False, "unassessed": ["unobserved"]},
        ]
    )
    for signals in (
        None,
        "clean",
        {},
        [None],
        ["finding"],
        [{}],
        [{"code": "x", "severity": "info", "detail": "x"}],
        [{"code": "x", "severity": "warning"}],
        [{"code": 1, "severity": "warning", "detail": "x"}],
        [{"code": "x", "severity": "warning", "detail": []}],
    ):
        cases.append({**clean, "signals": signals})
    return cases


@pytest.mark.parametrize("report", reports())
def test_integrity_details_constrain_candidate_and_original_baseline(report):
    record = rows(n=1)[0]
    # Match flat flags to the supplied report: refusal must check the report itself.
    record.update(
        integrity_clean=report["clean"],
        integrity_fully_checked=report["fully_checked"],
        disqualified=report["disqualified"],
        integrity=report,
    )
    with pytest.raises(ScoreError):
        candidate_arm([record])
    source = challenge()
    attempts = list(source.baseline.attempts)
    attempts[0] = dataclasses.replace(attempts[0], evidence=record)
    invalid = dataclasses.replace(source, baseline=dataclasses.replace(source.baseline, attempts=tuple(attempts)))
    with pytest.raises(ScoreError):
        baseline_arm(invalid)


@pytest.mark.parametrize("details", [{}, {"unassessed": []}, {"signals": []}, {"signals": [], "unassessed": []}])
def test_summary_only_integrity_does_not_invent_detector_observations(details):
    report = dict(clean=True, fully_checked=True, disqualified=False, **details)
    record = {**rows(n=1)[0], "integrity": report}
    assert episode_metrics_of(record)["integrity"] == report
    assert set(episode_metrics_of(record)["integrity"]) == set(report)
    assert candidate_arm([record])[0].passes == 1


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("keep_trajectories", [True, False])
def test_actual_producers_preserve_stamps_and_failed_valid_baseline_attempts(tmp_path, private, keep_trajectories):
    from hermes.trajectory import FINAL, TOOL_CALL, Step
    from hermesbench.runner import LocalToolExecutor, run_suite, verify_digest
    from hermesbench.sink import JsonlEpisodeSink
    from hermesbench.tasks import Task
    from hermesbench.verify import redact_for_release

    task = Task.from_record(
        dict(
            task_id="cpu-complete-evidence",
            prompt="create done",
            tools=["terminal"],
            verify="test -f done",
            hidden_verify="test -f done" if private else None,
            max_steps=4,
            timeout_s=5,
        )
    )
    public = redact_for_release(dataclasses.asdict(task), salt="CPU-fixture-master-salt")
    task = dataclasses.replace(task, metadata=public["metadata"])
    origin = dict(mode="fixture", namespace="cpu-producer", issuer="fixture")
    context = dict(epoch=EPOCH, origin=origin, round_id="r-1", bundle_sha256="sha256:" + "a" * 64)
    policies = []

    class Policy:
        tokens_used = 87000  # Scripted accounting, no candidate-model inference.

        def __init__(self, passed):
            self.passed = passed

        def next_steps(self, task, history):
            if not history:
                return [
                    Step(
                        kind=TOOL_CALL,
                        tool="terminal",
                        call_id="fixture",
                        args={"command": "touch done" if self.passed else "true"},
                    )
                ]
            return [Step(kind=FINAL, content="CPU fixture complete")]

    def factory(_):
        policy = Policy(len(policies) < 4)
        policies.append(policy)
        return policy

    log = tmp_path / "sink.jsonl"
    with JsonlEpisodeSink(log, keep_trajectories=keep_trajectories) as sink:
        _, results = run_suite(
            [task],
            factory,
            LocalToolExecutor(allow_unsandboxed=True),
            tmp_path / "work",
            repeats=10,
            sink=sink,
            evaluation_context=context,
        )
    original = list(read_episodes(log))
    assert "verification" in original[0]
    assert ("trajectory" in original[0]) is keep_trajectories
    for records in (original, [r.to_record() for r in results]):
        assert unverifiable_tasks(records, current={task.task_id: verify_digest(task)}) == {}
        opened, refused = from_episode_log(
            records,
            epoch=EPOCH,
            task_pins={
                task.task_id: {
                    "private_check_required": private,
                    "verify_digest": verify_digest(task),
                    "hidden_verify_commitment": task.hidden_verify_commitment,
                }
            },
        )
        assert not refused and len(opened) == 1
        assert [a.evidence for a in opened[0].baseline.attempts] == records
        assert baseline_arm(opened[0], origin=origin).passes == 4
        arm = candidate_arm(records, private_required=private)[0]
        assert arm.passes == 4 and arm.attempts == 10 and sum(arm.tokens) == 870000
    # Transcripts/tool payloads and unknown metadata are not execution authority.
    opaque = results[0].to_record()
    opaque["trajectory"]["steps"].append({"payload": {"success": False, "integrity": {"clean": False}}})
    opaque["trajectory"]["metadata"]["transcript"] = {"verification": {"passed": False}}
    opaque["verification"]["stdout"] = '{"setup_failed":true,"setup_failed":false}'
    assert candidate_arm([opaque], private_required=private)[0].passes == 1
    # Known producer assertions remain authoritative on both serialization paths.
    for path, value in [
        (("verification", "passed"), False),
        (("verification", "passed"), "true"),
        (("verification", "exit_code"), 1),
        (("verification", "timed_out"), True),
        (("trajectory", "success"), False),
        (("trajectory", "task_id"), "wrong-task"),
        (("trajectory", "metadata", "public_passed"), False),
        (("trajectory", "metadata", "hidden_passed"), False if private else True),
        (("trajectory", "metadata", "integrity_disqualified"), True),
        (("trajectory", "metadata", "verify_exit_code"), 1),
        (("trajectory", "metadata", "executed"), False),
        (("trajectory", "metadata", "harness_final"), True),
    ]:
        changed = copy.deepcopy(results[0].to_record())
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ScoreError):
            candidate_arm([changed], private_required=private)
    invalid = dataclasses.replace(results[0], verification=dataclasses.replace(results[0].verification, passed=False))
    invalid_log = tmp_path / "invalid-sink.jsonl"
    with JsonlEpisodeSink(invalid_log, keep_trajectories=True) as sink:
        sink.append(invalid)
    with pytest.raises(ScoreError):
        candidate_arm(read_metrics(invalid_log), private_required=private)


def test_read_metrics_decodes_one_snapshot_only(tmp_path, monkeypatch):
    log = tmp_path / "episodes.jsonl"
    raw = json.dumps(rows(n=1)[0]).encode() + b"\n"
    calls = []
    original = Path.read_bytes

    def read(path):
        if path == log:
            calls.append(path)
            return raw if len(calls) == 1 else b"[]\n"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert read_metrics(log) == rows(n=1)
    assert len(calls) == 1


@pytest.mark.parametrize("surface", ["packet", "snapshot", "identity", "scorecard"])
def test_adjacent_competition_objects_reject_recursive_duplicate_assertions(tmp_path, surface):
    if surface == "packet":
        source = challenge()
        packet = tmp_path / "packet.json"
        packet.write_text(json.dumps(source.to_record()).replace('"epoch_id":', '"epoch_id":"stale","epoch_id":'))
        log = tmp_path / "baseline.jsonl"
        log.write_text("".join(json.dumps(a.evidence) + "\n" for a in source.baseline.attempts))
        store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture")
        with pytest.raises(LoopError, match="duplicate JSON key"):
            open_from_packet(packet_path=packet, episodes_path=log, round_id="r-1", hours=1, store=store)
        assert not store.path_for("r-1").exists()
    elif surface == "identity":
        root = tmp_path / "identity"
        store = RoundStore(root, require_private=False, mode="fixture")
        path = root / ".identity"
        path.write_text(json.dumps(store.identity).replace('"mode":', '"mode":"production","mode":'))
        with pytest.raises(StoreError, match="duplicate JSON key"):
            RoundStore(root, require_private=False)
    else:
        store, settlement = prepare(tmp_path, tokens={"alice": 60000})
        if surface == "snapshot":
            path = store.path_for("r-1")
            key = '"setup_failed":'
            expected = StoreError
        else:
            path = tmp_path / "cards/r-1-alice.json"
            key = '"accepted":'
            expected = CrownError
        raw = path.read_text()
        assert key in raw
        path.write_text(raw.replace(key, key + "false," + key))
        with pytest.raises(expected, match="duplicate JSON key"):
            settle(tmp_path)
        assert settlement.actions() == []
        with pytest.raises(SettlementError, match="no committed settlement"):
            settlement.record("r-1")


@pytest.mark.parametrize("attack", ["torn", "duplicate", "integrity", "snapshot-race"])
def test_settlement_invalid_evidence_creates_no_outcome_or_actions(tmp_path, monkeypatch, attack):
    store, settlement = prepare(tmp_path, tokens={"alice": 60000})
    log = tmp_path / "episodes/r-1/alice.jsonl"
    good = log.read_bytes()
    if attack in ("torn", "snapshot-race"):
        bad = good + b'{"unfinished":'
    elif attack == "duplicate":
        bad = good.replace(b'"setup_failed": false', b'"setup_failed": true, "setup_failed": false')
    else:
        records = [json.loads(line) for line in good.splitlines()]
        records[0]["integrity"] = {
            **IntegrityReport().to_record(),
            "signals": [IntegritySignal("new_disqualifier", "disqualifying", "CPU fixture").to_record()],
        }
        bad = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    if attack == "snapshot-race":
        original = Path.read_bytes
        reads = []

        def race(path):
            if path.resolve() == log.resolve():
                reads.append(path)
                return good if len(reads) == 2 else bad
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", race)
    else:
        log.write_bytes(bad)
    before = store.path_for("r-1").read_bytes()
    with pytest.raises(ScoreError):
        settle(tmp_path)
    assert store.path_for("r-1").read_bytes() == before
    assert settlement.actions() == []
    with pytest.raises(SettlementError, match="no committed settlement"):
        settlement.record("r-1")


def test_valid_settlement_hashes_the_decoded_snapshot(tmp_path):
    _, settlement = prepare(tmp_path, tokens={"alice": 60000})
    result = settle(tmp_path)
    record = settlement.record("r-1")
    assert result == record
    entry = record["entries"][0]
    assert (
        entry["episodes"]["sha256"]
        == "sha256:" + hashlib.sha256(Path(entry["episodes"]["path"]).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("raw", [b"[]", b'{"nested":{"x":1,"x":2}}', b'{"nested":[{"x":NaN}]}'])
def test_object_parser_rejects_ambiguous_nonobject_and_nonfinite_artifacts(raw):
    with pytest.raises(ValueError):
        evidence_object(raw)

"""Typed source/store authority regressions. CPU fixtures, no live GitHub or inference."""

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from competition_support import GitHubTransport, rows, window
from settlement_support import ActionTransport, prepare, settle

from eval.rollout_track import check_scope
from eval.strategy_track import Commitment
from hermes.announce import close_round, commit_round, open_seed
from hermes.seed import CLOSED, COMMITTED, Round, SeedError, open_round
from validator.intake import Intake
from validator.judge import judge_one
from validator.pr_admission import AdmissionError, GitHubSource, admission_for, admit, assignment_identity, main
from validator.score import ScoreError, score
from validator.settlement import GitHubActions, SettlementStore
from validator.store import RoundStore, StoreError

BAD_FIELDS = {
    "schema_version": [None, True, False, 1.0, 1.875, "1", 0, 2, {}, [], ""],
    "replicas": [None, True, False, 4.0, 4.875, "4", 0, -1, 5, {}, [], ""],
    "round_id": [None, True, 1, 1.5, {}, [], ""],
    "seed": [None, True, 12345678901234567890123456789012, 1.5, {}, [], "", "short"],
    "state": [None, True, 1, 1.0, "1", {}, [], "", "unknown"],
    "commitment": [None, True, 1, 1.5, {}, [], "", "sha256:" + "0" * 64],
    "task_ids": [
        None,
        True,
        1,
        1.5,
        "task",
        {},
        {"tc-log-rotation-order": "other"},
        [],
        [""],
        [None],
        [1],
        [True],
        [{}],
        [[]],
        ["x", "x"],
    ],
    "miner_ids": [
        None,
        True,
        1,
        1.5,
        "alice",
        {},
        {"alice": "other"},
        [],
        [""],
        [None],
        [1],
        [True],
        [{}],
        [[]],
        ["alice", "alice"],
    ],
}
VARIANTS = [(field, index, value) for field, values in BAD_FIELDS.items() for index, value in enumerate(values)]
VARIANTS += [(field, "missing", None) for field in BAD_FIELDS]


def altered(record, field, index, value):
    result = copy.deepcopy(record)
    if index == "missing":
        del result[field]
    else:
        result[field] = value
    return result


@pytest.fixture
def authority(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    store = RoundStore(tmp_path / "rounds", mode="fixture", namespace="typed-round")
    win = window(store)
    win.assignment = replace(win.assignment, seed="b" * 64)
    store.save(win)
    intake = Intake(tmp_path / "bundles", tmp_path / "receipts.jsonl", mode="fixture", namespace="typed-round")
    receipt = intake.accept(round_id=win.round_id, miner_id="alice", files={"SOUL.md": "CPU fixture"})
    record = Commitment(win.round_id, "alice", receipt.bundle_sha256, (win.task_id,)).to_record()
    transport = GitHubTransport(win.assignment.to_record(), record)
    argv = [
        "--repository",
        transport.repository,
        "--pr",
        "7",
        "--round",
        win.round_id,
        "--store",
        str(store.root),
        "--intake-root",
        str(intake.root),
        "--receipts",
        str(intake.receipts),
    ]
    return store, intake, receipt, transport, argv


@pytest.mark.parametrize("field,index,value", VARIANTS, ids=[f"{f}-{i}" for f, i, _ in VARIANTS])
def test_known_schema_refuses_in_scope_cli_and_persisted_load(authority, tmp_path, field, index, value):
    store, intake, _, transport, argv = authority
    original = store.load("r-1").assignment
    transport.round_record = altered(original.to_record(), field, index, value)
    with pytest.raises(SeedError):
        Round.from_record(transport.round_record)
    assert check_scope(json.loads(transport.head_text), transport.round_record)
    before = store.path_for("r-1").read_bytes()
    receipts = intake.receipts.read_bytes()
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace="typed-round")
    assert main(argv, transport=transport) == 2
    assert store.path_for("r-1").read_bytes() == before
    assert intake.receipts.read_bytes() == receipts
    win = store.load("r-1")
    assert win.admissions == win.submissions == {}
    assert win.snapshot()["verdicts"] == []
    assert settlement.actions() == []
    with settlement.transaction() as db:
        assert db.execute("SELECT count(*) FROM rounds").fetchone()[0] == 0
    assert not list(tmp_path.glob("cards/**/*.json"))
    snapshot = json.loads(before)
    snapshot["assignment"] = transport.round_record
    store.path_for("r-1").write_text(json.dumps(snapshot))
    for override in (None, original):
        with pytest.raises(StoreError, match="invalid stored assignment"):
            store.load("r-1", assignment=override)


@pytest.mark.parametrize(
    "field,index,value", [v for v in VARIANTS if v[0] not in ("schema_version", "commitment") and v[1] != "missing"]
)
def test_direct_constructor_does_not_launder_known_fields(authority, field, index, value):
    assignment = authority[0].load("r-1").assignment
    with pytest.raises(SeedError):
        replace(assignment, **{field: value})


@pytest.mark.parametrize("value", [None, False, 1, 1.0, "", "task", {}, {"task": "other"}])
def test_producer_rejects_non_collection_pools(value):
    with pytest.raises(SeedError):
        open_round("round", value, ["alice"])
    with pytest.raises(SeedError):
        open_round("round", ["task"], value)


@pytest.mark.parametrize("seed", [None, False, 0, 1.0, [], {}])
def test_producer_does_not_generate_seed_for_malformed_input(seed):
    with pytest.raises(SeedError):
        open_round("round", ["task"], ["alice"], seed=seed)


@pytest.mark.parametrize("raw", [None, True, 1, 1.0, "round", [], ["round"]])
def test_round_record_requires_object(raw):
    with pytest.raises(SeedError):
        Round.from_record(raw)


def test_json_lists_and_internal_tuple_producers_remain_distinct(authority):
    assignment = authority[0].load("r-1").assignment
    for field in ("task_ids", "miner_ids"):
        record = assignment.to_record()
        record[field] = tuple(record[field])
        with pytest.raises(SeedError):
            Round.from_record(record)
    assert replace(assignment, task_ids=list(assignment.task_ids)).task_ids == assignment.task_ids


def change_assignment(assignment, field, *, require_owner=True):
    changes = {
        "seed": "Unicode seed is supported Δ: " * 3,
        "task_ids": (*assignment.task_ids, "extra-task"),
        "miner_ids": (*assignment.miner_ids[:-1], "extra-miner"),
        "replicas": 3,
        "round_id": "r-other",
    }
    result = replace(assignment, **{field: changes[field]})
    if require_owner:
        assert result.owns("alice", assignment.task_ids[0]), "both valid assignments must own this task"
    return result


@pytest.mark.parametrize("field", ["seed", "task_ids", "miner_ids", "replicas", "round_id"])
def test_different_valid_source_assignment_cannot_gain_authority(authority, field):
    store, _, _, transport, argv = authority
    assignment = store.load("r-1").assignment
    transport.round_record = change_assignment(assignment, field).to_record()
    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=transport) == 2
    assert store.path_for("r-1").read_bytes() == before
    assert store.load("r-1").admissions == {}


@pytest.mark.parametrize("field", ["seed", "task_ids", "miner_ids", "replicas", "round_id", "missing", "null"])
def test_assignment_only_persisted_change_refuses_all_consumers(authority, tmp_path, field):
    store, intake, receipt, transport, argv = authority
    assert main(argv, transport=transport) == 0
    win = store.load("r-1")
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace="typed-round")
    settlement.activate(store, "r-1", transport.repository)
    original_admissions = copy.deepcopy(win.admissions)
    original = win.assignment
    record = json.loads(store.path_for("r-1").read_text())
    if field == "missing":
        del record["assignment"]
    elif field == "null":
        record["assignment"] = None
    else:
        record["assignment"] = change_assignment(original, field).to_record()
    store.path_for("r-1").write_text(json.dumps(record))
    if field == "null":
        with pytest.raises(StoreError):
            store.load("r-1")
    else:
        win = store.load("r-1")
        assert win.admissions == original_admissions
        with pytest.raises(AdmissionError):
            admission_for(win, "alice")
        called = []
        result = judge_one(
            window=win,
            miner_id="alice",
            intake=intake,
            run=lambda *a: called.append(a),
            model_revision=win.challenge.epoch["model_revision"],
            harness_digest=win.challenge.epoch["harness_digest"],
            workspace=tmp_path / "episodes",
        )
        assert not result.ok and not called and not win.snapshot()["verdicts"]
        with pytest.raises(ScoreError):
            score(
                window=win,
                miner_id="alice",
                rows=rows(origin=store.identity, digest=receipt.bundle_sha256),
                model_revision=win.challenge.epoch["model_revision"],
                harness_digest=win.challenge.epoch["harness_digest"],
            )
    with pytest.raises(ValueError):
        settlement.settle_round(store, scorecards=tmp_path / "cards", episodes=tmp_path / "episodes", round_id="r-1")
    assert settlement.actions() == []
    with settlement.transaction() as db:
        assert db.execute("SELECT count(*) FROM rounds").fetchone()[0] == 0


@pytest.mark.parametrize("state", [COMMITTED, CLOSED])
def test_admission_requires_open_source_and_stored_assignment(authority, state):
    store, _, _, transport, argv = authority
    win = store.load("r-1")
    win.assignment = replace(win.assignment, state=state)
    store.save(win)
    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=transport) == 2
    assert store.path_for("r-1").read_bytes() == before
    transport.round_record["state"] = state
    assert main(argv, transport=transport) == 2


def test_unrevealed_announcement_refuses_cli(authority):
    store, _, _, transport, argv = authority
    transport.round_record = store.load("r-1").assignment.to_record(reveal_seed=False)
    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=transport) == 2
    assert store.path_for("r-1").read_bytes() == before


def test_ordering_admission_retry_and_assignment_lifecycle_keep_identity(authority, tmp_path):
    store, intake, receipt, transport, argv = authority
    win = store.load("r-1")
    # Multiple tasks make canonical set ordering observable for both pools.
    win.assignment = replace(win.assignment, task_ids=(*win.assignment.task_ids, "extra-task"))
    store.save(win)
    transport.round_record = win.assignment.to_record()
    transport.round_record["task_ids"].reverse()
    transport.round_record["miner_ids"].reverse()
    assert main(argv, transport=transport) == 0
    win = store.load("r-1")
    original = admission_for(win, "alice")
    win.assignment = replace(win.assignment, miner_ids=tuple(reversed(win.assignment.miner_ids)))
    store.save(win)
    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=transport) == 0
    assert store.path_for("r-1").read_bytes() == before
    win = store.load("r-1")
    win.assignment = replace(win.assignment, state=CLOSED)
    win.freeze(now=time.time(), reason="CPU fixture")
    store.save(win)
    restored = store.load("r-1")
    assert admission_for(restored, "alice") == original
    assert assignment_identity(restored.assignment) == original["assignment_identity"]
    assert score(
        window=restored,
        miner_id="alice",
        rows=rows(origin=store.identity, digest=receipt.bundle_sha256),
        model_revision=restored.challenge.epoch["model_revision"],
        harness_digest=restored.challenge.epoch["harness_digest"],
    ).accepted


def test_missing_binding_is_not_resealed_by_retry_or_authoritative_read(authority):
    store, intake, _, transport, argv = authority
    assert main(argv, transport=transport) == 0
    win = store.load("r-1")
    del win.admissions["alice"]["assignment_identity"]
    store.save(win)
    before = store.path_for("r-1").read_bytes()
    with pytest.raises(AdmissionError, match="trusted recovery"):
        admission_for(store.load("r-1"), "alice")
    metadata = GitHubSource(transport.repository, transport=transport).collect(7, round_id="r-1")
    with pytest.raises(AdmissionError, match="trusted recovery"):
        admit(metadata=metadata, round_id="r-1", store=store, intake=intake)
    assert store.path_for("r-1").read_bytes() == before


@pytest.mark.parametrize("field", ["seed", "task_ids", "miner_ids", "replicas"])
def test_graded_assignment_mutation_prevents_settlement_credit(tmp_path, field):
    store, settlement = prepare(tmp_path, tokens={"alice": 60000})
    win = store.load("r-1")
    win.assignment = change_assignment(win.assignment, field, require_owner=False)
    store.save(win)
    with pytest.raises(ValueError):
        settle(tmp_path)
    assert settlement.actions() == []
    with settlement.transaction() as db:
        assert db.execute("SELECT count(*) FROM rounds").fetchone()[0] == 0


def test_closed_assignment_still_allows_graded_settlement_and_retry(tmp_path):
    store, settlement = prepare(tmp_path, tokens={"alice": 60000})
    win = store.load("r-1")
    win.assignment = replace(win.assignment, state=CLOSED)
    store.save(win)
    result = settle(tmp_path)
    assert result["outcome"]["winner"]["miner_id"] == "alice"
    assert not result["authorizes_payment"] and not result["authorizes_model_promotion"]
    assert settle(tmp_path) == result
    assert len(settlement.actions()) == 2


@pytest.mark.parametrize("number", [7.0, 7.875, "7", True, False, None, [], {}])
@pytest.mark.parametrize("read", [1, 2])
def test_provider_number_is_an_exact_integer_on_both_source_reads(authority, number, read):
    store, _, _, transport, argv = authority
    count = 0
    if number is True:
        transport.number = 1
        argv[3] = "1"

    def response(args, **kwargs):
        nonlocal count
        result = transport(args, **kwargs)
        if args[-1].endswith(f"pulls/{transport.number}"):
            count += 1
            if count == read:
                value = json.loads(result.stdout)
                value["number"] = number
                result.stdout = json.dumps(value)
        return result

    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=response) == 2
    assert store.path_for("r-1").read_bytes() == before


@pytest.mark.parametrize("number", [1.0, True, "1", None, [], {}])
@pytest.mark.parametrize("kind", ["label_add", "label_remove", "review", "close"])
def test_provider_number_is_exact_for_active_and_historical_actions(monkeypatch, number, kind):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    action = dict(
        repository="org/repo",
        pr=1,
        author="alice",
        head_sha="a" * 40,
        kind=kind,
        key="fixture-key",
        label="crown",
        body="CPU fixture",
    )
    calls = []

    def transport(argv, **kwargs):
        calls.append(argv)
        assert argv[5] == "GET"
        value = (
            {"login": "bot"}
            if argv[-1] == "user"
            else dict(
                number=number,
                base={"repo": {"full_name": "org/repo"}},
                head={"sha": action["head_sha"]},
                user={"login": "alice"},
                state="closed" if kind == "label_remove" else "open",
                merged=kind == "label_remove",
                draft=False,
            )
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    adapter = GitHubActions("org/repo", transport=transport)
    for method in (adapter.reconcile, adapter.apply):
        with pytest.raises(ValueError, match="identity/state"):
            method(action)
    assert all(call[-1] == "user" or call[-1].endswith("pulls/1") for call in calls)


def test_invalid_provider_number_leaves_outbox_pending_and_retry_recovers(tmp_path):
    _, settlement = prepare(tmp_path, tokens={"alice": 60000}, pr_numbers={"alice": 1})
    settle(tmp_path)
    transport = ActionTransport(tmp_path / "github-fixture.json")

    def invalid(argv, **kwargs):
        result = transport(argv, **kwargs)
        if argv[-1].endswith("pulls/1"):
            record = json.loads(result.stdout)
            record["number"] = True
            result.stdout = json.dumps(record)
        return result

    with pytest.raises(ValueError):
        settlement.deliver(GitHubActions("example/spark", transport=invalid))
    assert all(a["status"] == "pending" for a in settlement.actions())
    assert json.loads(transport.path.read_text())["mutations"] == []
    settlement.deliver(GitHubActions("example/spark", transport=transport))
    assert all(a["status"] == "acknowledged" for a in settlement.actions())
    before = transport.path.read_bytes()
    settlement.deliver(GitHubActions("example/spark", transport=transport))
    assert transport.path.read_bytes() == before


@pytest.mark.parametrize("field,index,value", [v for v in VARIANTS if v[0] not in ("seed", "state")])
def test_reveal_cannot_launder_committed_round_fields(tmp_path, field, index, value):
    path, round_ = commit_round(
        round_id="r-1", task_ids=["task"], miner_ids=["alice", "bob", "carol", "dave"], replicas=4, root=tmp_path
    )
    record = altered(json.loads(path.read_text()), field, index, value)
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    with pytest.raises(SeedError):
        open_seed(round_id="r-1", seed=round_.seed, root=tmp_path)
    assert path.read_bytes() == before


def test_genuine_unicode_seed_multi_replica_producer_lifecycle(tmp_path):
    path, committed = commit_round(
        round_id="r-1",
        task_ids=["t2", "t1"],
        miner_ids=["bob", "alice"],
        replicas=2,
        seed="Not restricted to hex Δ: " * 3,
        root=tmp_path,
    )
    assert "seed" not in json.loads(path.read_text())
    path, opened = open_seed(round_id="r-1", seed=committed.seed, root=tmp_path)
    assert Round.from_record(json.loads(path.read_text())).assignments() == committed.assignments()
    identity = assignment_identity(opened)
    _, closed = close_round(round_id="r-1", root=tmp_path)
    assert assignment_identity(Round.from_record(closed)) == identity
    assert len(opened.assignments()) == 4


@pytest.mark.parametrize("variant", ["fractional", "tasks", "miners", "assignment", "pr-number"])
def test_ordinary_admission_subprocess_refuses_without_durable_authority(authority, tmp_path, variant):
    store, intake, _, transport, argv = authority
    record = transport.round_record
    if variant == "fractional":
        record["replicas"] = 4.875
    elif variant == "tasks":
        record["task_ids"] = {record["task_ids"][0]: "other"}
    elif variant == "miners":
        record["miner_ids"] = dict.fromkeys(record["miner_ids"], "other")
    elif variant == "assignment":
        record = change_assignment(store.load("r-1").assignment, "seed").to_record()
    transport.round_record = record
    routes = {}
    for endpoint in [
        "user",
        "repos/example/spark/pulls/7",
        "repos/example/spark/pulls/7/files?per_page=100&page=1",
        f"repos/example/spark/contents/datasets/strategies.jsonl?ref={transport.base}",
        f"repos/example/spark/contents/datasets/strategies.jsonl?ref={transport.head}",
        f"repos/example/spark/contents/datasets/rounds/r-1.json?ref={transport.base}",
    ]:
        result = transport(
            ["gh", "api", "--hostname", "github.com", "--method", "GET", endpoint],
            env={"GH_TOKEN": "fixture-credential"},
        )
        if variant == "pr-number" and endpoint.endswith("pulls/7"):
            pr = json.loads(result.stdout)
            pr["number"] = 7.0
            result.stdout = json.dumps(pr)
        routes[endpoint] = result.stdout
    (tmp_path / "routes.json").write_text(json.dumps(routes))
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nprint(json.loads(Path(__file__).with_name('routes.json').read_text())[sys.argv[-1]])\n"
    )
    gh.chmod(0o755)
    before = store.path_for("r-1").read_bytes()
    env = {"PATH": str(tmp_path) + ":/usr/bin:/bin", "GH_TOKEN": "fixture-credential", "LANG": "C.UTF-8"}
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    command = [sys.executable, "-m", "validator.pr_admission", *argv]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    (tmp_path / "command.json").write_text(
        json.dumps(
            dict(argv=command, cwd=str(Path.cwd()), exit=result.returncode, stdout=result.stdout, stderr=result.stderr)
        )
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert store.path_for("r-1").read_bytes() == before
    assert store.load("r-1").admissions == {}
    assert len(intake.read_receipts()) == 1


def test_admitted_assignment_cannot_regress_to_committed_state(authority):
    store, _, _, transport, argv = authority
    assert main(argv, transport=transport) == 0
    win = store.load("r-1")
    win.assignment = replace(win.assignment, state=COMMITTED)
    store.save(win)
    with pytest.raises(AdmissionError, match="cannot return to committed"):
        admission_for(store.load("r-1"), "alice")


def test_legacy_valid_seal_without_assignment_binding_has_no_authority(authority):
    # Construct the previous format using its documented digest calculation. It is
    # deliberately refused, never used as an evaluation oracle or a positive control.
    from validator.pr_admission import digest

    store, _, _, transport, argv = authority
    assert main(argv, transport=transport) == 0
    win = store.load("r-1")
    record = win.admissions["alice"]
    del record["assignment_identity"]
    record["round_identity"] = digest(
        {"round_id": win.round_id, "challenge": win.challenge.snapshot(), "origin": win.store_identity}
    )
    record["admission_id"] = digest({k: v for k, v in record.items() if k != "admission_id"})
    store.save(win)
    before = store.path_for("r-1").read_bytes()
    with pytest.raises(AdmissionError, match="trusted recovery"):
        admission_for(store.load("r-1"), "alice")
    assert main(argv, transport=transport) == 2
    assert store.path_for("r-1").read_bytes() == before

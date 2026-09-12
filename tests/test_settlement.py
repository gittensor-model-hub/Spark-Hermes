"""Score-to-crown, durable crash recovery, real CLI and controlled GitHub delivery."""

import copy
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest
from settlement_support import REPOSITORY, ActionTransport, cli, prepare, settle

from validator.crown import CrownError, contenders_from, select
from validator.score import policy_hash
from validator.settlement import GitHubActions, SettlementError, SettlementStore, main
from validator.store import RoundStore


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    prepare(tmp_path)
    return tmp_path


def test_actual_fixture_cli_round_exports_persisted_intent(world):
    cli(
        world,
        "validator.crown",
        "select",
        "--store",
        world / "rounds",
        "--settlement-root",
        world / "settlement",
        "--scorecards",
        world / "cards",
        "--episodes",
        world / "episodes",
        "--out",
        world / "settled.json",
    )
    record = json.loads((world / "settled.json").read_text())
    outcome = record["outcome"]
    assert outcome["winner"]["miner_id"] == "alice"
    assert outcome["winner"]["score"] == 0.3103  # (87000 - 60000) / 87000, flat independent arms
    assert [s["miner_id"] for s in outcome["ranked"]] == ["alice", "carol"]
    refused = record["entries"][1]["scorecard"]
    assert refused["decision"]["accepted"] is False
    assert refused["reduction_interval"] == [0.05, 0.05]
    assert outcome["ineligible"][0]["miner_id"] == "bob"
    assert record["mode"] == "fixture" and not record["authorizes_payment"]
    assert record["policy_hash"] == policy_hash(record["policy"])
    assert RoundStore(world / "rounds").load("r-1").state == "settled"
    intent = cli(
        world, "validator.crown", "actions", "--store", world / "rounds", "--settlement-root", world / "settlement"
    )
    actions = json.loads(intent.stdout)
    assert actions[0]["action"]["kind"] == "label_add"  # Not lost after standing changes!
    dry_run = cli(world, "validator.settlement", "deliver", "--settlement-root", world / "settlement")
    assert all(row["delivered"] is False for row in json.loads(dry_run.stdout))
    state = SettlementStore(world / "settlement")
    assert all(row["status"] == "pending" and row["attempts"] == 0 for row in state.actions())
    assert settle(world) == record
    exported = cli(world, "validator.settlement", "export", "--settlement-root", world / "settlement")
    assert json.loads(exported.stdout) == record
    assert json.loads((world / "github-fixture.json").read_text())["mutations"] == []


def test_stale_files_never_enter_active_field_and_fully_equal_ties_are_stable(world):
    stale = json.loads((world / "cards/r-1-alice.json").read_text())
    stale.update(round_id="historical", miner_id="historical", task_id="other-task")
    (world / "cards/historical.json").write_text(json.dumps(stale))
    field = contenders_from(world / "cards", store=RoundStore(world / "rounds"), round_id="r-1")
    import dataclasses

    tied = [dataclasses.replace(c, received_at=10) for c in field]
    assert select(tied).to_record() == select(list(reversed(tied))).to_record()
    assert settle(world)["outcome"]["winner"]["miner_id"] == "alice"


@pytest.mark.parametrize(
    "mutation", ["task", "epoch", "round", "policy", "policy_hash", "decision", "missing", "tokens"]
)
def test_incomplete_changed_or_forged_active_card_cannot_settle(world, mutation):
    path = world / "cards/r-1-alice.json"
    card = json.loads(path.read_text())
    if mutation == "task":
        card["task_id"] = "other"
    elif mutation == "epoch":
        card["identity"]["epoch"]["epoch_id"] = "stale"
    elif mutation == "round":
        card["round_id"] = "stale"
    elif mutation == "policy":
        card["policy"].pop("bootstrap_seed")
        card["policy_hash"] = policy_hash(card["policy"])
    elif mutation == "policy_hash":
        card["policy_hash"] = "wrong"
    elif mutation == "decision":
        card["decision"]["accepted"] = False
    elif mutation == "tokens":
        card["candidate"]["tokens"][0] = True
    if mutation == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps(card))
    with pytest.raises((CrownError, SettlementError)):
        settle(world)
    state = SettlementStore(world / "settlement")
    assert state.actions() == []
    with pytest.raises(SettlementError, match="no committed"):
        state.record("r-1")


def test_changed_episode_log_aborts_atomic_persistence(world):
    log = world / "episodes/r-1/alice.jsonl"
    data = [json.loads(line) for line in log.read_text().splitlines()]
    data[0]["tokens_used"] += 100
    log.write_text("".join(json.dumps(row) + "\n" for row in data))
    # Same rounded median/decision is insufficient: actual per-attempt evidence binds it.
    with pytest.raises(SettlementError, match="disagrees"):
        settle(world)
    assert SettlementStore(world / "settlement").actions() == []


def test_no_eligible_round_keeps_incumbent_and_old_round_cannot_reactivate(world):
    first = settle(world)
    prepare(world, round_id="r-2", tokens={"alice": 82650})
    second = settle(world)
    assert second["outcome"]["winner"] is None
    assert second["standing"]["winner"] == first["standing"]["winner"]
    assert second["standing"]["barren_rounds"] == 1
    assert all(not a["kind"].startswith("label_") for a in second["actions"])
    with pytest.raises(SettlementError, match="not active"):
        settle(world, round_id="r-1")
    with pytest.raises(SettlementError, match="historical"):
        SettlementStore(world / "settlement").activate(RoundStore(world / "rounds"), "r-1", REPOSITORY)


def test_empty_round_is_a_valid_no_winner_outcome(tmp_path):
    from competition_support import window

    store = RoundStore(tmp_path / "rounds", mode="fixture", namespace="empty")
    win = window(store, spread=False)
    win.freeze(now=30, reason="CPU fixture")
    win.grade(now=31)
    store.save(win)
    state = SettlementStore(tmp_path / "settlement", mode="fixture", namespace="empty")
    state.activate(store, "r-1", REPOSITORY)
    record = settle(tmp_path)
    assert record["outcome"]["winner"] is None and record["entries"] == [] and record["actions"] == []


def child_settle(root, stage=None):
    def crash(name):
        if name == stage:
            os._exit(73)

    settle(Path(root), hook=crash)


def child_deliver(root, stage):
    root = Path(root)
    count = 0

    def crash(name):
        nonlocal count
        if name == stage:
            count += 1
            if count == 2:  # Crash on the non-idempotent review, after the initial label.
                os._exit(73)

    SettlementStore(root / "settlement").deliver(
        GitHubActions(REPOSITORY, transport=ActionTransport(root / "github-fixture.json")),
        hook=crash,
    )


def run_child(target, *args):
    proc = multiprocessing.get_context("fork").Process(target=target, args=args)
    proc.start()
    proc.join(30)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        pytest.fail("child hung")
    return proc.exitcode


@pytest.mark.parametrize("stage", ["before_persist", "before_commit", "after_persist"])
def test_abrupt_process_death_around_atomic_commit(world, stage):
    assert run_child(child_settle, world, stage) == 73
    state = SettlementStore(world / "settlement")
    if stage == "after_persist":
        assert state.record("r-1")["outcome"]["winner"]["miner_id"] == "alice"
        assert len(state.actions()) == 6
    else:
        assert state.actions() == []
        with pytest.raises(SettlementError):
            state.record("r-1")
    record = settle(world)
    assert len(state.actions()) == 6
    assert len({a["key"] for a in state.actions()}) == 6
    assert settle(world) == record
    assert RoundStore(world / "rounds").load("r-1").state == "settled"


def test_concurrent_processes_commit_one_winner_and_action_set(world):
    ctx = multiprocessing.get_context("fork")
    children = [ctx.Process(target=child_settle, args=(world,)) for _ in range(4)]
    for child in children:
        child.start()
    for child in children:
        child.join(30)
        if child.is_alive():
            child.terminate()
            child.join()
        assert child.exitcode == 0
    state = SettlementStore(world / "settlement")
    with sqlite3.connect(state.path) as db:
        assert db.execute("SELECT count(*) FROM rounds").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 6
    assert state.record("r-1")["outcome"]["winner"]["miner_id"] == "alice"


@pytest.mark.parametrize("stage", ["before_deliver", "after_deliver", "before_ack", "after_ack"])
def test_restart_reconciles_review_and_does_not_reissue_acknowledged_actions(world, stage):
    settle(world)
    assert run_child(child_deliver, world, stage) == 73
    state = SettlementStore(world / "settlement")
    acknowledged = {a["key"]: a["attempts"] for a in state.actions() if a["status"] == "acknowledged"}
    assert (
        main(
            ["deliver", "--settlement-root", str(world / "settlement"), "--allow-external"],
            transport=ActionTransport(world / "github-fixture.json"),
        )
        == 0
    )
    after = json.loads((world / "github-fixture.json").read_text())
    assert len(after["mutations"]) == 6
    assert all(len(pr["reviews"]) == 1 for pr in after["prs"].values())
    assert all(a["status"] == "acknowledged" for a in state.actions())
    assert {a["key"]: a["attempts"] for a in state.actions() if a["key"] in acknowledged} == acknowledged
    assert (
        main(
            ["deliver", "--settlement-root", str(world / "settlement"), "--allow-external"],
            transport=ActionTransport(world / "github-fixture.json"),
        )
        == 0
    )
    assert json.loads((world / "github-fixture.json").read_text()) == after


def test_failed_and_ambiguous_delivery_stays_pending_until_reconciled(world):
    settle(world)
    state = SettlementStore(world / "settlement")
    with pytest.raises(SettlementError):
        state.deliver(GitHubActions(REPOSITORY, transport=ActionTransport(world / "github-fixture.json", fail=True)))
    assert all(a["status"] == "pending" for a in state.actions())
    with pytest.raises(SettlementError):
        state.deliver(
            GitHubActions(REPOSITORY, transport=ActionTransport(world / "github-fixture.json", lost_response=True))
        )
    assert all(a["status"] == "pending" for a in state.actions())
    state.deliver(GitHubActions(REPOSITORY, transport=ActionTransport(world / "github-fixture.json")))
    assert len(json.loads((world / "github-fixture.json").read_text())["mutations"]) == 6


def test_fixture_cannot_be_delivered_live_or_rebound_into_production(world, monkeypatch):
    settle(world)
    monkeypatch.delenv("GH_TOKEN")
    assert main(["deliver", "--settlement-root", str(world / "settlement"), "--allow-external"]) == 2
    state = SettlementStore(world / "production")
    with pytest.raises(SettlementError, match="trust domains"):
        state.activate(RoundStore(world / "rounds"), "r-1", REPOSITORY)
    with pytest.raises(ValueError, match="immutable"):
        SettlementStore(world / "settlement", mode="production")


@pytest.mark.parametrize("mutation", ["head", "repo", "author", "merged", "malformed"])
def test_changed_remote_identity_cannot_receive_actions(world, mutation):
    settle(world)
    path = world / "github-fixture.json"
    data = json.loads(path.read_text())
    pr = data["prs"]["7"]
    if mutation == "head":
        pr["head"]["sha"] = "3" * 40
    elif mutation == "repo":
        pr["base"]["repo"]["full_name"] = "wrong/repo"
    elif mutation == "author":
        pr["user"]["login"] = "wrong"
    elif mutation == "merged":
        pr["merged"] = True
    else:
        pr.pop("head")
    path.write_text(json.dumps(data))
    state = SettlementStore(world / "settlement")
    with pytest.raises(SettlementError):
        state.deliver(GitHubActions(REPOSITORY, transport=ActionTransport(path)))
    assert json.loads(path.read_text())["mutations"] == []
    assert all(a["status"] == "pending" for a in state.actions())


def test_review_reconciliation_requires_author_commit_and_full_body(world):
    settle(world)
    state = SettlementStore(world / "settlement")
    review = next(a["action"] for a in state.actions() if a["action"]["kind"] == "review")
    path = world / "github-fixture.json"
    data = json.loads(path.read_text())
    forged = {
        "id": 999,
        "body": review["body"] + "\n\n" + GitHubActions.marker(review),
        "user": {"login": "attacker"},
        "commit_id": review["head_sha"],
        "state": "COMMENTED",
    }
    data["prs"]["7"]["reviews"] = [copy.deepcopy(forged)]
    path.write_text(json.dumps(data))
    adapter = GitHubActions(REPOSITORY, transport=ActionTransport(path))
    assert adapter.reconcile(review) is None
    adapter.apply(review)
    assert adapter.reconcile(review)["evidence"]["review_id"] == 2


def test_production_judge_refuses_fixture_replay_before_execution(tmp_path):
    from competition_support import window

    store = RoundStore(tmp_path / "rounds")
    window(store)
    before = store.path_for("r-1").read_bytes()
    result = cli(
        tmp_path,
        "validator.judge",
        "judge",
        "--round",
        "r-1",
        "--store",
        store.root,
        "--fixture-episodes",
        tmp_path / "untrusted",
        expected=2,
    )
    assert "fixture episode replay requires a fixture state root" in result.stderr
    assert store.path_for("r-1").read_bytes() == before


def test_readonly_production_metadata_and_action_adapter_need_credentials(monkeypatch):
    from validator.pr_admission import AdmissionError

    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(AdmissionError, match="missing GitHub credential"):
        GitHubActions(REPOSITORY, transport=lambda *args, **kwargs: pytest.fail("must fail before transport"))


def test_concurrent_deliverers_share_order_and_never_duplicate_reviews(world):
    settle(world)
    ctx = multiprocessing.get_context("fork")
    children = [ctx.Process(target=child_deliver, args=(world, "never")) for _ in range(3)]
    for child in children:
        child.start()
    for child in children:
        child.join(30)
        if child.is_alive():
            child.terminate()
            child.join()
        assert child.exitcode == 0
    state = SettlementStore(world / "settlement")
    assert all(a["status"] == "acknowledged" and a["attempts"] == 1 for a in state.actions())
    remote = json.loads((world / "github-fixture.json").read_text())
    assert len(remote["mutations"]) == 6
    assert all(len(pr["reviews"]) == 1 for pr in remote["prs"].values())


@pytest.mark.parametrize("already_absent", [False, True])
def test_merged_historical_crown_cleanup_unblocks_later_actions(tmp_path, already_absent):
    prepare(tmp_path, tokens={"alice": 60000})
    settle(tmp_path)
    state = SettlementStore(tmp_path / "settlement")
    remote_path = tmp_path / "github-fixture.json"
    adapter = GitHubActions(REPOSITORY, transport=ActionTransport(remote_path))
    state.deliver(adapter)
    remote = json.loads(remote_path.read_text())
    remote["prs"]["7"].update(merged=True, state="closed")
    if already_absent:
        remote["prs"]["7"]["labels"] = []
    remote_path.write_text(json.dumps(remote))
    prepare(tmp_path, round_id="r-2", tokens={"bob": 60000}, pr_numbers={"bob": 8})
    second = settle(tmp_path)
    assert [a["kind"] for a in second["actions"]] == ["label_remove", "label_add", "review"]
    assert all(not row["delivered"] for row in state.deliver())
    assert all(a["status"] == "pending" for a in state.actions() if a["round_id"] == "r-2")
    state.deliver(adapter)
    assert all(a["status"] == "acknowledged" for a in state.actions())
    remote = json.loads(remote_path.read_text())
    assert remote["prs"]["7"]["labels"] == []
    assert remote["prs"]["8"]["labels"] and len(remote["prs"]["8"]["reviews"]) == 1
    assert sum(m["method"] == "DELETE" for m in remote["mutations"]) == (0 if already_absent else 1)
    state.deliver(adapter)
    assert json.loads(remote_path.read_text()) == remote


@pytest.mark.parametrize("mutation", ["head", "author", "repo", "number", "merged-type"])
def test_historical_cleanup_retains_identity_checks(world, mutation):
    settle(world)
    action = {**SettlementStore(world / "settlement").actions()[0]["action"], "kind": "label_remove"}
    path = world / "github-fixture.json"
    remote = json.loads(path.read_text())
    pr = remote["prs"]["7"]
    pr.update(merged=True, state="closed")
    if mutation == "head":
        pr["head"]["sha"] = "9" * 40
    elif mutation == "author":
        pr["user"]["login"] = "stranger"
    elif mutation == "repo":
        pr["base"]["repo"]["full_name"] = "stranger/repo"
    elif mutation == "number":
        pr["number"] = 99
    else:
        pr["merged"] = "true"
    path.write_text(json.dumps(remote))
    adapter = GitHubActions(REPOSITORY, transport=ActionTransport(path))
    with pytest.raises(SettlementError, match="identity/state"):
        adapter.reconcile(action)
    with pytest.raises(SettlementError, match="identity/state"):
        adapter.apply(action)
    assert json.loads(path.read_text())["mutations"] == []

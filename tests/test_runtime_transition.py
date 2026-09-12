"""Original-consumer runtime cutover, immutable ancestry and crash/race controls."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import venv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from admin.artifacts import StageError, read_record
from admin.candidates import CandidateStore
from admin.cycles import CycleController
from admin.parents import ParentAuthority
from admin.release import ReleaseAuthority
from admin.runtime_transition import commit, current_history, propose


@pytest.fixture(scope="module")
def originals(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-originals")
    runtime = root / "retained-installation"
    venv.EnvBuilder(with_pip=False).create(runtime)
    site = next((runtime / "lib").glob("python*/site-packages"))
    sites = [p for p in sys.path if p.endswith("site-packages")]
    (site / "cpu-dependencies.pth").write_text("\n".join(sites) + "\n")
    for name in ("admin", "eval", "hermes", "hermesbench", "miner", "proof", "teacher", "validator"):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin
        shutil.copytree(Path(spec.origin).parent, site / name, ignore=shutil.ignore_patterns("__pycache__"))
    with (site / "admin/__init__.py").open("a") as stream:
        stream.write("\n# Fixed retained supported fixture runtime, before any evidence.\n")
    python = runtime / "bin/python"
    history = root / "history"
    driver = Path(__file__).with_name("runtime_transition_fixture.py")
    argv = [str(python), "-I", str(driver), "source", "--root", str(history)]
    with (root / "setup.log").open("w") as log:
        result = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    assert result.returncode == 0, (root / "setup.log").read_text()[-4000:]
    summary = read_record(history / "transition-source.json")
    assert summary["second"]["status"] == "complete"
    prepared = read_record(Path(summary["second"]["workspace"]) / "models/sft/prepared.json")
    assert prepared["parent"]["decision"]["strict"] is True
    assert summary["interrupted"]["next_stage"] == "evaluate"
    databases = {}
    for index, path in enumerate(history.rglob("*.sqlite3")):
        backup = root / f"original-database-{index}.sqlite3"
        with sqlite3.connect(path) as source, sqlite3.connect(backup) as target:
            source.backup(target)
        databases[path] = backup
    return {
        "root": root,
        "history": history,
        "python": python,
        "summary": summary,
        "databases": databases,
        "site": site,
    }


@pytest.fixture
def campaign(originals, tmp_path):
    # Reuse immutable upstream producer evidence. Each mutation/race restores only
    # its isolated fixture databases to those originals; no approval is reissued.
    for path, backup in originals["databases"].items():
        with sqlite3.connect(backup) as source, sqlite3.connect(path) as target:
            source.backup(target)
    source = ReleaseAuthority(originals["history"] / "releases")
    target = ReleaseAuthority(tmp_path / "target", mode="fixture", namespace=source.identity["namespace"])
    yield originals, source, target
    for path, backup in originals["databases"].items():
        with sqlite3.connect(backup) as source, sqlite3.connect(path) as target:
            source.backup(target)


def proposal(campaign):
    data, source, target = campaign
    record = propose(source, data["summary"]["retention"]["id"], target)
    return record["id"]


def original_command(data, *args):
    argv = [str(data["python"]), "-I", "-m", "admin.cli", *map(str, args)]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=data["root"], capture_output=True, text=True, timeout=600)
    with (data["root"] / "original-commands.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "argv": argv,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "seconds": time.monotonic() - started,
                }
            )
            + "\n"
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_ordinary_refusal_and_baseline_parent_authority(campaign):
    data, source, target = campaign
    prior = source.status()
    with pytest.raises(StageError, match="evaluator/environment changed"):
        source.active_pair()
    identifier = proposal(campaign)
    imported = commit(target, identifier)
    pair = target.active_pair()
    original = data["summary"]["second"]["incumbent"]
    assert prior["candidate"] == original["candidate"]
    assert pair["authority_kind"] == "verified-historical-baseline"
    assert pair["runtime"] != data["summary"]["retention"]["payload"]["runtime"]
    expected = source.candidates().store.get(prior["candidate"], kind="candidate")["payload"]
    assert pair["model"] == expected["model"] and pair["agent_id"] == expected["agent_id"]
    parent, authority = ParentAuthority(target.store.root).approved_parent(
        imported["approval"],
        identity=target.identity,
        profile=expected["model"]["profile"],
        repository=expected["model"]["base_model"],
        revision=expected["model"]["revision"],
    )
    assert parent == Path(expected["model"]["merged"])
    assert authority["decision"]["result"] == "baseline-import"
    assert not authority["decision"]["measured_gain"] and not authority["decision"]["authorizes_production_promotion"]
    with pytest.raises(StageError):
        target.activate(imported["approval"])
    assert imported["generation"] == 0 and len(imported["history"]) == 1
    assert source.status() == prior
    original_command(
        data, "runtime", "inspect", "--root", source.store.root, "--id", data["summary"]["retention"]["id"]
    )
    with source.store.connect() as db, target.store.connect() as other:
        assert list(db.execute("SELECT * FROM confirmation_use ORDER BY family")) == list(
            other.execute("SELECT * FROM confirmation_use ORDER BY family")
        )


@pytest.mark.parametrize("boundary", ["verified", "source_committed", "before_target_commit", "published"])
def test_crash_resume_is_single_atomic_baseline(campaign, boundary):
    _, source, target = campaign
    identifier = proposal(campaign)

    def crash(stage):
        if stage == boundary:
            raise RuntimeError("injected publication crash")

    with pytest.raises(RuntimeError, match="injected"):
        commit(target, identifier, hook=crash)
    with target.store.connect() as db:
        count = db.execute("SELECT count(*) FROM incumbent").fetchone()[0]
    assert count == (1 if boundary == "published" else 0)
    result = commit(target, identifier)
    assert result == commit(target, identifier)
    assert len(result["history"]) == 1 and result["generation"] == 0
    with pytest.raises(StageError, match="retired"):
        source.activate(source.status()["approval"])


def test_duplicate_and_competing_transition_races(campaign, tmp_path):
    data, source, target = campaign
    first = proposal(campaign)
    other = ReleaseAuthority(tmp_path / "competitor", mode="fixture", namespace=source.identity["namespace"])
    second = propose(source, data["summary"]["retention"]["id"], other)["id"]

    def attempt(pair):
        try:
            return commit(*pair)
        except StageError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [(target, first), (other, second)]))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert any(isinstance(r, str) and "single successor" in r for r in results)
    winner, identifier = (target, first) if isinstance(results[0], dict) else (other, second)
    with ThreadPoolExecutor(max_workers=2) as pool:
        duplicate = list(pool.map(attempt, [(winner, identifier), (winner, identifier)]))
    assert duplicate[0] == duplicate[1] and len(duplicate[0]["history"]) == 1


def test_source_epoch_advancement_invalidates_verified_proposal(campaign):
    data, source, target = campaign
    identifier = proposal(campaign)
    previous = source.status()
    first = data["summary"]["first"]["incumbent"]["approval"]
    advanced = original_command(data, "release", "rollback", "--root", source.store.root, "--id", first)
    assert advanced["generation"] == previous["generation"] + 1
    assert advanced["candidate"] != previous["candidate"]
    with pytest.raises(StageError, match="snapshot advanced"):
        commit(target, identifier)
    assert not target.store.records(kind="runtime-parent")


def test_source_new_confirmation_reservation_invalidates_proposal(campaign):
    data, source, target = campaign
    identifier = proposal(campaign)
    with source.store.connect() as db:
        before = list(db.execute("SELECT * FROM confirmation_use ORDER BY family"))
    driver = Path(__file__).with_name("runtime_transition_fixture.py")
    with (data["root"] / "reserve.log").open("w") as log:
        result = subprocess.run(
            [str(data["python"]), "-I", str(driver), "reserve", "--root", str(data["history"])],
            cwd=data["root"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    assert result.returncode == 0, (data["root"] / "reserve.log").read_text()[-2000:]
    with source.store.connect() as db:
        after = list(db.execute("SELECT * FROM confirmation_use ORDER BY family"))
    assert len(after) == len(before) + 6
    with pytest.raises(StageError, match="snapshot advanced"):
        commit(target, identifier)
    assert not target.store.records(kind="runtime-parent")


@pytest.mark.parametrize("change", ["model", "agent", "prepared", "recipe", "data", "completion", "runtime", "catalog"])
def test_original_artifact_mutations_refuse_before_and_after_cutover(campaign, change):
    data, source, target = campaign
    identifier = proposal(campaign)
    commit(target, identifier)
    candidate = source.candidates().store.get(source.status()["candidate"], kind="candidate")["payload"]
    prepared = read_record(Path(candidate["prepared"]["path"]))
    if change == "model":
        path = Path(candidate["model"]["merged"]) / "model.safetensors"
    elif change in {"agent", "prepared", "recipe"}:
        path = Path(candidate[change]["path"])
    elif change == "data":
        path = Path(prepared["data"])
    elif change == "catalog":
        path = Path(source.configuration()["data_policy"]["path"])
    elif change == "runtime":
        path = data["site"] / "admin/__init__.py"
    else:
        decision = source.store.get(source.status()["approval"], kind="release-decision")["payload"]
        evaluation = source.experiments.get(decision["evaluation"]["id"], kind="crossed-evaluation")["payload"]
        path = Path(evaluation["logs"]["Q11"]["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"changed original\n")
        with pytest.raises((StageError, OSError, ValueError)):
            target.active_pair()
        with pytest.raises((StageError, OSError, ValueError)):
            commit(target, identifier)
    finally:
        path.write_bytes(original)
    assert target.active_pair()["authority_kind"] == "verified-historical-baseline"


def test_missing_original_reservation_and_issuer_refuse(campaign):
    _, source, target = campaign
    identifier = proposal(campaign)
    with source.store.connect() as db:
        db.execute("DELETE FROM confirmation_use WHERE family=(SELECT family FROM confirmation_use LIMIT 1)")
    with pytest.raises(StageError, match="reservation"):
        commit(target, identifier)


def test_target_substitution_and_untrusted_json_refuse(campaign):
    _, _, target = campaign
    identifier = proposal(campaign)
    with target.store.connect() as db:
        db.execute("INSERT INTO experiment_runs VALUES ('unexpected','running')")
    with pytest.raises(StageError, match="target authority changed"):
        commit(target, identifier)
    with pytest.raises(StageError, match="no committed authority"):
        commit(target, "sha256:" + "0" * 64)


def test_all_source_authority_writers_and_competition_refuse_cutover(campaign):
    _, source, target = campaign
    identifier = proposal(campaign)
    commit(target, identifier)
    from admin.competition_pair import active_pair
    from admin.evaluation import execute_crossed

    calls = [
        lambda: source.configure(),
        lambda: source.freeze(),
        lambda: source.decide("absent"),
        lambda: source.activate(source.status()["approval"]),
        lambda: CycleController(source.store.root).start("new", {}),
        lambda: CycleController(source.store.root).resume("absent"),
        lambda: source.store.put("release-decision", {}),
        lambda: source.experiments.put("crossed-plan", {}),
        lambda: source.candidates().register(),
        lambda: execute_crossed(source, "absent"),
        lambda: active_pair(source.store.root),
    ]
    for call in calls:
        with pytest.raises(StageError, match="retired"):
            call()


def test_domain_parent_profile_and_old_evaluation_refuse(campaign, tmp_path):
    data, source, target = campaign
    foreign = ReleaseAuthority(tmp_path / "production", mode="production", namespace=source.identity["namespace"])
    with pytest.raises(StageError, match="trust mode/namespace"):
        propose(source, data["summary"]["retention"]["id"], foreign)
    identifier = proposal(campaign)
    imported = commit(target, identifier)
    with pytest.raises(StageError, match="another model profile"):
        ParentAuthority(target.store.root).approved_parent(
            imported["approval"], identity=target.identity, profile="bf16", repository="wrong", revision="wrong"
        )
    old = source.store.get(source.status()["approval"], kind="release-decision")["payload"]
    with pytest.raises(StageError, match="no committed authority"):
        target.evaluation(old["evaluation"]["id"])


def test_current_readonly_eligibility_does_not_issue_bridge(campaign):
    data, source, target = campaign
    before = source.status()
    history = current_history(source, data["summary"]["retention"]["payload"]["runtime"])
    assert history["state"] == before
    assert not target.store.records(kind="runtime-parent")
    with pytest.raises(StageError, match="evaluator/environment changed"):
        CandidateStore(source.candidates().store.root).resolve(before["candidate"])


def test_target_issuer_substitution_after_verification_refuses(campaign):
    _, source, target = campaign
    identifier = proposal(campaign)
    path = target.store.root / ".identity"
    original = path.read_bytes()

    def substitute(stage):
        if stage == "verified":
            changed = json.loads(original)
            changed["issuer"] = "substituted-target"
            path.write_text(json.dumps(changed))

    try:
        with pytest.raises(StageError, match="target.*issuer|target.*identity"):
            commit(target, identifier, hook=substitute)
    finally:
        path.write_bytes(original)
    with source.store.connect() as db:
        assert db.execute("SELECT count(*) FROM runtime_cutover").fetchone()[0] == 0


def test_retained_runtime_cannot_reenroll_old_workspace_or_approval(campaign, tmp_path):
    data, source, target = campaign
    identifier = proposal(campaign)
    original = source.candidates().store.get(source.status()["candidate"], kind="candidate")["payload"]
    commit(target, identifier)
    paths = {
        "workspace": original["workspace"],
        "merged-record": original["model"]["record"],
        "agent": original["agent"]["path"],
        "workload": original["workload"]["path"],
        "parent": original["parent"]["path"],
    }
    commands = [
        [
            "candidates",
            "register",
            "--root",
            str(tmp_path / "new-candidates"),
            "--mode",
            "fixture",
            "--namespace",
            source.identity["namespace"],
            *[v for k, p in paths.items() for v in ("--" + k, p)],
        ],
        [
            "prepare",
            "--root",
            str(tmp_path / "new-workspace"),
            "--release-root",
            str(source.store.root),
            "--parent-approval",
            source.status()["approval"],
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [str(data["python"]), "-I", "-m", "admin.cli", *command],
            cwd=data["root"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 2 and "retired" in result.stderr, result.stderr
    # Reconfiguration of the original candidate store must refuse too, before an
    # empty authority could reinterpret its confirmations as an unused pool.
    new = ReleaseAuthority(tmp_path / "new-release", mode="fixture", namespace=source.identity["namespace"])
    with pytest.raises(StageError, match="retired"):
        new.configure(
            candidates=source.candidates().store.root,
            incumbent=source.status()["candidate"],
            data_policy=Path(source.configuration()["data_policy"]["path"]),
            policy=source.configuration()["policy"],
        )


def test_missing_workspace_ownership_refuses_source_snapshot(campaign):
    _, source, target = campaign
    identifier = proposal(campaign)
    candidate = source.candidates().store.get(source.status()["candidate"], kind="candidate")["payload"]
    with sqlite3.connect(Path(candidate["workspace"]) / "preparation-authority/authority.sqlite3") as db:
        db.execute("DELETE FROM metadata WHERE key LIKE 'campaign:%'")
    with pytest.raises(StageError, match="workspace/campaign ownership"):
        commit(target, identifier)


def test_removing_whole_interrupted_reservation_refuses_fresh_verification(campaign):
    data, source, _ = campaign
    interrupted = next(
        j["output"]["value"]["id"] for j in data["summary"]["interrupted"]["jobs"] if j["stage"] == "plan"
    )
    with source.store.connect() as db:
        db.execute("DELETE FROM confirmation_use WHERE experiment=?", (interrupted,))
    with pytest.raises(StageError, match="reservation"):
        proposal(campaign)


def test_unsupported_target_journal_refuses_before_retirement(campaign):
    _, source, target = campaign
    identifier = proposal(campaign)
    with target.store.connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
    with pytest.raises(StageError, match="DELETE journals"):
        commit(target, identifier)
    with source.store.connect() as db:
        assert not db.execute("SELECT * FROM runtime_cutover").fetchone()


def test_actual_late_supervisor_requires_quiescence_and_fresh_snapshot(campaign):
    data, source, target = campaign
    old = proposal(campaign)
    driver = Path(__file__).with_name("runtime_transition_fixture.py")
    argv = [str(data["python"]), "-I", str(driver), "quiescence", "--root", str(data["history"])]
    with (data["root"] / "quiescence.log").open("w") as log:
        submitted = subprocess.run(argv, cwd=data["root"], stdout=log, stderr=subprocess.STDOUT, timeout=600)
    assert submitted.returncode == 0, (data["root"] / "quiescence.log").read_text()[-2000:]
    job = read_record(data["history"] / "quiescence-submission.json")
    with pytest.raises(StageError, match="outstanding supervised job"):
        commit(target, old)
    ready, finish = data["root"] / "supervisor-ready", data["root"] / "supervisor-finish"
    # Pause the explicit external CPU training substitute, preserving the actual
    # installed supervisor's claim, recipe validation and completion transactions.
    code = """
import sys, time
from pathlib import Path
from admin.cycle_jobs import FixtureTraining, run_job
original = FixtureTraining.__call__
def paused(self, *args, **kwargs):
    Path(sys.argv[3]).touch()
    deadline = time.monotonic() + 180
    while not Path(sys.argv[4]).exists():
        if time.monotonic() > deadline: raise RuntimeError('test release timeout')
        time.sleep(.02)
    return original(self, *args, **kwargs)
FixtureTraining.__call__ = paused
run_job(Path(sys.argv[1]), sys.argv[2])
"""
    with (data["root"] / "supervisor.log").open("w") as log:
        process = subprocess.Popen(
            [str(data["python"]), "-I", "-c", code, str(source.store.root), job["job"], str(ready), str(finish)],
            cwd=data["root"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            assert ready.exists(), (data["root"] / "supervisor.log").read_text()
            with source.store.connect() as db:
                assert (
                    db.execute("SELECT status FROM cycle_external_jobs WHERE id=?", (job["job"],)).fetchone()[0]
                    == "running"
                )
            with pytest.raises(StageError, match="outstanding supervised job"):
                commit(target, old)
            with source.store.connect() as db:
                assert not db.execute("SELECT * FROM runtime_cutover").fetchone()
            finish.touch()
            assert process.wait(timeout=120) == 0
        finally:
            finish.touch()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
    with pytest.raises(StageError, match="snapshot advanced"):
        commit(target, old)
    fresh = proposal(campaign)
    imported = commit(target, fresh)
    assert imported["generation"] == 0 and len(imported["history"]) == 1
    # A late duplicate supervisor observes the original completion; it cannot
    # change any source row or launch a second external execution after cutover.
    from admin.runtime_transition import _table_snapshot

    before = _table_snapshot(source.store)
    late = subprocess.run(
        [str(data["python"]), "-I", "-m", "admin.cycle_jobs", "--root", str(source.store.root), "--id", job["job"]],
        cwd=data["root"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert late.returncode == 0, late.stderr
    assert _table_snapshot(source.store) == before
    assert target.active_pair()["authority_kind"] == "verified-historical-baseline"


@pytest.mark.parametrize("missing", ["interpreter", "checkpoint", "approval"])
def test_missing_original_authority_or_artifacts_cannot_create_bridge(campaign, missing):
    data, source, target = campaign
    if missing == "approval":
        with source.store.connect() as db:
            db.execute("DELETE FROM records WHERE id=?", (source.status()["approval"],))
        with pytest.raises(StageError, match="no committed authority|verification refused"):
            proposal(campaign)
    else:
        if missing == "interpreter":
            path = data["python"]
        else:
            candidate = source.candidates().store.get(source.status()["candidate"], kind="candidate")["payload"]
            path = Path(candidate["model"]["merged"]) / "model.safetensors"
        saved = path.with_name(path.name + ".original-held-by-test")
        path.rename(saved)
        try:
            with pytest.raises((StageError, OSError)):
                proposal(campaign)
        finally:
            saved.rename(path)
    assert not target.store.records(kind="runtime-parent")
    with source.store.connect() as db:
        assert not db.execute("SELECT * FROM runtime_cutover").fetchone()


def test_actual_initial_pair_without_strict_approval_cannot_be_imported(campaign):
    _, source, target = campaign
    # Restore only the genuine configured initial pointer in an isolated mutation
    # case. Its candidate/bootstrap authority is actual producer evidence; no new
    # approval is fabricated or issued to make this negative case pass.
    initial = source.configuration()["initial_incumbent"]
    with source.store.connect() as db:
        db.execute("UPDATE incumbent SET candidate=?,approval=NULL,generation=0", (initial,))
    with pytest.raises(StageError, match="originally activated strict approval"):
        proposal(campaign)
    assert not target.store.records(kind="runtime-parent")

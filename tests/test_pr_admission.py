"""Read-only authenticated ingress, exercised without live GitHub or inference."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from competition_support import GitHubTransport, admit_receipt, rows, window

from eval.strategy_track import Commitment
from validator.intake import Intake, IntakeError
from validator.judge import judge_round
from validator.pr_admission import AdmissionError, GitHubSource, admit, main
from validator.store import RoundStore


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture-credential")
    store = RoundStore(tmp_path / "rounds", require_private=False, mode="fixture", namespace="cpu-ingress")
    win = window(store)
    intake = Intake(
        tmp_path / "bundles", tmp_path / "receipts.jsonl", mode="fixture", namespace=store.identity["namespace"]
    )
    receipt = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "A"}, now=10)
    record = Commitment(
        round_id="r-1", miner_id="alice", bundle_sha256=receipt.bundle_sha256, task_ids=(win.task_id,)
    ).to_record()
    transport = GitHubTransport(win.assignment.to_record(reveal_seed=True), record)
    return store, intake, receipt, transport


def test_admission_cli_is_persistent_and_idempotent(world, capsys):
    store, intake, receipt, transport = world
    argv = [
        "--repository",
        transport.repository,
        "--pr",
        "7",
        "--round",
        "r-1",
        "--store",
        str(store.root),
        "--intake-root",
        str(intake.root),
        "--receipts",
        str(intake.receipts),
    ]
    assert main(argv, transport=transport) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(argv, transport=transport) == 0
    assert json.loads(capsys.readouterr().out) == first
    restored = RoundStore(store.root).load("r-1")
    assert restored.admissions["alice"]["submission_id"] == receipt.submission_id
    assert restored.admissions["alice"]["origin"]["mode"] == "fixture"
    assert len(restored.submissions) == 1
    assert all("pulls/7" in p or "contents/" in p or p == "user" for p in transport.calls)


@pytest.mark.parametrize(
    "mutation", ["author", "round", "digest", "unknown", "scope", "prior", "head", "state", "json", "repo"]
)
def test_untrusted_or_mismatched_admission_never_changes_state(world, mutation):
    store, intake, _, transport = world
    before = store.path_for("r-1").read_bytes()
    record = json.loads(transport.head_text)
    if mutation == "author":
        transport.author = "bob"
    elif mutation == "round":
        record["round_id"] = "r-other"
    elif mutation == "digest":
        record["bundle_sha256"] = "sha256:" + "0" * 64
    elif mutation == "unknown":
        transport.author = record["miner_id"] = "unknown"
    elif mutation == "scope":
        transport.extra = "validator/score.py"
    elif mutation == "prior":
        transport.base_text = '{"round_id":"old"}'
    transport.head_text = json.dumps(record)
    original = transport
    count = 0

    def mutate(argv, **kwargs):
        nonlocal count
        result = original(argv, **kwargs)
        value = json.loads(result.stdout)
        if argv[-1].endswith("pulls/7"):
            count += 1
            if mutation == "head" and count == 2:
                value["head"]["sha"] = "3" * 40
            if mutation == "state" and count == 2:
                value["state"] = "closed"
            if mutation == "repo":
                value["base"]["repo"]["full_name"] = "wrong/repo"
        result.stdout = "[invalid" if mutation == "json" else json.dumps(value)
        return result

    with pytest.raises(ValueError):
        metadata = GitHubSource(transport.repository, transport=mutate).collect(7, round_id="r-1")
        admit(metadata=metadata, round_id="r-1", store=store, intake=intake, now=20)
    assert store.path_for("r-1").read_bytes() == before


def test_missing_credentials_and_caller_json_fail_closed(world, monkeypatch):
    store, intake, _, transport = world
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(AdmissionError, match="missing GitHub credential"):
        GitHubSource(transport.repository, transport=transport)
    with pytest.raises(AdmissionError, match="caller JSON"):
        admit(metadata={"trusted": True, "author": "alice"}, round_id="r-1", store=store, intake=intake)


def test_two_uploads_commit_b_runs_exact_bytes(world, monkeypatch, tmp_path):
    store, intake, _, _ = world
    receipt_b = intake.accept(round_id="r-1", miner_id="alice", files={"SOUL.md": "B"}, now=11)
    admit_receipt(store, intake, receipt_b, monkeypatch)
    win = store.load("r-1")
    win.freeze(now=30, reason="fixture")
    store.save(win)
    consumed = []

    def run(miner, bundle_path, workspace):
        consumed.append((bundle_path, (bundle_path / "SOUL.md").read_bytes()))
        log = workspace / "episodes.jsonl"
        evidence = rows(origin=store.identity, digest=receipt_b.bundle_sha256)
        log.write_text("".join(json.dumps(r) + "\n" for r in evidence))
        return log

    result = judge_round(
        round_id="r-1",
        run=run,
        model_revision="a" * 40,
        store=store,
        intake=intake,
        workspace=tmp_path / "run",
        scorecard_dir=tmp_path / "cards",
    )
    assert result[0].ok, result[0].problem
    assert consumed == [(intake.bundle_dir(receipt_b).resolve(), b"B")]
    card = json.loads((tmp_path / "cards/r-1-alice.json").read_text())
    assert card["decision"]["accepted"] is True
    assert card["identity"]["origin"]["mode"] == "fixture"


@pytest.mark.parametrize("attack", ["bytes", "miner", "symlink"])
def test_tamper_never_invokes_runner(world, monkeypatch, tmp_path, attack):
    store, intake, receipt, _ = world
    admit_receipt(store, intake, receipt, monkeypatch)
    win = store.load("r-1")
    win.freeze(now=30, reason="fixture")
    store.save(win)
    path = intake.bundle_dir(receipt) / "SOUL.md"
    if attack == "bytes":
        path.write_text("changed")
    elif attack == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "other")
    else:
        other = intake.accept(round_id="r-1", miner_id="bob", files={"SOUL.md": "A"})
        intake._write_receipts([other])
    called = []
    result = judge_round(
        round_id="r-1",
        run=lambda *args: called.append(args),
        model_revision="a" * 40,
        store=store,
        intake=intake,
        workspace=tmp_path / "run",
        scorecard_dir=tmp_path / "cards",
    )
    assert not called and not result[0].ok
    assert not list((tmp_path / "cards").glob("*.json"))


def test_concurrent_distinct_and_duplicate_receipts_survive_restart(tmp_path):
    def upload(i):
        intake = Intake(tmp_path / "bundles", tmp_path / "receipts.jsonl")
        return intake.accept(round_id="r", miner_id=f"miner-{i % 12}", files={"SOUL.md": f"bundle {i % 12}"})

    with ThreadPoolExecutor(max_workers=12) as pool:
        receipts = list(pool.map(upload, range(72)))
    intake = Intake(tmp_path / "bundles", tmp_path / "receipts.jsonl")
    restored = intake.read_receipts()
    assert len(restored) == 12
    assert len({r.submission_id for r in receipts}) == 12
    for receipt in restored:
        assert intake.verify(receipt).is_dir()
        assert len({r.received_at for r in receipts if r.submission_id == receipt.submission_id}) == 1


@pytest.mark.parametrize("damage", [b'{"round_id":', b"{not json}\n", b"{}\n"])
def test_corrupt_receipt_store_refuses_read_and_write(world, damage):
    _, intake, _, _ = world
    intake.receipts.write_bytes(damage)
    with pytest.raises(IntakeError):
        intake.read_receipts()
    with pytest.raises(IntakeError):
        intake.accept(round_id="r-1", miner_id="bob", files={"SOUL.md": "B"})
    assert intake.receipts.read_bytes() == damage


def test_interrupted_receipt_publication_keeps_old_state_and_retry_recovers(world, monkeypatch):
    _, intake, receipt, _ = world
    original = intake._write_receipts
    monkeypatch.setattr(
        intake, "_write_receipts", lambda _: (_ for _ in ()).throw(OSError("injected persistence failure"))
    )
    with pytest.raises(OSError, match="injected"):
        intake.accept(round_id="r-1", miner_id="bob", files={"SOUL.md": "B"})
    assert intake.read_receipts() == [receipt]
    monkeypatch.setattr(intake, "_write_receipts", original)
    retry = intake.accept(round_id="r-1", miner_id="bob", files={"SOUL.md": "B"})
    assert len(intake.read_receipts()) == 2
    assert intake.verify(retry).is_dir()


def test_fixture_round_cannot_be_copied_into_production(world, tmp_path):
    store, _, _, _ = world
    production = RoundStore(tmp_path / "production", require_private=False)
    production.path_for("r-1").write_bytes(store.path_for("r-1").read_bytes())
    with pytest.raises(ValueError, match="trust domain"):
        production.load("r-1")


def _process_upload(paths, index):
    intake = Intake(*paths, mode="fixture", namespace="multiprocess")
    return intake.accept(round_id="r", miner_id=f"miner-{index % 8}", files={"SOUL.md": str(index % 8)}).to_record()


def test_receipts_are_safe_across_processes(tmp_path):
    import multiprocessing

    paths = (tmp_path / "bundles", tmp_path / "receipts.jsonl")
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        delivered = pool.starmap(_process_upload, [(paths, i) for i in range(32)])
    intake = Intake(*paths, mode="fixture", namespace="multiprocess")
    assert len(delivered) == 32 and len(intake.read_receipts()) == 8
    assert all(intake.verify(r).is_dir() for r in intake.read_receipts())


def test_atomic_replace_failure_exposes_no_partial_receipt(world, monkeypatch):
    import validator.persistence as persistence

    _, intake, original, _ = world
    before = intake.receipts.read_bytes()
    replace = persistence.os.replace

    def fail(source, destination):
        if destination == intake.receipts:
            raise OSError("injected replace failure")
        return replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        intake.accept(round_id="r-1", miner_id="bob", files={"SOUL.md": "new"})
    assert intake.receipts.read_bytes() == before
    assert intake.read_receipts() == [original]


def test_fixture_receipt_cannot_be_copied_to_production_intake(world, tmp_path):
    _, intake, _, _ = world
    production = Intake(tmp_path / "production-bundles", tmp_path / "production-receipts.jsonl")
    production.receipts.write_bytes(intake.receipts.read_bytes())
    with pytest.raises(IntakeError, match="trust domain"):
        production.read_receipts()


# Original source bytes must survive authentication without last-key-wins decoding.
# Provider response variants are robustness tests, not claims about GitHub's serializer.
_BAD_METADATA = [
    '{"schema_version":2,"schema_version":2}',
    '{"miner_id":"other","miner\\u005fid":"alice"}',
    '{"nested":[{"task_ids":[],"task_ids":[]}]}',
    '{"schema_version":NaN,"schema_version":2}',
    '{"schema_version":Infinity,"schema_version":2}',
    '{"schema_version":-Infinity,"schema_version":2}',
    '{"schema_version":1e9999,"schema_version":2}',
    '{"schema_version":-1e9999,"schema_version":2}',
    '{"schema_version":01,"schema_version":2}',
    '{"schema_version":2',
    "[]",
    "null",
]


def _admission_argv(store, intake, transport):
    return [
        "--repository",
        transport.repository,
        "--pr",
        str(transport.number),
        "--head",
        transport.head,
        "--round",
        "r-1",
        "--store",
        str(store.root),
        "--intake-root",
        str(intake.root),
        "--receipts",
        str(intake.receipts),
    ]


def _replace_blob(response, raw):
    import base64
    import hashlib

    wrapper = json.loads(response.stdout)
    wrapper["content"] = base64.b64encode(raw).decode()
    wrapper["sha"] = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    response.stdout = json.dumps(wrapper)
    return response


@pytest.mark.parametrize("raw", _BAD_METADATA)
@pytest.mark.parametrize("where", ["head", "base", "round"])
def test_complete_metadata_refused_by_gate_and_admission_before_effects(
    world, monkeypatch, tmp_path, capsys, raw, where
):
    from eval.strategy_track import check_one_commitment_per_round, gate
    from validator.settlement import SettlementStore

    store, intake, _, transport = world
    valid = transport.head_text
    record = json.loads(valid)
    if where == "head":
        transport.head_text = raw
    elif where == "base":
        transport.base_text = raw + "\n"
        transport.head_text = transport.base_text + valid
        assert check_one_commitment_per_round(Commitment.from_record(record), transport.base_text)
    if where != "round":
        assert gate(
            record=record,
            round_record=transport.round_record,
            receipts=intake.read_receipts(),
            base_text=transport.base_text,
            head_text=transport.head_text,
            changed_paths=["datasets/strategies.jsonl"],
            pull_request_author="alice",
        )
    before = store.path_for("r-1").read_bytes()
    receipts_before = intake.receipts.read_bytes()
    settlement = SettlementStore(tmp_path / "settlement", mode="fixture", namespace=store.identity["namespace"])

    def no_save(*args, **kwargs):
        pytest.fail("invalid metadata reached RoundStore.save")

    monkeypatch.setattr(RoundStore, "save", no_save)
    monkeypatch.setattr(type(store.load("r-1")), "submit", no_save)

    def response(argv, **kwargs):
        result = transport(argv, **kwargs)
        if where == "round" and "/contents/datasets/rounds/" in argv[6]:
            return _replace_blob(result, raw.encode())
        return result

    source = GitHubSource(transport.repository, transport=response)
    metadata = source.collect(7, round_id="r-1", expected_head=transport.head)
    with pytest.raises(AdmissionError):
        admit(metadata=metadata, round_id="r-1", store=store, intake=intake)
    assert main(_admission_argv(store, intake, transport), transport=response) == 2
    output = capsys.readouterr()
    assert "admission refused:" in output.err and "Traceback" not in output.err
    assert not output.out
    restored = store.load("r-1")
    assert not restored.admissions and not restored.submissions and not restored.snapshot()["verdicts"]
    assert store.path_for("r-1").read_bytes() == before
    assert intake.receipts.read_bytes() == receipts_before
    assert not list((tmp_path / "cards").glob("*.json"))
    assert not settlement.actions()
    with settlement.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 0


@pytest.mark.parametrize(
    "where", ["user", "pr-first", "pr-second", "files", "base-content", "head-content", "round-content"]
)
@pytest.mark.parametrize("assertion", ['"x":0,"x":0', '"x":NaN,"x":0', '"x":1e999,"x":0', '"x":0,"\\u0078":1'])
def test_nested_provider_responses_refuse_before_admission(world, where, assertion, capsys):
    store, intake, _, transport = world
    before = store.path_for("r-1").read_bytes()
    reads = 0

    def response(argv, **kwargs):
        nonlocal reads
        result = transport(argv, **kwargs)
        endpoint = argv[6]
        if endpoint.endswith("pulls/7"):
            reads += 1
        selected = (
            where == "user"
            and endpoint == "user"
            or where == "pr-first"
            and endpoint.endswith("pulls/7")
            and reads == 1
            or where == "pr-second"
            and endpoint.endswith("pulls/7")
            and reads == 2
            or where == "files"
            and "/files?" in endpoint
            or where == "base-content"
            and "/strategies.jsonl?" in endpoint
            and endpoint.endswith(transport.base)
            or where == "head-content"
            and "/strategies.jsonl?" in endpoint
            and endpoint.endswith(transport.head)
            or where == "round-content"
            and "/rounds/" in endpoint
        )
        if selected:
            # An unknown nested provider field still cannot hide contradictory assertions.
            result.stdout = result.stdout.replace("{", '{"provider_extra":[{' + assertion + "}],", 1)
        return result

    assert main(_admission_argv(store, intake, transport), transport=response) == 2
    assert "malformed JSON" in capsys.readouterr().err
    assert store.path_for("r-1").read_bytes() == before
    assert not store.load("r-1").submissions


@pytest.mark.parametrize("raw", [b"\xff", b'{"round_id":"r-1"}\n{}', b"[" * 2000 + b"]" * 2000])
def test_invalid_blob_encoding_complete_value_and_depth_refuse(world, raw, capsys):
    store, intake, _, transport = world
    before = store.path_for("r-1").read_bytes()

    def response(argv, **kwargs):
        result = transport(argv, **kwargs)
        if "/rounds/" in argv[6]:
            return _replace_blob(result, raw)
        return result

    assert main(_admission_argv(store, intake, transport), transport=response) == 2
    assert "admission refused:" in capsys.readouterr().err
    assert store.path_for("r-1").read_bytes() == before


@pytest.mark.parametrize(
    "raw", ["{}", '{"round_id":[],"miner_id":"other"}', '{"round_id":"old","miner_id":"other","schema_version":"bad"}']
)
def test_malformed_historical_commitment_is_not_silently_skipped(world, raw):
    from eval.strategy_track import check_one_commitment_per_round

    store, intake, _, transport = world
    commitment = Commitment.from_record(json.loads(transport.head_text))
    transport.base_text = raw + "\n"
    transport.head_text = transport.base_text + transport.head_text
    assert check_one_commitment_per_round(commitment, transport.base_text)
    assert main(_admission_argv(store, intake, transport), transport=transport) == 2
    assert not store.load("r-1").admissions


def test_valid_fork_blob_lists_history_literal_strings_and_retry(world, capsys):
    store, intake, _, transport = world
    record = json.loads(transport.head_text)
    # Literal JSON-looking payload, NaN and Unicode line separators are ordinary strings.
    record["notes"] = 'NaN Infinity {"miner_id":"other","miner_id":"alice"}\u2028text'
    historical = {**record, "round_id": "r-old", "miner_id": "bob"}
    transport.base_text = json.dumps(historical, ensure_ascii=False) + "\n"
    transport.head_text = transport.base_text + json.dumps(record, ensure_ascii=False)
    endpoints = []

    def response(argv, **kwargs):
        endpoints.append(argv[6])
        result = transport(argv, **kwargs)
        if argv[6].endswith("pulls/7"):
            value = json.loads(result.stdout)
            value["head"]["repo"]["full_name"] = "genuine-fork/spark"
            value["body"] = 'NaN {"state":"closed","state":"open"}'
            result.stdout = json.dumps(value)
        return result

    argv = _admission_argv(store, intake, transport)
    assert main(argv, transport=response) == 0
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["registry_delta"]["notes"] == record["notes"]
    before = store.path_for("r-1").read_bytes()
    assert main(argv, transport=response) == 0
    assert json.loads(capsys.readouterr().out) == admitted
    assert store.path_for("r-1").read_bytes() == before
    assert f"repos/genuine-fork/spark/contents/datasets/strategies.jsonl?ref={transport.head}" in endpoints
    assert len(store.load("r-1").admissions) == 1


def test_validly_decoded_fork_content_with_wrong_blob_sha_still_refuses(world):
    store, intake, _, transport = world

    def response(argv, **kwargs):
        result = transport(argv, **kwargs)
        value = json.loads(result.stdout)
        if "/contents/" in argv[6]:
            value["sha"] = "0" * 40
        elif argv[6].endswith("pulls/7"):
            value["head"]["repo"]["full_name"] = "genuine-fork/spark"
        result.stdout = json.dumps(value)
        return result

    assert main(_admission_argv(store, intake, transport), transport=response) == 2
    assert not store.load("r-1").admissions


@pytest.mark.parametrize("field", ["miner_id", "bundle_sha256", "task_ids"])
@pytest.mark.parametrize("where", ["head", "base"])
def test_valid_commitment_cannot_hide_overwritten_identity(world, field, where, capsys):
    store, intake, _, transport = world
    candidate = transport.head_text
    record = json.loads(candidate)
    # Historical same-round alice rewritten to bob would evade exclusion under
    # last-key-wins parsing. All other fields are a valid committed record.
    if where == "base":
        record["miner_id"] = "bob"
    selected = json.dumps(record[field])
    other = json.dumps(
        {
            "miner_id": "alice" if where == "base" else "bob",
            "bundle_sha256": "sha256:" + "0" * 64,
            "task_ids": ["wrong-task"],
        }[field]
    )
    raw = json.dumps(record).replace(
        json.dumps(field) + ": " + selected,
        json.dumps(field) + ": " + other + ", " + json.dumps(field) + ": " + selected,
    )
    if where == "head":
        transport.head_text = raw
    else:
        transport.base_text = raw + "\n"
        transport.head_text = transport.base_text + candidate
    before = store.path_for("r-1").read_bytes()
    assert main(_admission_argv(store, intake, transport), transport=transport) == 2
    assert "duplicate JSON key" in capsys.readouterr().err
    assert store.path_for("r-1").read_bytes() == before


@pytest.mark.parametrize("field", ["task_ids", "commitment", "replicas"])
def test_authenticated_round_cannot_hide_overwritten_scope(world, field, capsys):
    store, intake, _, transport = world
    raw = json.dumps(transport.round_record)
    selected = json.dumps(transport.round_record[field])
    raw = raw.replace(
        json.dumps(field) + ": " + selected, json.dumps(field) + ": null, " + json.dumps(field) + ": " + selected
    )

    def response(argv, **kwargs):
        result = transport(argv, **kwargs)
        if "/rounds/" in argv[6]:
            return _replace_blob(result, raw.encode())
        return result

    before = store.path_for("r-1").read_bytes()
    assert main(_admission_argv(store, intake, transport), transport=response) == 2
    assert "duplicate JSON key" in capsys.readouterr().err
    assert store.path_for("r-1").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [{"bundle_sha256": []}, {"bundle_sha256": "sha256:bad"}, {"task_ids": ["t", "t"]}, {"task_ids": {}}, {"notes": {}}],
)
def test_historical_optional_commitment_fields_remain_well_formed(world, change):
    store, intake, _, transport = world
    historical = {"round_id": "old", "miner_id": "bob", **change}
    transport.base_text = json.dumps(historical) + "\n"
    transport.head_text = transport.base_text + transport.head_text
    assert main(_admission_argv(store, intake, transport), transport=transport) == 2
    assert not store.load("r-1").admissions

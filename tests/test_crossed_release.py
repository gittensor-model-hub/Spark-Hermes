"""Independent numerical expectations plus real four-cell producer/authority consumers."""

import copy
import json
import random
from pathlib import Path

import pytest
from release_support import setup_release

from admin.artifacts import StageError, content_digest, read_record, write_record
from admin.evaluation import execute_crossed
from admin.parents import ParentAuthority
from admin.release import ReleaseAuthority
from admin.serving_identity import TrustedServing
from hermes.cotraining import CELLS, DEFAULT_POLICY, crossed_report


def numerical(families=6, attempts=10):
    policy = dict(DEFAULT_POLICY)
    schedule = [
        {"task_id": f"t{f}", "family_id": f"f{f}", "attempt_id": str(i), "seed": i}
        for f in range(families)
        for i in range(attempts)
    ]
    matrix = {
        "policy": policy,
        "policy_hash": content_digest(policy),
        "schedule": schedule,
        "factors": {"agents": ["a0", "a1"], "models": ["m0", "m1"]},
        "evaluator": "e",
        "workload": "w",
        "environment": "env",
        "budget": {"steps": 4},
        "sampling": {"temperature": 0},
        "cells": {},
    }
    for name, count in zip(CELLS, (2, 4, 5, 9)):
        matrix["cells"][name] = {
            **{k: matrix[k] for k in ("evaluator", "workload", "environment", "budget", "sampling", "policy_hash")},
            "agent": matrix["factors"]["agents"][int(name[1])],
            "model": matrix["factors"]["models"][int(name[2])],
            "rows": [{**r, "success": int(r["attempt_id"]) < count, "tokens": 100, "latency": 1.0} for r in schedule],
        }
    return matrix


def test_independent_constant_family_oracle():
    report = crossed_report(numerical())
    assert report["agent_effect"] == pytest.approx(0.2)
    assert report["model_effect"] == pytest.approx(0.3)
    assert report["interaction"] == pytest.approx(0.2)
    assert report["joint_gain"] == pytest.approx(0.7)
    assert report["joint_interval"] == pytest.approx([0.7, 0.7])
    assert report["eligible"] and not report["authorizes_activation"]


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "agent",
        "model",
        "evaluator",
        "workload",
        "environment",
        "budget",
        "sampling",
        "policy_hash",
        "seed",
        "attempt",
        "duplicate",
        "count",
        "tokens",
        "latency",
        "nan",
        "success",
    ],
)
def test_every_cell_and_identity_refuses(cell, change):
    m = numerical()
    c = m["cells"][cell]
    if change == "missing":
        del m["cells"][cell]
    elif change in {"agent", "model", "evaluator", "workload", "environment", "budget", "sampling", "policy_hash"}:
        c[change] = "changed"
    elif change in {"seed", "attempt"}:
        c["rows"][0]["seed" if change == "seed" else "attempt_id"] = 999
    elif change == "duplicate":
        c["rows"][0] = c["rows"][1]
    elif change == "count":
        c["rows"].pop()
    elif change == "nan":
        c["rows"][0]["tokens"] = float("nan")
    elif change == "success":
        c["rows"][0]["success"] = 1
    else:
        c["rows"][0][change] = -1
    with pytest.raises(StageError):
        crossed_report(m)


@pytest.mark.parametrize("field", list(DEFAULT_POLICY))
def test_missing_policy_fields_fail_closed(field):
    m = numerical()
    del m["policy"][field]
    m["policy_hash"] = content_digest(m["policy"])
    with pytest.raises(StageError):
        crossed_report(m)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("noise", "lower bound"),
        ("family", "per-family"),
        ("tokens", "tokens regression"),
        ("latency", "latency regression"),
        ("zero", "baseline must be positive"),
        ("families", "independent families"),
        ("attempts", "attempts per task"),
        ("boundary", "min_gain"),
    ],
)
def test_uncertainty_support_and_guardrails(change, reason):
    m = numerical(families=5 if change == "families" else 6, attempts=9 if change == "attempts" else 10)
    if change == "noise":
        for r in m["cells"]["Q00"]["rows"]:
            r["success"] = int(r["attempt_id"]) < 5
        for r in m["cells"]["Q11"]["rows"]:
            r["success"] = int(r["attempt_id"]) < (10 if int(r["family_id"][1:]) < 3 else 1)
    if change == "family":
        for r in m["cells"]["Q11"]["rows"]:
            if r["family_id"] == "f0":
                r["success"] = False
    if change in {"tokens", "latency"}:
        # An expensive failed attempt must still count in the cost guardrail.
        m["cells"]["Q11"]["rows"][-1][change] = 10000
    if change == "zero":
        for r in m["cells"]["Q00"]["rows"]:
            r["tokens"] = 0
    if change == "boundary":
        m["policy"]["min_gain"] = 0.7
        m["policy_hash"] = content_digest(m["policy"])
        for c in m["cells"].values():
            c["policy_hash"] = m["policy_hash"]
    report = crossed_report(m)
    assert not report["eligible"]
    assert any(reason in r for r in report["reasons"])
    if change == "noise":
        # Separate NumPy implementation of paired FAMILY resampling and percentile interpolation.
        import numpy as np

        deltas = [0.5] * 3 + [-0.4] * 3
        rng = random.Random(20260912)
        samples = [sum(deltas[int(rng.random() * 6)] for _ in range(6)) / 6 for _ in range(2000)]
        assert report["joint_interval"] == pytest.approx(np.quantile(samples, [0.025, 0.975]))


def test_real_producer_release_parent_and_artifact_swaps(tmp_path):
    authority, candidates, workspaces, freeze = setup_release(tmp_path)
    plan = authority.freeze(**freeze)
    evidence = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    report = evidence["payload"]["report"]
    assert report["joint_gain"] == pytest.approx(0.7)
    assert report["joint_interval"] == pytest.approx([0.7, 0.7])
    decision = authority.decide(evidence["id"])
    assert decision["payload"]["result"] == "accepted", decision
    assert decision["payload"]["fixture_only"] is True
    state = authority.activate(decision["id"])
    assert state["candidate"] == candidates[1]["id"] and state["generation"] == 1
    assert authority.activate(decision["id"])["generation"] == 1
    model = candidates[1]["payload"]["model"]
    parent, _ = ParentAuthority(authority.store.root).approved_parent(
        decision["id"],
        identity=workspaces[1].identity,
        profile=model["profile"],
        repository=model["base_model"],
        revision=model["revision"],
    )
    assert parent == Path(model["merged"])
    candidate = candidates[1]["payload"]
    artifacts = [Path(candidate[k]["path"]) for k in ("agent", "prepared", "recipe", "workload")]
    artifacts += [
        Path(model["record"]),
        Path(model["merged"]) / "model.safetensors",
        Path(model["merged"]) / "tokenizer.json",
        Path(candidate["parent"]["path"]) / "model.safetensors",
        workspaces[1].corpus / "sft.jsonl",
        Path(evidence["payload"]["artifact"]["path"]),
        Path(evidence["payload"]["logs"]["Q10"]["path"]),
    ]
    for path in artifacts:
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises((StageError, ValueError)):
                authority.resolve_decision(decision["id"])
            with pytest.raises((StageError, ValueError)):
                ParentAuthority(authority.store.root).approved_parent(
                    decision["id"],
                    identity=workspaces[1].identity,
                    profile=model["profile"],
                    repository=model["base_model"],
                    revision=model["revision"],
                )
        finally:
            path.write_bytes(original)
    assert authority.status()["generation"] == 1
    with pytest.raises(StageError, match="already started"):
        execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    changed = copy.deepcopy(freeze)
    changed["sampling"]["temperature"] = 0.1
    with pytest.raises(StageError):
        authority.freeze(**changed)
    production = ReleaseAuthority(tmp_path / "production", mode="production", namespace=authority.identity["namespace"])
    with pytest.raises(StageError):
        production.configure(
            candidates=authority.candidates().store.root,
            incumbent=candidates[0]["id"],
            data_policy=tmp_path / "confirmation.json",
            policy=DEFAULT_POLICY,
        )
    # Copied JSON is never a committed decision in another configured issuer.
    with pytest.raises(StageError):
        production.resolve_decision(decision["id"])


def test_real_strict_cli_and_refused_incumbent(tmp_path):
    from learning_support import cli

    authority, candidates, _, freeze = setup_release(tmp_path, policy={**DEFAULT_POLICY, "min_gain": 0.8})
    issued = json.loads(
        cli(
            tmp_path,
            "admin.cli",
            "cotraining",
            "freeze",
            "--root",
            authority.store.root,
            "--config",
            tmp_path / "freeze.json",
        ).stdout
    )
    result = json.loads(
        cli(
            tmp_path,
            "admin.cli",
            "cotraining",
            "run",
            "--root",
            authority.store.root,
            "--id",
            issued["id"],
            "--allow-unsandboxed",
        ).stdout
    )
    refused = json.loads(
        cli(
            tmp_path, "admin.cli", "release", "decide", "--root", authority.store.root, "--id", result["id"], expected=3
        ).stdout
    )
    cli(tmp_path, "admin.cli", "release", "activate", "--root", authority.store.root, "--id", refused["id"], expected=2)
    assert authority.status()["candidate"] == candidates[0]["id"]


def test_trusted_serving_controlled_authenticated_transport(monkeypatch):
    config = {
        "url": "https://serving.example/v1",
        "api_key_env": "SPARK_TEST_KEY",
        "alias": "reused-alias",
        "deployment_id": "deployment-1",
        "engine": "controlled",
        "precision": "bf16",
        "device": "cpu-fixture",
        "environment": "fixed",
    }
    options = {
        "model_id": "sha256:" + "a" * 64,
        "origin": {"mode": "fixture", "namespace": "controlled"},
        "sampling": {"temperature": 0, "top_p": 1},
        "budget": {"max_tokens": 100},
    }
    with pytest.raises(StageError, match="credential"):
        TrustedServing(config, **options)
    monkeypatch.setenv("SPARK_TEST_KEY", "fixture-secret")
    adapter = TrustedServing(config, **options)
    calls = []

    class Response:
        def __init__(self, request):
            self.url = request.full_url
            self.request = request

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            payload = json.loads(self.request.data)
            calls.append(payload)
            assert self.request.headers["Authorization"] == "Bearer fixture-secret"
            nonce = payload.get("nonce", payload.get("spark_identity", {}).get("nonce"))
            identity = {**adapter.expected, "nonce": nonce}
            return json.dumps(
                identity
                if self.url.endswith("/identity")
                else {
                    "spark_identity": identity,
                    "choices": [{"message": {"content": "CPU fixture"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            return Response(request)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    complete = adapter.completion(42)
    assert complete([{"role": "user", "content": "hello"}])[0] == "CPU fixture"
    assert calls[-1]["seed"] == 42 and calls[-1]["max_tokens"] == 100
    for field in adapter.expected:
        bad = {**adapter.expected, "nonce": "n", field: "changed"}
        with pytest.raises(StageError):
            adapter._check(bad, "n")
    with pytest.raises(StageError):
        adapter._check({}, "n")


def test_family_macro_is_not_task_or_attempt_weighted():
    m = numerical()
    # Add nine hard tasks to one family. That family still receives exactly one sixth of total weight.
    for i in range(9):
        for attempt in range(10):
            row = {"task_id": f"extra-{i}", "family_id": "f0", "attempt_id": str(attempt), "seed": attempt}
            m["schedule"].append(row)
            for c in m["cells"].values():
                c["rows"].append({**row, "success": False, "tokens": 100, "latency": 1})
    r = crossed_report(m)
    assert r["cells"]["Q11"]["families"]["f0"] == pytest.approx(0.09)
    assert r["cells"]["Q11"]["quality"] == pytest.approx((0.09 + 5 * 0.9) / 6)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("cost", "tokens regression"),
        ("noise", "lower bound"),
        ("regression", "per-family"),
        ("support", "independent families"),
    ],
)
def test_actual_producer_guardrail_refusal(tmp_path, change, reason):
    from admin.candidates import file_identity

    authority, candidates, _, freeze = setup_release(tmp_path, families=5 if change == "support" else 6)
    model_id = candidates[1]["payload"]["model_id"]
    config = freeze["serving"][model_id]
    path = Path(config["fixture"]["path"])
    script = read_record(path)
    rows = script["agents"][candidates[1]["payload"]["agent_id"]]
    if change == "cost":
        rows["confirm-0"]["9"]["prompt_tokens"] = 10000
    if change in {"regression", "noise"}:
        for task_id, attempts in rows.items():
            if change == "noise" or task_id == "confirm-0":
                for row in attempts.values():
                    row["responses"][0] = row["responses"][0].replace("printf '4'", "printf '5'")
    write_record(path, script)
    config["fixture"] = file_identity(path)
    plan = authority.freeze(**freeze)
    evidence = execute_crossed(authority, plan["id"], allow_unsandboxed=True)
    decision = authority.decide(evidence["id"])
    assert decision["payload"]["result"] == "refused"
    assert any(reason in r for r in decision["payload"]["reasons"])
    before = authority.status()
    with pytest.raises(StageError):
        authority.activate(decision["id"])
    assert authority.status() == before


def test_freeze_rejects_reused_confirmation_and_known_exposure(tmp_path):
    from admin.candidates import file_identity

    authority, _, _, freeze = setup_release(tmp_path)
    authority.freeze(**freeze)
    for field in ("sampling", "budget", "schedule"):
        changed = copy.deepcopy(freeze)
        if field == "sampling":
            changed[field]["temperature"] = 0.2
        elif field == "budget":
            changed[field]["max_tokens"] = 2048
        else:
            changed[field][0]["seed"] = 50
        with pytest.raises(StageError, match="already used"):
            authority.freeze(**changed)
    config = authority.configuration()
    # A separate configured fixture authority still must reject declared selection exposure.
    policy_path = tmp_path / "exposed.json"
    data = read_record(Path(config["data_policy"]["path"]))
    data["memberships"][-1]["exposure"] = ["selection"]
    write_record(policy_path, data)
    other = ReleaseAuthority(tmp_path / "exposed-release", mode="fixture", namespace=authority.identity["namespace"])
    other.configure(
        candidates=authority.candidates().store.root,
        incumbent=freeze["old"],
        data_policy=policy_path,
        policy=DEFAULT_POLICY,
    )
    with pytest.raises(StageError, match="selection-used"):
        other.freeze(**freeze)
    original = Path(config["data_policy"]["path"])
    raw = original.read_bytes()
    original.write_bytes(raw + b" ")
    with pytest.raises(StageError):
        authority.plan(authority.experiments.records(kind="crossed-plan")[0]["id"])
    original.write_bytes(raw)
    assert file_identity(original) == config["data_policy"]


def test_evaluator_environment_and_issuer_swaps(tmp_path, monkeypatch):
    import admin.candidates

    authority, candidates, _, freeze = setup_release(tmp_path)
    source = authority.candidates()
    actual = admin.candidates.crossed_runtime_identity()
    for key in ("evaluator", "environment", "files"):
        changed = {**actual, key: "changed"}
        with monkeypatch.context() as context:
            context.setattr(admin.candidates, "crossed_runtime_identity", lambda: changed)
            with pytest.raises(StageError, match="evaluator/environment"):
                source.resolve(candidates[0]["id"])
    from admin.artifacts import AuthorityStore

    rogue = AuthorityStore(
        tmp_path / "rogue", role="candidate", mode="fixture", namespace=authority.identity["namespace"]
    )
    copied = rogue.put("candidate", candidates[0]["payload"])
    with pytest.raises(StageError, match="issuer"):
        admin.candidates.CandidateStore(rogue.root).resolve(copied["id"])
    changed = copy.deepcopy(freeze)
    changed["serving"].pop(next(iter(changed["serving"])))
    with pytest.raises(StageError, match="both exact"):
        authority.freeze(**changed)


def test_second_release_uses_fresh_families_strict_parent_and_rollback(tmp_path):
    from release_support import successor

    authority, candidates, _, freeze = setup_release(tmp_path, pool_families=12)
    first = execute_crossed(authority, authority.freeze(**freeze)["id"], allow_unsandboxed=True)
    approval = authority.decide(first["id"])
    authority.activate(approval["id"])
    third, next_plan = successor(tmp_path, authority, candidates[1], approval)
    second = execute_crossed(authority, authority.freeze(**next_plan)["id"], allow_unsandboxed=True)
    next_approval = authority.decide(second["id"])
    assert next_approval["payload"]["result"] == "accepted"
    assert authority.activate(next_approval["id"])["candidate"] == third["id"]
    # Reconsidering the old evidence reports a stale baseline and leaves incumbent alone.
    stale = authority.decide(first["id"])
    assert stale["payload"]["result"] == "refused"
    assert "stale evaluation" in stale["payload"]["reasons"][0]
    with pytest.raises(StageError):
        authority.activate(stale["id"])
    rollback = authority.activate(approval["id"], rollback=True)
    assert rollback["generation"] == 3
    assert rollback["candidate"] == candidates[1]["id"]
    assert rollback["history"][-1]["intent"] == "rollback to previously accepted pair"
    # ABA: the baseline pair matches again, but the old measurement generation does not.
    with pytest.raises(StageError, match="stale approval"):
        authority.activate(next_approval["id"])
    assert authority.decide(second["id"])["payload"]["result"] == "refused"
    assert authority.activate(next_approval["id"], rollback=True)["generation"] == 4


def test_trusted_serving_real_local_https(tmp_path, monkeypatch):
    """A real TLS socket exercises authentication/nonce/identity parsing; no model is loaded."""
    import ssl
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("SPARK_LOCAL_TLS_KEY", "local-fixture-credential")
    observations = []
    identity = {
        "schema": "spark-serving-identity-v1",
        "model_id": "sha256:" + "b" * 64,
        "representation": "merged-sft",
        "deployment_id": "local-scripted-deployment",
        "engine": "scripted-CPU-boundary",
        "precision": "fixture",
        "device": "cpu",
        "environment": "local-tls-fixture",
        "mode": "fixture",
        "namespace": "local-tls",
    }
    corrupt = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observations.append(request)
            assert self.headers.get("Authorization") == "Bearer local-fixture-credential"
            nonce = request.get("nonce", request.get("spark_identity", {}).get("nonce"))
            observed = {**identity, "nonce": nonce}
            if corrupt:
                observed["model_id"] = "sha256:" + "c" * 64
            payload = (
                observed
                if self.path.endswith("/identity")
                else {
                    "spark_identity": observed,
                    "choices": [{"message": {"content": "scripted CPU boundary"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                }
            )
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {
            "url": f"https://localhost:{server.server_port}/v1",
            "api_key_env": "SPARK_LOCAL_TLS_KEY",
            "alias": "reused-alias",
            **{k: identity[k] for k in ("deployment_id", "engine", "precision", "device", "environment")},
        }
        adapter = TrustedServing(
            config,
            model_id=identity["model_id"],
            origin={"mode": "fixture", "namespace": "local-tls"},
            sampling={"temperature": 0, "top_p": 1},
            budget={"max_tokens": 10},
        )
        complete = adapter.completion(123)
        assert complete([{"role": "user", "content": "CPU fixture"}])[0] == "scripted CPU boundary"
        assert observations[-1]["seed"] == 123
        corrupt.append(True)
        with pytest.raises(StageError, match="identity differs"):
            complete([{"role": "user", "content": "CPU fixture"}])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_initial_parent_uses_actual_local_hub_cache_without_download(tmp_path, monkeypatch):
    """Test the read-only cache adapter, not production approval, using labelled byte fixtures."""
    import httpx
    from huggingface_hub import constants

    from admin.candidates import check_initial_parent

    revision = "a" * 40
    cache = tmp_path / "hub"
    snapshot = cache / "models--fixture--upstream" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    write_record(snapshot / "config.json", {"fixture_only": True, "model_type": "qwen3_5"})
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))

    def forbidden_network(*args, **kwargs):
        raise AssertionError("the parent-cache adapter must never attempt a download")

    monkeypatch.setattr(httpx.Client, "send", forbidden_network)
    prepared = {"base_model": "fixture/upstream", "revision": revision}
    identity = {"mode": "production", "namespace": "cache-adapter-unit-test"}
    check_initial_parent(snapshot, prepared, identity=identity)
    with pytest.raises(StageError, match="exact pinned"):
        check_initial_parent(tmp_path / "renamed-alias", prepared, identity=identity)
    with pytest.raises(StageError, match="no download was attempted"):
        check_initial_parent(snapshot, {**prepared, "revision": "b" * 40}, identity=identity)


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("change", ("boolean-seed", "boolean-budget", "sampling-type"))
def test_exact_paired_configuration_types(cell, change):
    matrix = numerical()
    if change == "boolean-seed":
        matrix["cells"][cell]["rows"][0]["seed"] = False
    elif change == "boolean-budget":
        matrix["budget"] = {"steps": 1}
        for c in matrix["cells"].values():
            c["budget"] = {"steps": 1}
        matrix["cells"][cell]["budget"] = {"steps": True}
    else:
        matrix["cells"][cell]["sampling"] = {"temperature": 0.0}
    with pytest.raises(StageError):
        crossed_report(matrix)

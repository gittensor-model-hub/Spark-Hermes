"""Real CPU runner handoff; fixture completions never authorize production use."""

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from release_support import setup_release

from admin.artifacts import StageError, read_record, write_record
from admin.competition_pair import FixtureServing, base_agent_id, base_profile, build_epoch
from admin.serving_identity import TrustedServing
from hermesbench import runner
from hermesbench.sink import read_episodes
from hermesbench.tasks import Task
from validator.intake import bundle_digest
from validator.judge import runner_for
from validator.persistence import state_identity

SALT = "competition CPU fixture salt"


def setup_execution(root, monkeypatch, release=None):
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("SPARKDISTILL_WITHHELD_ROOT", raising=False)
    monkeypatch.setenv("HERMESBENCH_WITHHELD_SALT", SALT)
    origin_root = root / "rounds"
    identity = state_identity(
        origin_root, mode="fixture", namespace=release.identity["namespace"] if release else "bootstrap-cpu"
    )
    task = Task(
        task_id="cpu-pair",
        prompt="Write four into answer.txt.",
        tools=("terminal",),
        verify='test "$(cat answer.txt)" = 4',
        hidden_verify='test "$(cat answer.txt)" = 4',
        max_steps=4,
    )
    tasks = root / "tasks"
    (tasks / "v0").mkdir(parents=True)
    (tasks / "v0/task.yaml").write_text(yaml.safe_dump(asdict(task)))
    epoch = build_epoch(
        [task],
        attempt_ids=[str(i) for i in range(10)],
        epoch_id=None if release else "bootstrap-4b",
        profile="rtx5090-poc",
        release_root=release.store.root if release else None,
        task_root=tasks,
        salt=SALT,
    )
    fixture_root = release.store.root if release else origin_root
    script = {
        "schema": "spark-serving-fixture-v1",
        "origin": state_identity(fixture_root),
        "model_id": epoch["model_revision"],
        "agents": {
            epoch["agent_id"]: {
                task.task_id: {
                    str(i): {
                        "responses": [
                            "<tool_call>\n<function=terminal>\n<parameter=command>\n"
                            + ("printf 4 > answer.txt" if i < 3 else "printf 5 > answer.txt")
                            + "\n</parameter>\n</function>\n</tool_call>",
                            "CPU fixture complete.",
                        ],
                        "prompt_tokens": 50,
                        "completion_tokens": 50,
                    }
                    for i in range(10)
                }
            }
        },
    }
    path = root / "serving.json"
    write_record(path, script)
    context = {"epoch": epoch, "round_id": "r-1", "origin": identity, "bundle_sha256": ""}
    options = dict(
        round_id="r-1",
        base_url="",
        model=epoch["model_revision"] if release else base_profile("rtx5090-poc")["repository"],
        api_key_env="NONE",
        task_id=task.task_id,
        repeats=10,
        repo_root=None,
        allow_unsandboxed=True,
        task_root=tasks,
        evaluation_context=context,
        profile="rtx5090-poc",
        release_root=release.store.root if release else None,
        fixture_root=fixture_root,
        fixture_serving=path,
    )
    return options, context, script


def test_bootstrap_4b_baseline_executes_real_tools_and_stamps_profile(tmp_path, monkeypatch):
    options, context, _ = setup_execution(tmp_path, monkeypatch)
    log = runner_for(**options)("baseline", None, tmp_path / "episodes")
    rows = list(read_episodes(log))
    assert len(rows) == 10
    assert sum(r["metrics"]["success"] for r in rows) == 3
    assert all(r["metrics"]["hidden_passed"] == r["metrics"]["success"] for r in rows)
    assert all(r["metrics"]["tool_calls"] == 1 and r["metrics"]["tokens_used"] == 200 for r in rows)
    assert all(r["evidence"]["model_revision"] == base_profile("rtx5090-poc")["revision"] for r in rows)
    assert all(r["evidence"]["agent_id"] == base_agent_id("rtx5090-poc") for r in rows)
    assert {r["evidence"]["attempt_id"] for r in rows} == set(context["epoch"]["attempt_ids"])
    assert (tmp_path / "episodes/ws-baseline/cpu-pair/answer.txt").read_text() == "4"
    assert all(r["trajectory"]["metadata"]["executed"] for r in rows)


def test_derived_pair_and_miner_bundle_reach_policy(tmp_path, monkeypatch):
    authority, _, _, _ = setup_release(tmp_path / "release-source")
    options, context, script = setup_execution(tmp_path / "run", monkeypatch, authority)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SOUL.md").write_text("Exact miner CRLF\r\nsecond line\r\n", newline="")
    context["bundle_sha256"] = bundle_digest({"SOUL.md": (bundle / "SOUL.md").read_bytes().decode()})
    observed = []
    complete = FixtureServing.completion

    def capture(self, task_id, attempt):
        call = complete(self, task_id, attempt)

        def record(messages, **kwargs):
            observed.append(copy.deepcopy(messages))
            return call(messages, **kwargs)

        return record

    monkeypatch.setattr(FixtureServing, "completion", capture)
    log = runner_for(**options)("alice", bundle, tmp_path / "episodes")
    rows = list(read_episodes(log))
    assert len(rows) == 10 and len(observed) == 20
    assert all("CPU fixture agent 0" in m[0]["content"] and "Exact miner CRLF" in m[0]["content"] for m in observed)
    assert all(r["evidence"]["model_revision"] == script["model_id"] for r in rows)
    assert all(r["evidence"]["incumbent"] == context["epoch"]["incumbent"] for r in rows)
    assert all(r["evidence"]["bundle_sha256"] == context["bundle_sha256"] for r in rows)


@pytest.mark.parametrize(
    "change", ["model", "agent", "generation", "bool-generation", "missing-release", "agent-bytes"]
)
def test_derived_wrong_or_stale_pair_refuses_before_completion(tmp_path, monkeypatch, change):
    authority, _, _, _ = setup_release(tmp_path / "release-source")
    options, context, _ = setup_execution(tmp_path / "run", monkeypatch, authority)
    if change == "model":
        options["model"] = "sha256:" + "e" * 64
    elif change == "agent":
        context["epoch"]["agent_id"] = "sha256:" + "e" * 64
    elif change == "generation":
        context["epoch"]["incumbent"]["generation"] += 1
    elif change == "bool-generation":
        context["epoch"]["incumbent"]["generation"] = False
    elif change == "missing-release":
        options["release_root"] = None
    else:
        path = Path(authority.active_pair()["agent"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    calls = []
    monkeypatch.setattr(FixtureServing, "completion", lambda *a: calls.append(a))
    with pytest.raises((RuntimeError, ValueError)):
        runner_for(**options)("baseline", None, tmp_path / "episodes")
    assert not calls
    assert not list((tmp_path / "episodes").rglob("*.jsonl"))


@pytest.mark.parametrize("change", ["fixture-origin", "fixture-model", "production", "context", "baseline-to-miner"])
def test_fixture_authority_and_expected_baseline_cannot_be_rebound(tmp_path, monkeypatch, change):
    options, context, script = setup_execution(tmp_path, monkeypatch)
    if change == "fixture-origin":
        script["origin"]["issuer"] = "untrusted"
    elif change == "fixture-model":
        script["model_id"] = "wrong-model"
    elif change == "production":
        options["fixture_root"] = tmp_path / "production"
        context["origin"] = state_identity(options["fixture_root"])
        script["origin"] = context["origin"]
    else:
        actual = runner.main

        def change_transport(argv):
            if change == "context":
                path = Path(argv[argv.index("--evaluation-context") + 1])
                record = read_record(path)
                record["epoch"]["model_revision"] = "new expectation"
                write_record(path, record)
            else:
                bundle = tmp_path / "unexpected-bundle"
                bundle.mkdir()
                argv += ["--miner-dir", str(bundle)]
            return actual(argv)

        monkeypatch.setattr(runner, "main", change_transport)
    write_record(options["fixture_serving"], script)
    with pytest.raises((RuntimeError, ValueError)):
        runner_for(**options)("baseline", None, tmp_path / "episodes")
    assert not list((tmp_path / "episodes").rglob("*.jsonl"))


def test_derived_requires_trusted_serving_and_does_not_fall_back_to_alias(tmp_path, monkeypatch):
    authority, _, _, _ = setup_release(tmp_path / "release-source")
    options, _, _ = setup_execution(tmp_path / "run", monkeypatch, authority)
    options.update(fixture_serving=None, fixture_root=None, base_url="http://127.0.0.1:1/v1")
    with pytest.raises(StageError, match="requires trusted serving"):
        runner_for(**options)("baseline", None, tmp_path / "episodes")
    assert not list((tmp_path / "episodes").rglob("*.jsonl"))


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_trusted_serving_verifies_exact_derived_model_each_completion(tmp_path, monkeypatch, wrong_identity):
    authority, _, _, _ = setup_release(tmp_path / "release-source")
    options, context, script = setup_execution(tmp_path / "run", monkeypatch, authority)
    config = tmp_path / "trusted-serving.json"
    write_record(
        config,
        {
            "url": "https://controlled-serving.invalid",
            "api_key_env": "SPARK_TEST_SERVING_KEY",
            "alias": "active",
            "deployment_id": "controlled-cpu-serving-fixture",
            "engine": "controlled-fixture",
            "precision": "bf16",
            "device": "cpu-fixture",
            "environment": "explicit-test-fixture",
        },
    )
    monkeypatch.setenv("SPARK_TEST_SERVING_KEY", "test-fixture-credential")
    options.update(fixture_serving=None, fixture_root=None, serving_config=config)
    calls, counts = [], {}

    def request(self, route, payload):
        calls.append((route, copy.deepcopy(payload)))
        nonce = payload.get("nonce", payload.get("spark_identity", {}).get("nonce"))
        identity = {**self.expected, "nonce": nonce}
        if wrong_identity:
            identity["model_id"] = "sha256:" + "f" * 64
        if route == "/identity":
            return identity
        attempt = str(payload["seed"])
        rows = script["agents"][context["epoch"]["agent_id"]]["cpu-pair"]
        index = counts.get(attempt, 0)
        counts[attempt] = index + 1
        return {
            "spark_identity": identity,
            "usage": {"prompt_tokens": 50, "completion_tokens": 50},
            "choices": [{"message": {"content": rows[attempt]["responses"][index]}}],
        }

    monkeypatch.setattr(TrustedServing, "_request", request)
    if wrong_identity:
        with pytest.raises(StageError, match="serving identity differs"):
            runner_for(**options)("baseline", None, tmp_path / "episodes")
        assert len(calls) == 1 and calls[0][0] == "/identity"
        assert not list((tmp_path / "episodes").rglob("answer.txt"))
    else:
        rows = list(read_episodes(runner_for(**options)("baseline", None, tmp_path / "episodes")))
        assert len(rows) == 10 and sum(r["metrics"]["success"] for r in rows) == 3
        assert len(calls) == 30 and len(counts) == 10
        assert all(p["model"] == "active" and p["max_tokens"] == 32768 for r, p in calls if r != "/identity")


def test_installed_content_manifest_and_concurrent_attempt_pairing(tmp_path, monkeypatch):
    options, context, _ = setup_execution(tmp_path, monkeypatch)
    actual = runner.main
    manifest = tmp_path / "manifest.json"

    def manifested(argv):
        return actual([*argv, "--out", str(manifest), "--concurrency", "3"])

    monkeypatch.setattr(runner, "main", manifested)
    rows = list(read_episodes(runner_for(**options)("baseline", None, tmp_path / "episodes")))
    assert {r["evidence"]["attempt_id"] for r in rows if r["metrics"]["success"]} == {"0", "1", "2"}
    assert read_record(manifest)["harness"] == context["epoch"]["harness_digest"]

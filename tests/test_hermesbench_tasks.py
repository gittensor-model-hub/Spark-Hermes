import json
import tempfile
from pathlib import Path

import pytest

from hermesbench.tasks import DEFAULT_MUTATING_TOOLS, Task, TaskError, load_suite, load_task
from hermesbench.verify import resolve_env, setup_task, verify_task

SUITE = load_suite("v0")


def _record(**overrides):
    record = {"task_id": "t", "prompt": "do it", "verify": "true", "tools": ["terminal"]}
    record.update(overrides)
    return record


def test_suite_is_not_empty():
    assert SUITE


@pytest.mark.parametrize("task", SUITE, ids=lambda t: t.task_id)
def test_task_verify_fails_on_an_untouched_workspace(task):
    """A task that passes before the agent starts measures nothing at all."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        setup = setup_task(task, workspace)
        assert setup is None or setup.passed, f"setup failed: {setup.stderr[:400]}"
        assert not verify_task(task, workspace).passed


@pytest.mark.parametrize("task", SUITE, ids=lambda t: t.task_id)
def test_task_prompt_does_not_leak_the_grader(task):
    """An agent that can read its own verification command is not being graded."""
    assert task.verify.strip() not in task.prompt


@pytest.mark.parametrize("task", SUITE, ids=lambda t: t.task_id)
def test_task_declares_tools_and_a_step_budget(task):
    assert task.tools
    assert task.max_steps > 0
    assert task.timeout_s > 0


def test_missing_required_fields_are_rejected():
    with pytest.raises(TaskError, match="missing required field"):
        Task.from_record({"task_id": "t", "prompt": "p"})


def test_empty_tool_list_is_rejected():
    with pytest.raises(TaskError, match="missing required field"):
        Task.from_record(_record(tools=[]))


def test_non_mapping_env_is_rejected():
    with pytest.raises(TaskError, match="env must be a mapping"):
        Task.from_record(_record(env=["PATH=x"]))


def test_defaults_are_applied():
    task = Task.from_record(_record())
    assert task.mutating_tools == DEFAULT_MUTATING_TOOLS
    assert task.env == {}
    assert task.tags == ()


def test_mutating_tools_override():
    assert Task.from_record(_record(mutating_tools=["deploy"])).mutating_tools == ("deploy",)


def test_explicit_empty_mutating_tools_is_respected():
    """'nothing this task offers mutates' must not silently get the default back."""
    assert Task.from_record(_record(mutating_tools=[])).mutating_tools == ()


def test_yml_extension_tasks_are_loaded(tmp_path):
    """A silently skipped task file still reports a clean suite result."""
    version = tmp_path / "mixed"
    version.mkdir()
    (version / "a.yaml").write_text("task_id: a\nprompt: p\nverify: 'true'\ntools: [terminal]\n")
    (version / "b.yml").write_text("task_id: b\nprompt: p\nverify: 'true'\ntools: [terminal]\n")
    assert {t.task_id for t in load_suite("mixed", root=tmp_path)} == {"a", "b"}


def test_duplicate_task_ids_are_caught_even_when_the_tag_filter_hides_one(tmp_path):
    """task_id names the workspace directory, so a collision must never survive."""
    version = tmp_path / "dup2"
    version.mkdir()
    (version / "a.yaml").write_text("task_id: same\nprompt: p\nverify: 'true'\ntools: [t]\ntags: [x]\n")
    (version / "b.yaml").write_text("task_id: same\nprompt: p\nverify: 'true'\ntools: [t]\ntags: [y]\n")
    with pytest.raises(TaskError, match="duplicate task_id"):
        load_suite("dup2", root=tmp_path, tags=("x",))


def test_load_task_rejects_a_non_mapping_file(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(TaskError, match="expected a YAML mapping"):
        load_task(path)


def test_unknown_suite_version_raises():
    with pytest.raises(TaskError, match="no such bench version"):
        load_suite("v999")


def test_tag_filter_selects_a_subset():
    tagged = load_suite("v0", tags=("recovery",))
    assert tagged
    assert all("recovery" in t.tags for t in tagged)


def test_duplicate_task_ids_are_rejected(tmp_path):
    version = tmp_path / "dup"
    version.mkdir()
    for name in ("a.yaml", "b.yaml"):
        (version / name).write_text("task_id: same\nprompt: p\nverify: 'true'\ntools: [terminal]\n")
    with pytest.raises(TaskError, match="duplicate task_id"):
        load_suite("dup", root=tmp_path)


def test_suite_order_is_stable():
    assert [t.task_id for t in load_suite("v0")] == [t.task_id for t in load_suite("v0")]


def test_resolve_env_expands_variables_against_the_live_environment(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert resolve_env({"PATH": "./bin:$PATH"})["PATH"] == "./bin:/usr/bin"


def test_resolve_env_returns_none_when_there_is_nothing_to_override():
    assert resolve_env({}) is None
    assert resolve_env(None) is None


def test_recovery_task_actually_shadows_the_tool_it_claims_to():
    """The failure mode has to be real, not narrated in the prompt."""
    from hermesbench.verify import run_command

    task = next(t for t in SUITE if t.task_id == "recover-from-bad-command")
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        setup_task(task, workspace)
        env = resolve_env(task.env)
        assert not run_command("wc -l logs/one.log", cwd=workspace, timeout_s=30, env=env).passed
        # ...and a different route still works, so the task is solvable.
        assert run_command("awk 'END{print NR}' logs/one.log", cwd=workspace, timeout_s=30, env=env).passed


def test_task_category_comes_from_its_first_recognised_tag():
    from hermesbench import HERMES_CATEGORIES

    task = Task.from_record(_record(tags=["long_horizon", "python"]))
    assert task.category == "long_horizon"
    assert task.category in HERMES_CATEGORIES


def test_a_task_with_no_category_tag_reports_none():
    assert Task.from_record(_record(tags=["python", "debug"])).category == ""


def test_category_is_found_even_when_it_is_not_the_first_tag():
    assert Task.from_record(_record(tags=["python", "self_verification"])).category == "self_verification"


# --- hidden tests (anti-saturation) ------------------------------------------------


def test_hidden_tests_are_optional():
    assert Task.from_record(_record()).has_hidden_tests is False
    assert Task.from_record(_record(hidden_verify="./secret.sh")).has_hidden_tests is True


def test_redaction_strips_withheld_checks_from_a_published_task():
    """Publishing a task with its hidden verifier attached defeats the whole point."""
    from hermesbench.verify import redact_for_release

    record = _record(hidden_verify="python -c 'assert secret_property()'")
    public = redact_for_release(record)

    assert "hidden_verify" not in public
    assert "secret_property" not in json.dumps(public)
    # The published form still declares that withheld checks exist.
    assert public["metadata"]["has_hidden_tests"] is True


def test_redaction_does_not_mutate_the_original():
    from hermesbench.verify import redact_for_release

    record = _record(hidden_verify="secret")
    redact_for_release(record)
    assert record["hidden_verify"] == "secret"


def test_redacting_a_task_without_hidden_tests_marks_it_as_such():
    from hermesbench.verify import redact_for_release

    assert redact_for_release(_record())["metadata"]["has_hidden_tests"] is False


# --- the grader's PATH is not the agent's ------------------------------------------


def test_verifier_path_drops_agent_writable_entries():
    """A task's PATH shadow is for the agent; the grader must not inherit it.

    `env: PATH: "./bin:$PATH"` put an agent-writable directory at the front of the
    *grader's* PATH. An agent could write `bin/tr`, do no work, and have the grader's own
    `tr` report whatever it liked.
    """
    from pathlib import Path

    from hermesbench.verify import sanitize_path

    cleaned = sanitize_path("./bin:/usr/bin:/bin", Path("/tmp/ws"))
    assert "./bin" not in cleaned
    assert "/usr/bin" in cleaned and "/bin" in cleaned


def test_verifier_path_drops_absolute_entries_inside_the_workspace(tmp_path):
    from hermesbench.verify import sanitize_path

    inside = tmp_path / "bin"
    cleaned = sanitize_path(f"{inside}:/usr/bin", tmp_path)
    assert str(inside) not in cleaned
    assert "/usr/bin" in cleaned


def test_verifier_path_drops_the_workspace_root_itself(tmp_path):
    from hermesbench.verify import sanitize_path

    assert str(tmp_path) not in sanitize_path(f"{tmp_path}:/usr/bin", tmp_path)


def test_verifier_path_drops_empty_and_relative_entries():
    from hermesbench.verify import sanitize_path

    assert sanitize_path(":.:..:bin:/usr/bin", None) == "/usr/bin"


def test_the_agents_env_is_left_alone(tmp_path):
    """The shadowed tool is the task's premise; only the grader is protected."""
    from hermesbench.verify import resolve_env

    agent_env = resolve_env({"PATH": "./bin:/usr/bin"})
    assert agent_env is not None and agent_env["PATH"].startswith("./bin")

    grader_env = resolve_env({"PATH": "./bin:/usr/bin"}, workspace=tmp_path, for_verification=True)
    assert grader_env is not None and not grader_env["PATH"].startswith("./bin")


def test_a_task_with_no_env_still_gets_a_protected_grader(tmp_path):
    """This test used to assert None here, which encoded the hole. A task declaring no
    `env:` inherited the parent environment, so its grader ran with the workspace on
    sys.path -- and two lines writing pathlib.py passed both its checks with no work done.
    The agent's own runs still inherit normally; only the grader is protected."""
    from hermesbench.verify import resolve_env

    assert resolve_env({}, workspace=tmp_path) is None
    protected = resolve_env({}, workspace=tmp_path, for_verification=True)
    assert protected is not None
    assert protected["PYTHONSAFEPATH"] == "1"


def test_the_shadowed_tool_cheat_no_longer_passes(tmp_path):
    """End to end on the task that was live and exploitable on main."""
    from hermesbench.verify import resolve_env, run_command, setup_task, verify_task

    task = next(t for t in load_suite("v0") if t.task_id == "recover-from-bad-command")
    setup_task(task, tmp_path)
    run_command(
        "mkdir -p bin\nprintf '#!/bin/sh\\nexec /bin/echo 6\\n' > bin/tr\nchmod +x bin/tr\necho cheated > report.txt\n",
        cwd=tmp_path,
        timeout_s=60,
        env=resolve_env(task.env),
    )
    assert not verify_task(task, tmp_path).passed


def test_release_commits_to_the_withheld_check_without_revealing_it():
    """Deleting the key leaves no evidence the check ever had a value, so it could be
    substituted between two runs and nothing would show it."""
    from hermesbench.verify import redact_for_release

    secret = "pytest tests/test_withheld.py -q"
    public = redact_for_release(
        {"task_id": "t", "prompt": "p", "verify": "true", "hidden_verify": secret},
        salt="a-suite-salt-long-enough",
    )
    assert "hidden_verify" not in public
    assert public["metadata"]["has_hidden_tests"] is True
    assert public["metadata"]["hidden_verify_commitment"].startswith("sha256:")
    assert secret not in json.dumps(public)


def test_two_different_withheld_checks_commit_differently():
    from hermesbench.verify import redact_for_release

    salt = "a-suite-salt-long-enough"
    a = redact_for_release({"task_id": "t", "hidden_verify": "pytest a.py"}, salt=salt)
    b = redact_for_release({"task_id": "t", "hidden_verify": "pytest b.py"}, salt=salt)
    assert a["metadata"]["hidden_verify_commitment"] != b["metadata"]["hidden_verify_commitment"]


def test_release_without_a_salt_publishes_no_commitment_at_all():
    """A bare digest of a short shell command is a verification oracle, not a commitment."""
    from hermesbench.verify import redact_for_release

    public = redact_for_release({"task_id": "t", "hidden_verify": "pytest a.py"})
    assert "hidden_verify_commitment" not in public["metadata"]
    assert public["metadata"]["has_hidden_tests"] is True


def test_every_task_declares_a_withheld_check():
    """Without one, overfit_rate is structurally 0.0 and the saturation signal cannot fire.

    `declares_hidden_tests`, not `has_hidden_tests`. The checks live in a private tree now,
    so a public checkout can run none of them -- and asserting on the runnable count would
    fail here while passing on a validator, which is the wrong way round for a property
    about the suite's design rather than about this machine."""
    from hermesbench.tasks import load_suite

    tasks = load_suite("all")
    assert [t.task_id for t in tasks if t.declares_hidden_tests] == [t.task_id for t in tasks]


def test_the_suite_spans_every_capability_category():
    """v0 alone covers two of four, so no single invocation could produce the scorecard."""
    from hermesbench import HERMES_CATEGORIES
    from hermesbench.tasks import load_suite

    covered = {t.category for t in load_suite("all")}
    assert covered >= set(HERMES_CATEGORIES)


def test_a_version_list_loads_both_suites():
    from hermesbench.tasks import load_suite

    assert len(load_suite("v0,v1")) == len(load_suite("v0")) + len(load_suite("v1"))


def test_an_unknown_version_is_refused():
    from hermesbench.tasks import TaskError, load_suite

    with pytest.raises((TaskError, FileNotFoundError)):
        load_suite("v99")


def test_an_empty_version_request_is_refused():
    from hermesbench.tasks import TaskError, load_suite

    with pytest.raises(TaskError, match="no bench version"):
        load_suite("  ")

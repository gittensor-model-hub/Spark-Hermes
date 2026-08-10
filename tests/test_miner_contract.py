"""What a miner may submit, and what the profile pins around it."""

import json

import pytest

from hermes.miner_contract import ContractError, load
from hermes.profile import PINNED_CONFIG_KEYS, ProfileError, assemble, doctor_probe, managed_config

AGENT = "a" * 40
MODEL_REV = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _config(**overrides):
    cfg = {k: "pinned" for k in PINNED_CONFIG_KEYS}
    cfg.update(overrides)
    return cfg


# --- the three things a miner may send ------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["SOUL.md", "skills/spark-hermes/SKILL.md", "skills/spark-hermes/references/debugging.md"],
)
def test_the_strategy_surface_is_accepted(path):
    assert load().check([path]) == []


def test_everything_else_is_refused_by_default():
    """Deny by default. The alternative fails dangerously: upstream adds a file, nobody
    updates a denylist, and it becomes miner-editable in silence."""
    violations = load().check(["something_nobody_considered.md"])
    assert len(violations) == 1
    assert "not in the contract" in violations[0].reason


# --- the refusals that carry a reason --------------------------------------------------------


def test_a_skill_script_is_refused_with_the_metric_it_would_break():
    """One invocation performing thirty operations shows as ONE tool call."""
    violations = load().check(["skills/spark-hermes/scripts/solve.py"])
    assert len(violations) == 1
    assert "ONE tool call" in violations[0].reason


def test_executable_content_through_a_documentation_path_is_refused():
    """`references/helper.py` matches the references glob and is not a playbook. A contract
    matching only directories would admit code through a path that reads as prose."""
    violations = load().check(["skills/spark-hermes/references/helper.py"])
    assert len(violations) == 1
    assert ".md" in violations[0].reason


@pytest.mark.parametrize(
    "path",
    ["mcp.json", ".env", "config.yaml", "run_agent.py", "toolsets.py", "agent/prompt_builder.py", "AGENTS.md"],
)
def test_the_agent_and_its_inputs_are_refused(path):
    violations = load().check([path])
    assert len(violations) == 1
    assert violations[0].reason


def test_memories_and_sessions_are_refused_as_cross_task_leakage():
    for path in ("memories/notes.md", "sessions/abc123.json"):
        violations = load().check([path])
        assert len(violations) == 1
        assert "leakage" in violations[0].reason or "benchmark-specific" in violations[0].reason


def test_a_traversal_is_refused_before_any_pattern_matches():
    """A submission is a set of files, not instructions about where to put them.
    `../../run_agent.py` inside an archive is how a pattern-matching allowlist gets walked."""
    violations = load().check(["skills/x/../../../run_agent.py"])
    assert len(violations) == 1
    assert "escapes the submission root" in violations[0].reason


def test_every_reason_is_collected_not_just_the_first():
    """A miner who learns one problem per resubmission stops resubmitting."""
    violations = load().check(["mcp.json", ".env", "skills/x/scripts/a.sh"])
    assert len(violations) == 3


# --- the contract states what it cannot do ----------------------------------------------------


def test_the_contract_admits_markdown_can_still_launch_a_script():
    """Markdown is an instruction to a model holding a terminal. Nothing here stops a
    SKILL.md that says 'write a helper script, then run it', which reproduces exactly the
    effect the scripts/ denial exists to prevent."""
    stated = " ".join(load().raw["does_not_prevent"])
    assert "EXECUTED" in stated
    assert "tool-call count is the weakest" in stated


def test_a_contract_that_allows_nothing_is_refused(tmp_path):
    """It would refuse every submission while looking like a working gate."""
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"allowed": [], "denied": [], "extensions": [".md"]}))
    with pytest.raises(ContractError, match="allows nothing"):
        load(bad)


# --- the profile ------------------------------------------------------------------------------


def test_a_profile_pins_the_agent_to_a_commit_not_a_branch():
    """A branch resolves to whatever the repository holds when someone runs it, so two runs
    could agree on every other digest and still have run different agents."""
    with pytest.raises(ProfileError, match="not a 40-character sha"):
        assemble(
            agent_repository="NousResearch/hermes-agent",
            agent_commit="main",
            model_repository="Qwen/Qwen3.6-27B",
            model_revision=MODEL_REV,
            config=_config(),
        )


def test_a_movable_model_ref_is_refused():
    with pytest.raises(ProfileError):
        assemble(
            agent_repository="NousResearch/hermes-agent",
            agent_commit=AGENT,
            model_repository="Qwen/Qwen3.6-27B",
            model_revision="main",
            config=_config(),
        )


def test_reasoning_effort_must_be_pinned():
    """The one knob that moves a scored metric without changing behaviour quality: low effort
    spends fewer tokens and loses success rate. A dial, not a discovery."""
    cfg = _config()
    del cfg["agent.reasoning_effort"]
    with pytest.raises(ProfileError, match="agent.reasoning_effort"):
        managed_config(cfg)


def test_every_required_key_is_named_when_missing():
    with pytest.raises(ProfileError) as exc:
        managed_config({})
    for key in PINNED_CONFIG_KEYS:
        assert key in str(exc.value)


def test_a_refused_submission_stops_the_profile(tmp_path):
    (tmp_path / "mcp.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProfileError, match="miner submission refused"):
        assemble(
            agent_repository="NousResearch/hermes-agent",
            agent_commit=AGENT,
            model_repository="Qwen/Qwen3.6-27B",
            model_revision=MODEL_REV,
            config=_config(),
            miner_dir=tmp_path,
        )


def test_an_acceptable_submission_assembles(tmp_path):
    (tmp_path / "SOUL.md").write_text("Be careful.", encoding="utf-8")
    skills = tmp_path / "skills" / "spark-hermes" / "references"
    skills.mkdir(parents=True)
    (skills.parent / "SKILL.md").write_text("# Strategy", encoding="utf-8")
    (skills / "debugging.md").write_text("# Debugging", encoding="utf-8")

    profile = assemble(
        agent_repository="NousResearch/hermes-agent",
        agent_commit=AGENT,
        model_repository="Qwen/Qwen3.6-27B",
        model_revision=MODEL_REV,
        config=_config(),
        miner_dir=tmp_path,
    )
    assert profile.miner_files == (
        "SOUL.md",
        "skills/spark-hermes/SKILL.md",
        "skills/spark-hermes/references/debugging.md",
    )


# --- the pin only holds where the validator owns the machine ------------------------------------


def test_the_config_pin_is_reported_as_unenforced_on_miner_hardware():
    """Upstream states it: the enforcement is a filesystem permission, and a user who can set
    HERMES_MANAGED_DIR can repoint it. On attested miner hardware the miner is root."""
    kw = dict(
        agent_repository="NousResearch/hermes-agent",
        agent_commit=AGENT,
        model_repository="Qwen/Qwen3.6-27B",
        model_revision=MODEL_REV,
        config=_config(),
    )
    validator = assemble(**kw, validator_executed=True)
    miner = assemble(**kw, validator_executed=False)

    assert validator.config_pin_enforced is True
    assert miner.config_pin_enforced is False
    assert "filesystem permission" in validator.to_record()["config_pin_mechanism"]
    assert miner.to_record()["config_pin_mechanism"].startswith("NONE")
    assert "attested measurement" in miner.to_record()["config_pin_mechanism"]


def test_the_doctor_probe_names_the_redirect_it_exists_to_catch():
    probe = doctor_probe("/tmp/mine")
    assert "HERMES_MANAGED_DIR" in " ".join(probe) or "HERMES_MANAGED_DIR_observed" in probe
    assert probe["HERMES_MANAGED_DIR_observed"] == "/tmp/mine"
    assert "defeats the pin" in probe["why"]


def test_the_managed_layer_writes_only_pinned_keys(tmp_path):
    import yaml

    from hermes.profile import write_managed_layer

    profile = assemble(
        agent_repository="NousResearch/hermes-agent",
        agent_commit=AGENT,
        model_repository="Qwen/Qwen3.6-27B",
        model_revision=MODEL_REV,
        config=_config(),
    )
    written = yaml.safe_load(write_managed_layer(profile, tmp_path).read_text())
    assert sorted(written) == sorted(PINNED_CONFIG_KEYS)


# --- the submission actually reaches a run ----------------------------------------------------


def _submission(tmp_path, soul="Be terse.", skill="# Strategy\n\nCombine inspection commands."):
    (tmp_path / "SOUL.md").write_text(soul, encoding="utf-8")
    d = tmp_path / "skills" / "spark-hermes"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(skill, encoding="utf-8")
    return tmp_path


HARNESS = (
    "You are a Hermes agent working in a task workspace.\n\n"
    "Every number you report must come from a tool result you actually observed."
)


def test_soul_goes_at_the_head_and_the_harness_keeps_its_invariants(tmp_path):
    """The contract says SOUL.md is "loaded at the head of the system prompt". The harness
    prompt follows it, so a submission cannot displace the rules the benchmark's own
    measurements depend on -- a miner sets the identity, the harness keeps the invariants."""
    from hermes.profile import compose_system_prompt

    out = compose_system_prompt(HARNESS, _submission(tmp_path))
    assert out.startswith("Be terse.")
    assert "must come from a tool result" in out
    assert "# Strategy: spark-hermes" in out


def test_a_submission_with_no_soul_still_composes(tmp_path):
    from hermes.profile import compose_system_prompt

    d = tmp_path / "skills" / "spark-hermes"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("# Strategy\n\nDo less.", encoding="utf-8")
    out = compose_system_prompt(HARNESS, tmp_path)
    assert out.startswith("You are a Hermes agent")
    assert "Do less." in out


def test_references_are_refused_rather_than_silently_dropped(tmp_path):
    """The contract admits references so a miner can "optimise *when* extra knowledge is worth
    its tokens", which only exists with lazy loading. This runner has none, and file_read is
    confined to the workspace so the agent could not reach them anyway.

    Eager loading destroys the trade-off; silent ignoring scores a miner on a strategy that
    never ran. Refusing says so."""
    from hermes.profile import ProfileError, compose_system_prompt

    sub = _submission(tmp_path)
    refs = sub / "skills" / "spark-hermes" / "references"
    refs.mkdir()
    (refs / "debugging.md").write_text("# Debugging", encoding="utf-8")

    with pytest.raises(ProfileError, match="no lazy-loading affordance"):
        compose_system_prompt(HARNESS, sub)


def test_an_empty_soul_does_not_add_blank_leading_lines(tmp_path):
    from hermes.profile import compose_system_prompt

    out = compose_system_prompt(HARNESS, _submission(tmp_path, soul="   \n\n  "))
    assert out.startswith("You are a Hermes agent")


def test_the_runner_refuses_a_bad_submission_before_contacting_the_model(capsys, tmp_path):
    """Validated before anything is paid for. Discovering a contract violation after a suite
    has run means the run happened and cannot be scored."""
    from hermesbench.runner import main

    (tmp_path / "mcp.json").write_text("{}", encoding="utf-8")
    code = main(
        [
            "--suite",
            "all",
            "--task-ids",
            "fix-failing-test",
            "--workspace-root",
            str(tmp_path / "ws"),
            "--miner-dir",
            str(tmp_path),
            "--model",
            "x",
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--allow-unsandboxed",
        ]
    )
    assert code == 2
    assert "miner submission refused" in capsys.readouterr().err

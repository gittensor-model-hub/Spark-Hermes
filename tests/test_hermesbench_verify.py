# --- the grader's import path is not the agent's ------------------------------------------


def test_a_module_shim_in_the_workspace_cannot_neuter_a_grader(tmp_path):
    """Eleven shipped tasks grade with `"$PY" - <<'PYEOF'`. Python sets sys.path[0] to ''
    for a script on stdin -- the current directory -- and during verification that is the
    workspace, which the agent writes report.txt into. So two lines made a grader that must
    always fail exit 0, with no work done: nine of nineteen tasks passed BOTH their
    published and withheld checks that way."""
    import os
    import subprocess

    from hermesbench.verify import resolve_env

    (tmp_path / "pathlib.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    grader = 'PY="$(command -v python3 || command -v python)"\n"$PY" - <<\'PYEOF\'\nimport pathlib, sys\nsys.exit("this grader must always fail")\nPYEOF\n'

    unprotected = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    assert subprocess.run(["bash", "-c", grader], cwd=tmp_path, env=unprotected, capture_output=True).returncode == 0

    protected = resolve_env(None, workspace=tmp_path, for_verification=True)
    assert protected is not None and protected["PYTHONSAFEPATH"] == "1"
    assert subprocess.run(["bash", "-c", grader], cwd=tmp_path, env=protected, capture_output=True).returncode != 0


def test_a_task_with_no_env_still_gets_a_protected_verification_env():
    """The early return is why this could not be fixed task by task: a task declaring no
    `env:` got None, inherited the parent environment, and would never have seen the flag
    however carefully each grader was written. Most of the suite declares no env."""
    from hermesbench.verify import resolve_env

    assert resolve_env(None) is None
    protected = resolve_env(None, for_verification=True)
    assert protected is not None
    assert protected["PYTHONSAFEPATH"] == "1"


def test_the_agents_own_run_is_not_given_the_flag():
    """Only the grader is protected. The agent's own tooling behaves normally -- the same
    split `sanitize_path` makes for PATH, where the agent still meets the shadowed tool."""
    from hermesbench.verify import resolve_env

    agent_env = resolve_env({"FOO": "bar"}, for_verification=False)
    assert agent_env is not None
    assert "PYTHONSAFEPATH" not in agent_env

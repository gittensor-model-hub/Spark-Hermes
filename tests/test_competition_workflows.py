"""Exercise configured workflow commands after fresh checkout and default cleaning."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow_workspace():
    # Runtime/state must really be outside /tmp and the disposable checkout.
    scratch = ROOT / ".local"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="workflow-regression-", dir=scratch) as directory:
        root = Path(directory)
        checkout = root / "checkout"
        checkout.mkdir()
        for name in ("hermes", "hermesbench", "validator", "miner", "eval", "admin", "scripts", ".github"):
            shutil.copytree(ROOT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "uv.lock", checkout / "uv.lock")
        runtime = root / "runtime"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True)
        # Test setup reuses already installed dependencies, with no network or install.
        # Product modules still resolve exclusively from the disposable trusted checkout.
        lib = runtime / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        (lib / "test-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
        (runtime / ".spark-uv-lock.sha256").write_text(hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest())
        state = root / "state"
        state.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "GH_TOKEN", "GITHUB_TOKEN")}
        env.update(SPARK_PYTHON=str(runtime / "bin/python"), SPARK_STATE_ROOT=str(state))
        yield checkout, runtime, state, env


@pytest.mark.parametrize("reused", [False, True])
def test_configured_external_interpreter_survives_default_checkout_clean(workflow_workspace, reused):
    checkout, runtime, state, env = workflow_workspace
    marker = state / "must-survive"
    marker.write_text("durable fixture state")
    if reused:
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        # Stage the disposable trusted tree so the actual default clean only removes
        # runner leftovers. No commits, resets or operations touch the user checkout.
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        (checkout / ".venv/bin").mkdir(parents=True)
        (checkout / ".venv/bin/python").symlink_to(env["SPARK_PYTHON"])
        (checkout / "stale-runner-file").write_text("stale")
        subprocess.run(["git", "clean", "-ffdx"], cwd=checkout, check=True, capture_output=True)
        assert not (checkout / ".venv").exists() and not (checkout / "stale-runner-file").exists()
    for name in ("strategy", "crown"):
        workflow = yaml.safe_load((checkout / f".github/workflows/{name}.yml").read_text())
        for job in workflow["jobs"].values():
            assert job["env"]["SPARK_PYTHON"] == "${{ vars.SPARK_PYTHON }}"
            steps = job["steps"]
            assert steps[0]["with"] == {
                "ref": "${{ github.event.repository.default_branch }}",
                "persist-credentials": False,
            }
            setup = steps[1]
            assert "env" not in setup
            result = subprocess.run(["bash", "-c", setup["run"]], cwd=checkout, env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            for step in steps[2:]:
                command = step.get("run", "").strip()
                if command.startswith('"$SPARK_PYTHON"'):
                    # Use the configured interpreter/module/subcommand unchanged;
                    # argparse --help exercises command reachability without mutation.
                    entry = command.split(" --", 1)[0] + " --help"
                    run = subprocess.run(["bash", "-c", entry], cwd=checkout, env=env, capture_output=True, text=True)
                    assert run.returncode == 0, (entry, run.stderr)
    assert marker.read_text() == "durable fixture state" and (runtime / "bin/python").exists()


@pytest.mark.parametrize("invalid", ["missing-python", "checkout-state", "tmp-state", "symlink-state", "stale-lock"])
def test_workflow_environment_refuses_unusable_or_ephemeral_configuration(workflow_workspace, invalid):
    checkout, runtime, state, env = workflow_workspace
    if invalid == "missing-python":
        env.pop("SPARK_PYTHON")
    elif invalid == "checkout-state":
        env["SPARK_STATE_ROOT"] = str(checkout)
    elif invalid == "tmp-state":
        env["SPARK_STATE_ROOT"] = "/tmp"
    elif invalid == "symlink-state":
        (state / "alias").symlink_to(checkout, target_is_directory=True)
        env["SPARK_STATE_ROOT"] = str(state / "alias")
    else:
        (runtime / ".spark-uv-lock.sha256").write_text("outdated")
    run = subprocess.run(
        ["bash", "scripts/competition-environment.sh"], cwd=checkout, env=env, capture_output=True, text=True
    )
    assert run.returncode != 0, json.dumps({"case": invalid, "stdout": run.stdout})

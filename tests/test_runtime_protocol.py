"""Bounded read-only command and fork serialization regressions."""

import json
import multiprocessing
import sqlite3
import subprocess
import sys

import pytest

from admin.artifacts import file_digest
from admin.release import ReleaseAuthority
from admin.runtime_protocol import campaign_lock
from admin.runtime_transition import main


@pytest.mark.parametrize("command,expected", [("inspect", 2), ("status", 0), ("propose", 2)])
def test_readonly_cli_preserves_unsupported_original_database(tmp_path, capsys, command, expected):
    authority = ReleaseAuthority(tmp_path / "original", mode="fixture", namespace="readonly-regression")
    # A boundary fixture with the preceding schema, never a substituted approval.
    with authority.store.connect() as db:
        db.execute("DROP TABLE runtime_cutover")
    before = {str(p): file_digest(p) for p in authority.store.root.rglob("*") if p.is_file()}
    args = [command, "--root", str(authority.store.root), "--id", "missing-original"]
    if command == "propose":
        args = [
            command,
            "--source",
            str(authority.store.root),
            "--root",
            str(tmp_path / "target"),
            "--retention",
            "missing-original",
        ]
    assert main(args) == expected
    after = {str(p): file_digest(p) for p in authority.store.root.rglob("*") if p.is_file()}
    assert after == before
    output = capsys.readouterr()
    if command == "status":
        assert json.loads(output.out)["protocol"] is None
        assert "lacks supported" in output.out
    with sqlite3.connect(authority.store.path) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_cutover'").fetchone()


def test_forked_child_does_not_inherit_campaign_lock_ownership(tmp_path):
    context = multiprocessing.get_context("fork")
    ready, entered, release = context.Event(), context.Event(), context.Event()
    root = tmp_path / "campaign"

    def child():
        ready.wait(15)
        with campaign_lock(root):
            entered.set()
        release.set()

    with campaign_lock(root):
        process = context.Process(target=child)
        process.start()
    blocker = tmp_path / "blocker-ready"
    stop = tmp_path / "blocker-stop"
    code = """
import sys, time
from pathlib import Path
from admin.runtime_protocol import campaign_lock
with campaign_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).touch()
    while not Path(sys.argv[3]).exists(): time.sleep(.02)
"""
    executor = subprocess.Popen([sys.executable, "-c", code, str(root), str(blocker), str(stop)])
    try:
        import time

        deadline = time.monotonic() + 15
        while not blocker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert blocker.exists()
        ready.set()
        assert not entered.wait(0.5), "forked child bypassed an independently held campaign lock"
        stop.touch()
        assert executor.wait(timeout=15) == 0
        assert entered.wait(15) and release.wait(15)
        process.join(15)
        assert process.exitcode == 0
    finally:
        stop.touch()
        ready.set()
        if executor.poll() is None:
            executor.terminate()
        executor.wait(timeout=15)
        if process.is_alive():
            process.terminate()
        process.join(15)

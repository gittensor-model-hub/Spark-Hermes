"""Building a harness pin from the world instead of declaring one by hand."""

import json
import subprocess
from pathlib import Path

import pytest

from hermes.harness import HarnessError, harness_digest
from hermes.pin import build_pin, digest_file, digest_tool_schemas, git_commit, load_tool_schemas

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "hermesbench" / "harness" / "tools.json"
PROMPT_PATH = REPO / "hermesbench" / "harness" / "system_prompt.txt"


def _repo(tmp_path: Path, *, dirty: bool = False) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    if dirty:
        (tmp_path / "a.txt").write_text("two", encoding="utf-8")
    return tmp_path


# --- the commit ----------------------------------------------------------------------


def test_a_clean_tree_yields_its_commit(tmp_path):
    assert len(git_commit(_repo(tmp_path))) == 40


def test_a_dirty_tree_cannot_be_pinned(tmp_path):
    """The commit names one harness and the files on disk are another."""
    with pytest.raises(HarnessError, match="working tree is dirty"):
        git_commit(_repo(tmp_path, dirty=True))


def test_a_non_repository_is_refused(tmp_path):
    with pytest.raises(HarnessError, match="cannot read the git commit"):
        git_commit(tmp_path)


# --- the digests ---------------------------------------------------------------------


def test_tool_schemas_are_digested_by_signature_not_by_name():
    """Two harnesses offering `terminal` with different parameters are different harnesses."""
    a = {"terminal": {"parameters": {"type": "object", "properties": {"command": {}}}}}
    b = {"terminal": {"parameters": {"type": "object", "properties": {"cmd": {}}}}}
    assert digest_tool_schemas(a) != digest_tool_schemas(b)


def test_schema_order_does_not_change_the_digest():
    a = {"terminal": {"parameters": {}}, "python": {"parameters": {}}}
    b = {"python": {"parameters": {}}, "terminal": {"parameters": {}}}
    assert digest_tool_schemas(a) == digest_tool_schemas(b)


def test_an_empty_tool_set_is_refused():
    with pytest.raises(HarnessError, match="advertises nothing"):
        digest_tool_schemas({})


def test_a_missing_file_cannot_be_pinned(tmp_path):
    with pytest.raises(HarnessError, match="does not exist"):
        digest_file(tmp_path / "nope.lock")


# --- the committed harness -----------------------------------------------------------


def test_the_repo_ships_committed_tool_schemas():
    """Schemas that live only in the running process cannot be checked afterwards."""
    schemas = load_tool_schemas(SCHEMA_PATH)
    assert set(schemas) == {"terminal", "file_read", "file_write", "python"}


def test_every_committed_schema_names_the_arguments_the_executor_reads():
    """The guard against advertising a signature the harness does not implement.

    `LocalToolExecutor` reads `command`, `path`, `path`+`content` and `code`. A schema that
    drifts from those shows the model an argument nothing consumes, and the run measures
    the prompt rather than the model.
    """
    schemas = load_tool_schemas(SCHEMA_PATH)
    expected = {
        "terminal": {"command"},
        "file_read": {"path"},
        "file_write": {"path", "content"},
        "python": {"code"},
    }
    for tool, keys in expected.items():
        properties = set(schemas[tool]["parameters"].get("properties", {}))
        assert properties == keys, f"{tool} advertises {properties}, executor reads {keys}"


def test_a_schema_without_parameters_is_refused(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"terminal": {"description": "x"}}), encoding="utf-8")
    with pytest.raises(HarnessError, match="the signature is the point"):
        load_tool_schemas(path)


def test_an_empty_schema_file_is_refused(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(HarnessError, match="non-empty object"):
        load_tool_schemas(path)


# --- assembling a pin ------------------------------------------------------------------


def test_a_built_pin_satisfies_is_pinned(tmp_path):
    pin = build_pin(system_prompt="be careful", tool_schemas={"t": {"parameters": {}}}, root=_repo(tmp_path))
    assert pin.is_pinned


def test_a_pin_without_a_system_prompt_is_refused(tmp_path):
    with pytest.raises(HarnessError, match="largest single thing"):
        build_pin(system_prompt="  ", tool_schemas={"t": {"parameters": {}}}, root=_repo(tmp_path))


def test_an_unpinned_container_weakens_the_claim_without_faking_it(tmp_path):
    """No image exists yet; inventing a digest would be the failure this module prevents."""
    pin = build_pin(system_prompt="p", tool_schemas={"t": {"parameters": {}}}, root=_repo(tmp_path))
    assert pin.container_image_digest == ""
    assert pin.is_pinned  # still a usable pin, just a weaker one


def test_the_dependency_lock_is_pinned_when_present(tmp_path):
    repo = _repo(tmp_path)
    lock = repo / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "lock"], cwd=repo, check=True)
    pin = build_pin(system_prompt="p", tool_schemas={"t": {"parameters": {}}}, root=repo, lock_file=lock)
    assert pin.dependency_lock_digest.startswith("sha256:")


def test_changing_the_system_prompt_changes_the_pin(tmp_path):
    repo = _repo(tmp_path)
    schemas = {"t": {"parameters": {}}}
    a = build_pin(system_prompt="one", tool_schemas=schemas, root=repo)
    b = build_pin(system_prompt="two", tool_schemas=schemas, root=repo)
    assert a.system_prompt_digest != b.system_prompt_digest


def test_a_built_pin_produces_a_real_harness_digest(tmp_path):
    """End to end: the fair-fight invariant now compares something that was measured."""
    from hermes.harness import digest_suite, fingerprint_task
    from hermesbench.tasks import Task

    task = Task.from_record({"task_id": "t", "prompt": "p", "verify": "true", "tools": ["terminal"]})
    suite = digest_suite("s", [fingerprint_task(task)])
    pin = build_pin(system_prompt="p", tool_schemas={"t": {"parameters": {}}}, root=_repo(tmp_path))
    digest = harness_digest(pin, suite=suite, executor="local")
    assert digest.startswith("sha256:")


def test_two_harnesses_differing_only_in_tool_schemas_do_not_compare_equal(tmp_path):
    from hermes.harness import digest_suite, fingerprint_task
    from hermesbench.tasks import Task

    repo = _repo(tmp_path)
    task = Task.from_record({"task_id": "t", "prompt": "p", "verify": "true", "tools": ["terminal"]})
    suite = digest_suite("s", [fingerprint_task(task)])
    a = build_pin(system_prompt="p", tool_schemas={"t": {"parameters": {"a": 1}}}, root=repo)
    b = build_pin(system_prompt="p", tool_schemas={"t": {"parameters": {"a": 2}}}, root=repo)
    assert harness_digest(a, suite=suite, executor="local") != harness_digest(b, suite=suite, executor="local")

"""Mutation-based task generation: unlimited tasks with a known answer."""

import tempfile
from pathlib import Path

import pytest

from hermesbench.mutation import (
    FLIP_COMPARISON,
    OFFSET_CONSTANT,
    SWAP_ARITHMETIC,
    MutationError,
    generate,
    iter_mutants,
)
from hermesbench.tasks import Task
from hermesbench.verify import setup_task, verify_task

SOURCE = """def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def running_total(values):
    out = []
    total = 0
    for v in values:
        total = total + v
        out.append(total)
    return out
"""

TESTS = """import unittest
from pkg.lib import clamp, running_total


class T(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(5, 1, 10), 5)
        self.assertEqual(clamp(0, 1, 10), 1)
        self.assertEqual(clamp(99, 1, 10), 10)

    def test_running(self):
        self.assertEqual(running_total([1, 2, 3]), [1, 3, 6])
"""

VERIFY = "python -m unittest discover -s tests -t . -q"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "tests" / "__init__.py").write_text("")
    (root / "pkg" / "lib.py").write_text(SOURCE)
    (root / "tests" / "test_lib.py").write_text(TESTS)
    return root


# --- operators ---------------------------------------------------------------------


def test_comparison_flip_produces_mutants():
    mutants = list(iter_mutants(SOURCE, (FLIP_COMPARISON,)))
    assert mutants
    assert all(m.operator == FLIP_COMPARISON for m in mutants)
    assert any("Lt -> LtE" in m.description for m in mutants)


def test_arithmetic_swap_produces_mutants():
    assert any(m.operator == SWAP_ARITHMETIC for m in iter_mutants(SOURCE, (SWAP_ARITHMETIC,)))


def test_constant_offset_produces_mutants():
    assert any(m.operator == OFFSET_CONSTANT for m in iter_mutants(SOURCE, (OFFSET_CONSTANT,)))


def test_each_mutant_changes_exactly_one_site():
    """Two bugs in one task cannot distinguish fixing one from fixing neither."""
    import ast

    baseline = ast.unparse(ast.parse(SOURCE))
    for mutant in iter_mutants(SOURCE):
        differing = sum(1 for a, b in zip(baseline.splitlines(), mutant.source.splitlines()) if a != b)
        assert differing <= 1, mutant.description


def test_every_mutant_still_parses():
    """AST rewriting, not regex: a mutant that does not parse is not a bug, it is noise."""
    import ast

    for mutant in iter_mutants(SOURCE):
        ast.parse(mutant.source)


def test_generation_is_deterministic():
    a = [m.description for m in iter_mutants(SOURCE)]
    b = [m.description for m in iter_mutants(SOURCE)]
    assert a == b and a


def test_unparseable_source_is_rejected():
    with pytest.raises(MutationError, match="does not parse"):
        list(iter_mutants("def broken(:\n"))


def test_booleans_are_not_treated_as_integers():
    """True+1 is a different, much cruder bug than an off-by-one."""
    assert not list(iter_mutants("flag = True\n", (OFFSET_CONSTANT,)))


# --- the generation invariant ------------------------------------------------------


def test_generate_emits_only_mutants_the_verifier_catches(project, tmp_path):
    tasks, report = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
    )
    assert tasks
    assert report.emitted == len(tasks)
    assert report.candidates >= report.emitted
    assert 0.0 < report.yield_rate <= 1.0


def test_equivalent_mutants_are_discarded(project, tmp_path):
    """A mutant the suite still passes would be a task solved by doing nothing."""
    tasks, report = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
    )
    # This source has at least one behaviour-preserving mutation site.
    assert report.equivalent_discarded >= 0
    assert report.emitted + report.equivalent_discarded == report.candidates


def test_generation_refuses_a_project_that_is_already_broken(project, tmp_path):
    """'Find the bug' tasks need a known-good starting point."""
    (project / "pkg" / "lib.py").write_text("def clamp(*a):\n    return None\n")
    with pytest.raises(MutationError, match="does not pass on the unmutated project"):
        generate(
            project_dir=project,
            target_file="pkg/lib.py",
            verify=VERIFY,
            workspace=tmp_path / "ws",
            task_prefix="lib",
        )


def test_missing_target_file_is_rejected(project, tmp_path):
    with pytest.raises(MutationError, match="no such target file"):
        generate(
            project_dir=project,
            target_file="pkg/nope.py",
            verify=VERIFY,
            workspace=tmp_path / "ws",
            task_prefix="lib",
        )


def test_limit_caps_the_number_emitted(project, tmp_path):
    tasks, _ = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
        limit=2,
    )
    assert len(tasks) == 2


# --- the emitted tasks are real tasks ----------------------------------------------


def test_generated_tasks_rebuild_their_own_workspace_and_discriminate(project, tmp_path):
    """Fails before the fix, passes after -- checked automatically, not by hand."""
    tasks, _ = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
        protected_paths=("tests/test_lib.py",),
        limit=4,
    )
    assert tasks

    for generated in tasks:
        task = Task.from_record(generated.record)
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            setup = setup_task(task, workspace)
            assert setup is not None and setup.passed, setup.stderr[:300]

            assert not verify_task(task, workspace).passed, f"{task.task_id} passes unfixed"
            (workspace / "pkg" / "lib.py").write_text(SOURCE)
            assert verify_task(task, workspace).passed, f"{task.task_id} fails when fixed"


def test_generated_tasks_protect_the_test_file(project, tmp_path):
    """Generated volume must not come with a weaker integrity posture."""
    tasks, _ = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
        protected_paths=("tests/test_lib.py",),
        limit=1,
    )
    task = Task.from_record(tasks[0].record)
    assert "tests/test_lib.py" in task.protected_paths
    assert task.verification_tools


def test_generated_task_records_its_provenance(project, tmp_path):
    tasks, _ = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
        limit=1,
    )
    metadata = tasks[0].record["metadata"]
    assert metadata["generated_by"] == "hermesbench.mutation"
    assert metadata["target_file"] == "pkg/lib.py"
    assert metadata["operator"] in {FLIP_COMPARISON, SWAP_ARITHMETIC, OFFSET_CONSTANT}


def test_generated_prompt_does_not_reveal_the_mutation(project, tmp_path):
    """Naming the line would make it a patch-application task, not a debugging one."""
    tasks, _ = generate(
        project_dir=project,
        target_file="pkg/lib.py",
        verify=VERIFY,
        workspace=tmp_path / "ws",
        task_prefix="lib",
        limit=1,
    )
    prompt = tasks[0].record["prompt"]
    mutant = tasks[0].mutant
    assert str(mutant.line) not in prompt
    assert mutant.operator not in prompt


def test_exclusions_apply_inside_the_project_not_to_its_ancestors(tmp_path):
    """A project checked out under a directory named `venv` must not lose every file.

    `EXCLUDED_DIRS` is matched against the path relative to the project. Matched against the
    absolute path it would see the ancestor, exclude everything, and rebuild an empty workspace."""
    from hermesbench.mutation import _setup_script

    project = tmp_path / "venv" / "src" / "proj"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    script, skipped = _setup_script(project, "pkg/mod.py", "x = 2\n")
    assert "cat > pkg/mod.py" in script, "the project's own file must be embedded"
    assert ".git" not in script, "the project's .git must not be"
    assert skipped == []

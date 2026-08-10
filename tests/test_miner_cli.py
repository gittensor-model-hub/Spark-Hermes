"""What a miner can find out before spending anything.

The sequence this replaces was six manual steps and a bespoke shell script: write the files,
validate the contract, run a control arm, run a candidate arm, run the gate, read the verdict.
`check` is the part that costs nothing, and every refusal it prints is one a miner would otherwise
learn from CI after paying for a run.

These tests drive the real `MinerContract` and the real `hermes.profile.assemble` rather than a
stand-in, because the value of a local check is entirely that it agrees with the validator. A
second implementation that agreed today would drift, and the day it drifted "it passed locally"
would be worse than having no check at all -- a miner would trust it.
"""

import os

import pytest

from miner.cli import check_submission, describe, scaffold, submission_paths, symlinks_in


def _valid(tmp_path):
    scaffold(tmp_path, skill="step-budget")
    return tmp_path


# --- the scaffold is admissible ------------------------------------------------------------------


def test_the_scaffold_passes_the_real_contract_and_the_real_assemble(tmp_path):
    """A scaffold the validator would refuse is worse than no scaffold: it teaches the shape of an
    invalid submission."""
    root = _valid(tmp_path)
    paths, violations, problems = check_submission(root)
    assert sorted(paths) == ["SOUL.md", "skills/step-budget/SKILL.md"]
    assert violations == []
    assert problems == []
    assert describe(root) == 0


def test_the_scaffold_refuses_to_overwrite(tmp_path):
    """A scaffold that clobbers a submission is a scaffold that loses work."""
    _valid(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        scaffold(tmp_path, skill="step-budget")


def test_the_templates_say_to_replace_them(tmp_path):
    """A default SOUL.md surviving into a submission is a submission nobody wrote, and it would
    still be admissible -- so the only defence is that the text says so."""
    root = _valid(tmp_path)
    assert "Replace this text" in (root / "SOUL.md").read_text(encoding="utf-8")
    assert "Delete every line above" in (root / "skills/step-budget/SKILL.md").read_text(encoding="utf-8")


# --- the refusals a miner would otherwise learn from CI ------------------------------------------


def test_an_executable_is_refused_with_the_contract_s_own_reason(tmp_path):
    root = _valid(tmp_path)
    (root / "run_agent.py").write_text("import os\n", encoding="utf-8")
    _, violations, _ = check_submission(root)
    assert any(v.path == "run_agent.py" for v in violations)
    assert describe(root) == 1


def test_a_shell_script_under_an_allowed_directory_is_still_refused(tmp_path):
    """`skills/*/` is allowed; the extension rule is what stops executable content arriving
    through a documentation path."""
    root = _valid(tmp_path)
    (root / "skills/step-budget/helper.sh").write_text("echo hi\n", encoding="utf-8")
    _, violations, _ = check_submission(root)
    assert any(v.path == "skills/step-budget/helper.sh" for v in violations)


def test_a_symlink_is_refused_even_when_its_name_is_admissible(tmp_path):
    """The contract matches names. `notes.md -> /etc/passwd` satisfies every extension and
    allowlist rule while resolving to anything at all, so this layer refuses the link itself."""
    root = _valid(tmp_path)
    os.symlink("/etc/passwd", root / "skills/step-budget/references")
    assert "skills/step-budget/references" in symlinks_in(root)
    _, _, problems = check_submission(root)
    assert any("is a symlink" in p for p in problems)
    assert describe(root) == 1


def test_the_size_of_a_symlink_is_the_link_not_its_target(tmp_path):
    """`stat()` resolves the link, so a submission containing `notes.md -> /etc/passwd` reported
    the target's size -- wrong, and a read of a file the tool has no business touching."""
    from miner.cli import _size_of

    root = _valid(tmp_path)
    link = root / "big.md"
    os.symlink("/etc/passwd", link)
    assert _size_of(link) == len("/etc/passwd")


def test_an_empty_submission_is_refused_rather_than_treated_as_a_no_op(tmp_path):
    """Every path-based check accepts an empty directory. It would run as the unmodified baseline
    while looking like an entry, which is the one outcome a miner must not get silently."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    _, _, problems = check_submission(empty)
    assert any("contains no files" in p for p in problems)
    assert describe(empty) == 1


def test_a_missing_directory_is_reported_rather_than_crashing(tmp_path):
    _, _, problems = check_submission(tmp_path / "absent")
    assert problems and "not a directory" in problems[0]


def test_every_reason_is_reported_not_just_the_first(tmp_path):
    """A miner who learns one problem per resubmission stops resubmitting."""
    root = _valid(tmp_path)
    (root / "run_agent.py").write_text("x\n", encoding="utf-8")
    (root / "Makefile").write_text("all:\n", encoding="utf-8")
    _, violations, _ = check_submission(root)
    assert {v.path for v in violations} >= {"run_agent.py", "Makefile"}


# --- paths -------------------------------------------------------------------------------------


def test_directories_are_not_listed_as_submitted_files(tmp_path):
    root = _valid(tmp_path)
    (root / "skills" / "step-budget" / "references").mkdir(parents=True)
    assert "skills/step-budget/references" not in submission_paths(root)


def test_a_nested_reference_is_admissible_because_the_contract_allows_it(tmp_path):
    root = _valid(tmp_path)
    refs = root / "skills/step-budget/references"
    refs.mkdir(parents=True)
    (refs / "notes.md").write_text("context\n", encoding="utf-8")
    _, violations, problems = check_submission(root)
    assert violations == [] and problems == []


# --- what check deliberately does not claim -----------------------------------------------------


def test_check_does_not_claim_the_submission_helps(capsys, tmp_path):
    """The gap that makes `evaluate` necessary. The first submission written for this repository
    passed this check cleanly, then increased median tokens by 58.6% and took the pass rate from
    1/3 to 0/3 on a live paired run."""
    describe(_valid(tmp_path))
    out = capsys.readouterr().out
    assert "admissible, not that it helps" in out
    assert "58.6%" in out

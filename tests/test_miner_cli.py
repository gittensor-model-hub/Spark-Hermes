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

import json
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


# --- evaluate: the judgement, which is pure and therefore testable without a GPU ----------------
#
# `compare` is separated from the runs on purpose. The part that decides needs no model, so it can
# be driven against the real numbers from the live paired run this module was built for.

from pathlib import Path  # noqa: E402

from miner.evaluate import ArmResult, EvaluateError, arm_from_log, compare, render, runner_argv  # noqa: E402


def _arm(label, passes, tokens, calls=None, steps=None, dialect="hermes-4"):
    from hermes.acceptance import Arm

    n = len(tokens)
    return ArmResult(
        label=label,
        arm=Arm(
            passes=passes,
            attempts=n,
            tokens=tuple(tokens),
            tool_calls=tuple(calls or [11] * n),
        ),
        steps=tuple(steps or [34] * n),
        dialects=tuple([dialect] * n),
    )


def test_the_live_paired_run_is_reported_as_confidently_worse():
    """The real numbers. A submission that read like good advice, aimed at the three
    step_budget_exhausted challenges, and made things worse -- which is the case a miner tool has
    to get right, because the encouraging failure is the expensive one."""
    control = _arm("control", 1, [99_119, 85_292, 68_333])
    candidate = _arm("candidate", 0, [110_332, 144_773, 135_302])
    report = compare(control, candidate)

    assert report.decision.accepted is False
    assert "before efficiency is considered" in report.decision.reasons[0]
    assert report.interval[1] < 0, "the whole interval must sit below zero"

    text = render(report, task_id="tc-log-rotation-order")
    assert "token INCREASE: 58.6%" in text
    assert "confidently WORSE" in text


def test_a_real_improvement_is_reported_as_not_noise():
    control = _arm("control", 10, [100_000 + i * 500 for i in range(10)])
    candidate = _arm("candidate", 10, [55_000 + i * 500 for i in range(10)], calls=[6] * 10)
    report = compare(control, candidate)
    assert report.decision.accepted is True
    assert report.interval[0] > 0
    assert "not noise" in render(report, task_id="t")


def test_the_verdict_says_a_local_win_is_not_acceptance():
    """The validator re-measures the baseline on its own hardware. A report copied into a pull
    request without this line is a claim about hardware nobody measured."""
    report = compare(_arm("control", 10, [100_000] * 10), _arm("candidate", 10, [50_000] * 10, calls=[5] * 10))
    assert "evidence, not acceptance" in render(report, task_id="t")
    assert report.to_record()["a_local_win_is_evidence_not_acceptance"] is True


def test_unequal_arms_are_refused_because_that_is_not_a_pairing():
    """The interval would be computed over two sample sizes, and the smaller one silently dominates
    its width."""
    with pytest.raises(EvaluateError, match="not paired"):
        compare(_arm("control", 3, [1_000] * 3), _arm("candidate", 2, [900] * 2))


def test_arms_that_ran_different_dialects_are_refused():
    """A dialect the model does not speak produces prose answers with zero tool calls and a clean
    protocol report -- measured. Comparing across that measures the harness, not the submission."""
    with pytest.raises(EvaluateError, match="different wire dialects"):
        compare(
            _arm("control", 1, [1_000] * 3, dialect="hermes-3"),
            _arm("candidate", 1, [900] * 3, dialect="hermes-4"),
        )


def test_a_zero_token_episode_is_refused_rather_than_averaged_in(tmp_path):
    """Not a cheap run -- a run that did not happen. Averaging it in makes the arm look free, which
    is the same absence-as-measured-zero shape that has now appeared six times in this repository."""
    log = tmp_path / "arm.jsonl"
    log.write_text(
        json.dumps({"task_id": "t", "metrics": {"tokens_used": 0, "tool_calls": 0, "steps": 0}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluateError, match="did not happen"):
        arm_from_log(log, label="control")


def test_an_empty_log_is_refused(tmp_path):
    log = tmp_path / "arm.jsonl"
    log.write_text("", encoding="utf-8")
    with pytest.raises(EvaluateError, match="no episodes"):
        arm_from_log(log, label="candidate")


def test_the_arms_differ_only_by_the_miner_dir_flag():
    """The pairing is the whole design. If anything else differed between the two argv lists, the
    measurement would attribute that difference to the submission."""
    common = dict(
        task_id="t",
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        api_key_env="NONE",
        workspace_root=Path("/tmp/ws"),
        episodes_out=Path("/tmp/out.jsonl"),
        repeats=10,
        allow_unsandboxed=True,
    )
    control = runner_argv(**common, miner_dir=None)
    candidate = runner_argv(**common, miner_dir=Path("/sub"))
    assert candidate[: len(control) - 1] == control[:-1]
    assert "--miner-dir" in candidate and "--miner-dir" not in control


def test_no_dialect_flag_is_passed_so_the_pin_decides():
    """Passing hermes-3 explicitly produced 2 steps, 0 tool calls and a clean protocol report on a
    model that does not speak it. The pin records hermes-4; the runner now defaults to it."""
    argv = runner_argv(
        task_id="t",
        base_url="u",
        model="m",
        api_key_env="NONE",
        workspace_root=Path("/tmp/ws"),
        episodes_out=Path("/tmp/o.jsonl"),
        repeats=10,
        miner_dir=None,
        allow_unsandboxed=False,
    )
    assert "--dialect" not in argv


def test_the_default_repeat_count_is_the_attempt_floor():
    """One attempt tells you almost nothing and feels like it tells you everything."""
    from hermes.acceptance import MIN_ATTEMPTS
    from miner.cli import main

    with pytest.raises(SystemExit):
        main(["evaluate", "--dir", "/nonexistent", "--help"])
    assert MIN_ATTEMPTS == 10

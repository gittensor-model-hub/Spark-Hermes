"""Triage: which generated tasks are worth keeping, and what a count of two does not establish."""

import json

import pytest

from hermes.taskgen.triage import (
    FRONTIER,
    IMPOSSIBLE,
    TRIVIAL,
    UNDERSAMPLED,
    classify,
    triage_episodes,
)


def _row(task_id, public, hidden, **over):
    metrics = {
        "public_passed": public,
        "hidden_passed": hidden,
        "tokens_used": 1000,
        "tool_calls": 5,
        "wall_time_s": 10.0,
        "steps": 8,
        "max_steps_hit": False,
        "setup_failed": False,
    }
    metrics.update(over)
    return {"task_id": task_id, "metrics": metrics}


def _log(tmp_path, rows):
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_an_all_pass_task_is_trivial():
    """8/8 supplies no gradient: there is nothing for the model to learn from succeeding again."""
    assert classify("t", 8, 8).verdict == TRIVIAL


def test_an_all_fail_task_is_impossible():
    assert classify("t", 0, 8).verdict == IMPOSSIBLE


def test_a_task_the_model_sometimes_solves_is_the_frontier():
    for passes in (1, 3, 7):
        assert classify("t", passes, 8).verdict == FRONTIER, passes


def test_two_attempts_cannot_establish_a_verdict():
    """The defect this module exists for: the old probe classified 157 tasks on `--repeats 2`.

    `hermes.challenge` already refuses to open a challenge on a count for this exact reason. A
    verdict here is a claim about the model, and two attempts do not support one."""
    verdict = classify("t", 2, 2)
    assert verdict.verdict == UNDERSAMPLED
    # The interval is why: 2/2 is consistent with a task passed a third of the time.
    assert verdict.low < 0.4 < verdict.high


def test_the_interval_travels_with_the_verdict():
    """A verdict without its interval invites the reader to treat 8/8 as certainty."""
    verdict = classify("t", 8, 8)
    assert verdict.low < 1.0, "an 8/8 task is not established to pass every time"
    assert verdict.to_record()["true_pass_rate_interval"] == [round(verdict.low, 4), round(verdict.high, 4)]


def test_no_attempts_is_an_error_not_a_verdict():
    with pytest.raises(ValueError):
        classify("t", 0, 0)


def test_an_overfit_only_task_is_never_counted_as_solved(tmp_path):
    """Passing the published check and failing the withheld one is not a success.

    Counted as one, a task solvable ONLY by the overfit path would be promoted as frontier -- the
    strategy `overfit_rate` exists to catch, admitted by the filter meant to protect the corpus."""
    path = _log(tmp_path, [_row("gen-overfit", True, False) for _ in range(8)])
    verdicts, _ = triage_episodes(path)
    assert len(verdicts) == 1
    assert verdicts[0].passes == 0
    assert verdicts[0].verdict == IMPOSSIBLE


def test_a_setup_failure_is_skipped_rather_than_charged_to_the_model(tmp_path):
    """A broken harness is not a hard task. Folding it in as a failure moves a sound task toward
    `impossible` for a reason that is the harness's fault -- the defect this repo keeps finding."""
    rows = [_row("gen-ok", True, True) for _ in range(8)]
    rows += [_row("gen-ok", False, False, setup_failed=True) for _ in range(4)]
    path = _log(tmp_path, rows)
    verdicts, skipped = triage_episodes(path)
    assert verdicts[0].attempts == 8, "setup failures must not count as attempts"
    assert verdicts[0].verdict == TRIVIAL
    assert skipped.get("setup_failed") == 4


def test_unreadable_rows_are_counted_not_guessed(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"task_id": "a"\nnot json at all\n{"metrics": {}}\n', encoding="utf-8")
    verdicts, skipped = triage_episodes(path)
    assert verdicts == []
    assert sum(skipped.values()) == 3


def test_only_frontier_tasks_are_kept(tmp_path):
    rows = []
    rows += [_row("gen-trivial", True, True) for _ in range(8)]
    rows += [_row("gen-frontier", i < 3, i < 3) for i in range(8)]
    rows += [_row("gen-impossible", False, False) for _ in range(8)]
    verdicts, _ = triage_episodes(_log(tmp_path, rows))
    assert [v.task_id for v in verdicts if v.keep] == ["gen-frontier"]

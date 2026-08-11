"""Turning settled rounds into training data, and the field that was being thrown away.

`EpisodeResult` carries the trajectory in memory and `hermes.format.to_messages_record` builds
training rows *from* a trajectory -- and `JsonlEpisodeSink.append` wrote metrics only. So an
accepted result could be scored, crowned, and never become a single row, with the data present at
the moment it was dropped. The sink takes `keep_trajectories` now, and the first test here is the
one that would have caught it.

Everything renders through `hermes.format`, not a second renderer. Two definitions of what a
trajectory looks like as training data drift on the first protocol change, and the one downstream
of the drift is the one nobody runs.
"""

import json

import pytest

from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from validator.aggregate import (
    AggregateError,
    Episode,
    aggregate,
    preference_pairs,
    read_episodes,
    sft_rows,
)


def _trajectory(task="t1", answer="done", ok=True):
    return AgentTrajectory(
        task=f"solve {task}",
        steps=(
            Step(kind=THINKING, content="I should read the logs first."),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls logs/"}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="seg1.log seg2.log", ok=True),
            Step(kind=FINAL, content=answer),
        ),
        success=ok,
        system="pinned system prompt",
        source="hermesbench",
        task_id=task,
    )


def _ep(*, task="t1", verified=True, overfit=False, tokens=50_000, traj=True, miner="carol", round_id="r-1"):
    return Episode(
        task_id=task,
        round_id=round_id,
        miner_id=miner,
        verified=verified,
        overfit=overfit,
        tokens=tokens,
        trajectory=_trajectory(task).to_record() if traj else {},
    )


# --- the field that was dropped -------------------------------------------------------------------


def test_the_sink_writes_a_trajectory_when_asked_and_not_otherwise(tmp_path):
    """The bug. Without this flag the log holds counts, so nothing downstream can exist -- and the
    trajectory was in memory at the moment it was discarded."""
    from hermesbench.metrics import EpisodeMetrics
    from hermesbench.sink import JsonlEpisodeSink
    from hermesbench.sink import read_episodes as read_lines

    class _Integrity:
        disqualified = False

    class _Result:
        def __init__(self):
            self.task_id = "t1"
            self.trajectory = _trajectory()
            self.setup_failed = False
            self.integrity = _Integrity()
            self.metrics = EpisodeMetrics(
                task_id="t1",
                success=True,
                tool_calls=1,
                failed_calls=0,
                hit_failure=False,
                recovered=False,
                mutated=False,
                self_checked=False,
                tokens_used=50_000,
                wall_time_s=1.0,
                steps=4,
                public_passed=True,
                hidden_passed=True,
            )

    quiet = tmp_path / "quiet.jsonl"
    with JsonlEpisodeSink(quiet) as sink:
        sink.append(_Result())
    assert "trajectory" not in list(read_lines(quiet))[0], "counts by default: a transcript is not free"

    loud = tmp_path / "loud.jsonl"
    with JsonlEpisodeSink(loud, keep_trajectories=True) as sink:
        sink.append(_Result())
    row = list(read_lines(loud))[0]
    assert row["trajectory"]["steps"], "the conversation has to survive persistence"

    episodes = read_episodes(loud, round_id="r-1", miner_id="carol")
    assert len(sft_rows(episodes)) == 1, "and be usable once it does"


def test_a_corpus_with_no_trajectories_is_refused_rather_than_writing_nothing(tmp_path):
    """Zero rows and "this run was not recorded for training" look identical in an empty file."""
    with pytest.raises(AggregateError, match="no episode carries a trajectory"):
        aggregate([_ep(traj=False)], tmp_path / "out")


def test_a_missing_trajectory_is_counted_apart_from_a_failure(tmp_path):
    """It is missing data, not a failed episode, and a single count would hide which."""
    summary = aggregate([_ep(), _ep(verified=False, tokens=90_000), _ep(traj=False)], tmp_path / "out")
    assert summary.to_record()["episodes_without_trajectory"] == 1


# --- what becomes an SFT row ------------------------------------------------------------------------


def test_only_verified_episodes_become_rows():
    rows = sft_rows([_ep(verified=True), _ep(verified=False, tokens=90_000)])
    assert len(rows) == 1


def test_an_overfit_episode_is_not_imitated():
    """It passed what it was shown and failed what it was not. Training on it teaches the visible
    assertions, which is the failure `overfit_rate` exists to measure."""
    assert sft_rows([_ep(verified=False, overfit=True)]) == []


def test_an_overfit_episode_is_still_used_as_a_negative():
    """The sharpest negative in the corpus: it is what fitting the benchmark looks like. Excluded
    from imitation, retained for contrast."""
    pairs, _ = preference_pairs([_ep(verified=True, tokens=50_000), _ep(verified=False, overfit=True, tokens=20_000)])
    assert len(pairs) == 1
    assert pairs[0]["rejected_tokens"] == 20_000


def test_rows_are_rendered_by_the_shared_formatter():
    """Not a second renderer. Two definitions of a training row drift on the first protocol change,
    and the one downstream of the drift is the one nobody runs."""
    row = sft_rows([_ep()])[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert row["task_id"] == "t1" and row["round_id"] == "r-1" and row["miner_id"] == "carol"


# --- preference pairs ---------------------------------------------------------------------------------


def test_a_pair_is_the_same_task_in_the_same_round():
    """Holding the task, model, harness and surface fixed leaves only what the model did, which is
    the only difference a preference model can usefully learn."""
    pairs, _ = preference_pairs(
        [
            _ep(task="t1", verified=True),
            _ep(task="t2", verified=False, tokens=90_000),
        ]
    )
    assert pairs == [], "no pair spans two tasks"

    pairs, _ = preference_pairs(
        [
            _ep(task="t1", verified=True, round_id="r-1"),
            _ep(task="t1", verified=False, tokens=90_000, round_id="r-2"),
        ]
    )
    assert pairs == [], "no pair spans two rounds"


def test_a_task_with_no_failure_yields_no_pairs():
    assert preference_pairs([_ep(verified=True), _ep(verified=True, tokens=60_000)]) == ([], [])


def test_pairs_keep_the_widest_separation_first():
    """A cap that kept an arbitrary slice would spend the budget on the pairs the model can learn
    least from."""
    pairs, _ = preference_pairs(
        [
            _ep(verified=True, tokens=70_000),
            _ep(verified=True, tokens=50_000),
            _ep(verified=False, tokens=80_000),
            _ep(verified=False, tokens=95_000),
        ]
    )
    assert (pairs[0]["chosen_tokens"], pairs[0]["rejected_tokens"]) == (50_000, 95_000)


def test_the_pair_cap_is_applied_and_reported():
    """One task with forty of each yields 1,600 pairs and would dominate a set built from a dozen
    tasks with two apiece. A silent top-N reads as "everything" when it is not."""
    episodes = [_ep(verified=True, tokens=50_000 + i) for i in range(6)]
    episodes += [_ep(verified=False, tokens=90_000 + i) for i in range(6)]
    pairs, capped = preference_pairs(episodes, max_per_task=5)
    assert len(pairs) == 5
    assert capped == ["t1"]


def test_a_task_under_the_cap_is_not_reported_as_capped():
    """Otherwise every task looks truncated and the flag stops meaning anything."""
    pairs, capped = preference_pairs([_ep(verified=True), _ep(verified=False, tokens=90_000)], max_per_task=5)
    assert len(pairs) == 1 and capped == []


def test_both_sides_of_a_pair_are_full_message_lists():
    pairs, _ = preference_pairs([_ep(verified=True), _ep(verified=False, tokens=90_000)])
    pair = pairs[0]
    assert pair["chosen"][0]["role"] == "system" and pair["rejected"][0]["role"] == "system"
    assert pair["chosen_tokens"] < pair["rejected_tokens"]


# --- writing ------------------------------------------------------------------------------------------


def test_both_files_are_written_and_are_valid_jsonl(tmp_path):
    out = tmp_path / "datasets"
    summary = aggregate([_ep(verified=True), _ep(verified=False, tokens=90_000)], out)
    for name, expected in (("sft.jsonl", summary.sft_rows), ("preference.jsonl", summary.pairs)):
        lines = [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == expected


def test_only_settled_rounds_are_collected(tmp_path):
    """A round still open may yet change, and a dataset built from one would need rebuilding
    silently -- nothing records which rounds a dataset was made from."""
    from hermes.challenge import Attempt, Baseline, open_challenge
    from hermes.round import open_round
    from validator.aggregate import collect
    from validator.store import RoundStore

    store = RoundStore(tmp_path / "rounds", require_private=False)
    attempts = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=78_000 + i * 900,
            tool_calls=11,
            wall_time_s=1.0,
            steps=34,
            max_steps_hit=True,
        )
        for i in range(10)
    )
    challenge = open_challenge(
        Baseline(task_id="t1", attempts=attempts),
        epoch={"model_revision": "a" * 40, "harness_digest": "b" * 64},
        task_pins={"task_id": "t1", "hidden_verify_commitment": "sha256:" + "c" * 64},
    )
    window = open_round(challenge, round_id="r-1", opened_at=0.0, deadline=1_000.0)
    window.submit("carol", paths=["SOUL.md"], payload_digest="sha256:" + "d" * 64, received_at=10.0)
    store.save(window)

    logs = tmp_path / "judge" / "r-1"
    logs.mkdir(parents=True)
    (logs / "carol.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t1",
                "metrics": {"task_id": "t1", "public_passed": True, "hidden_passed": True, "tokens_used": 50_000},
                "trajectory": _trajectory().to_record(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert collect(store=store, episode_root=tmp_path / "judge") == []

    window.freeze(now=2_000.0)
    window.record_verdict(window.token(), "carol", passed=True)
    window.grade(now=2_001.0)
    window.settle(now=2_002.0)
    store.save(window)
    assert len(collect(store=store, episode_root=tmp_path / "judge")) == 1

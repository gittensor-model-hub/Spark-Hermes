"""Opening challenges from a baseline's own episode log, and the two zeros that look alike.

`hermesbench.runner --episodes-out` writes one `EpisodeMetrics.to_record()` per line and
`hermes.challenge` could package a `Baseline` -- and nothing carried one to the other, so the
first real challenges were going to be hand-assembled from a summary table. These tests drive
the join, and then guard the case that made it worth having: a task where every attempt fails
because the *grader* cannot pass.

The packets in `datasets/challenges/` are checked here too, against the real publish screen
rather than against an assertion about their shape. A packet that `hermes.round` would refuse to
serve is not a challenge, however well-formed the JSON is.
"""

import json
from pathlib import Path

import pytest

from hermes.challenge import (
    Attempt,
    Baseline,
    episode_metrics_of,
    from_episode_log,
    open_challenge,
    unverifiable_tasks,
)
from hermes.round import refuse_withheld_body

EPOCH = {"model_revision": "a" * 40, "harness_digest": "b" * 64}
PINS = {"task_id": "t", "hidden_verify_commitment": "sha256:" + "c" * 64}


def _row(task_id="t", *, passed=False, max_steps_hit=True, tokens=30_000, **kw):
    base = {
        "task_id": task_id,
        "public_passed": passed,
        "tokens_used": tokens,
        "tool_calls": 9,
        "wall_time_s": 40.0,
        "steps": 12,
        "max_steps_hit": max_steps_hit,
    }
    base.update(kw)
    return base


# --- the join ---------------------------------------------------------------------------------


def test_episodes_are_grouped_by_task_and_each_group_becomes_one_baseline():
    rows = [_row("a") for _ in range(10)] + [_row("b") for _ in range(10)]
    opened, refused = from_episode_log(rows, epoch=EPOCH)
    assert {c.task_id for c in opened} == {"a", "b"}
    assert not refused
    assert all(len(c.baseline.attempts) == 10 for c in opened)


def test_a_task_the_baseline_handles_is_refused_with_its_reason_rather_than_dropped():
    """A caller that only sees the successes cannot tell "the baseline handles this" from "the
    baseline is flaky here", and those call for different work."""
    rows = [_row("easy", passed=True, max_steps_hit=False) for _ in range(10)]
    opened, refused = from_episode_log(rows, epoch=EPOCH)
    assert not opened
    assert refused and refused[0][0] == "easy"
    assert "nothing for a miner to improve" in refused[0][1]


def test_a_baseline_below_the_attempt_floor_is_refused():
    opened, refused = from_episode_log([_row() for _ in range(3)], epoch=EPOCH, min_attempts=5)
    assert not opened and refused


def test_missing_optional_fields_do_not_crash_the_reader():
    """`hidden_passed` is absent from a record when no withheld check ran, and `malformed_turns`
    was added to the metrics after the first logs were written."""
    rows = [{"task_id": "t", "public_passed": False, "tokens_used": 1, "max_steps_hit": True} for _ in range(10)]
    opened, _ = from_episode_log(rows, epoch=EPOCH)
    assert opened and opened[0].baseline.attempts[0].hidden_passed is None


# --- the two zeros that look alike -------------------------------------------------------------
#
# The case that cost two rounds. `fix-failing-test` and `verify-speedup-claim` invoked a bare
# `python`, which the harness does not guarantee; the agent finished, declared itself done, and
# the grader failed it anyway. Both scored 10/10 once the interpreter was resolved, but the log
# they had already written still reads as a 0/10 capability gap -- and `hermes.challenge` opened
# challenges on both.


def test_a_log_with_no_stamp_is_unverifiable_rather_than_assumed_to_match():
    """Absence of a mismatch is not evidence of agreement, and defaulting the other way is
    exactly how a stale log gets published."""
    rows = [_row("t") for _ in range(10)]
    problems = unverifiable_tasks(rows, current={"t": "sha256:aaa"})
    assert "no verify_digest" in problems["t"]


def test_a_stamp_that_disagrees_with_the_tree_is_refused_per_task():
    rows = [_row("stale", verify_digest="sha256:old"), _row("fine", verify_digest="sha256:now")]
    problems = unverifiable_tasks(rows, current={"stale": "sha256:now", "fine": "sha256:now"})
    assert "different grader" in problems["stale"]
    assert "fine" not in problems, "one repaired verifier must not invalidate the other baselines"


def test_a_matching_stamp_passes():
    rows = [_row("t", verify_digest="sha256:now") for _ in range(10)]
    assert unverifiable_tasks(rows, current={"t": "sha256:now"}) == {}


def test_the_override_publishes_an_unstamped_log_and_is_the_only_way_to():
    rows = [_row("t") for _ in range(10)]
    assert unverifiable_tasks(rows, current={"t": "x"}, trust_unstamped=True) == {}


def test_a_recovered_stamp_is_compared_rather_than_trusted():
    """`--baseline-ref` reads the task YAML out of the old tree, so "I checked by hand" becomes a
    comparison. A recovered stamp that disagrees is still a refusal."""
    rows = [_row("t") for _ in range(10)]
    assert unverifiable_tasks(rows, current={"t": "sha256:now"}, as_of={"t": "sha256:now"}) == {}
    problems = unverifiable_tasks(rows, current={"t": "sha256:now"}, as_of={"t": "sha256:old"})
    assert "different grader" in problems["t"]


def test_the_runner_stamps_the_digest_the_checker_compares_against():
    """Two spellings of this digest is how the writer and the reader end up disagreeing about
    whether a grader changed. Same function, called from both sides."""
    from hermesbench.runner import verify_digest
    from hermesbench.tasks import load_suite

    # A real task rather than a stub with a `verify` attribute: the stub type-checks against
    # nothing and would keep passing if the function started reading another field.
    task = next(t for t in load_suite("all") if t.task_id == "tc-log-rotation-order")
    stamp = verify_digest(task)
    assert stamp.startswith("sha256:")
    assert unverifiable_tasks([_row(task.task_id, verify_digest=stamp)], current={task.task_id: stamp}) == {}


def test_the_two_repaired_verifiers_are_recoverable_from_git_and_the_rest_are_not_disturbed():
    """The end-to-end claim, against the real history rather than a fixture.

    `6b3ef7d` repaired both verifiers, so digesting the suite at its parent must disagree with
    the tree for exactly those two tasks. If this ever fails on the other seventeen the check has
    become a blanket refusal and would quietly stop anything from being published.
    """
    from hermes.challenge import _verify_digests_at
    from hermesbench.runner import verify_digest
    from hermesbench.tasks import load_suite

    before = _verify_digests_at("6b3ef7d^")
    if not before:
        pytest.skip("shallow checkout: the pre-repair tree is not in this history")
    now = {t.task_id: verify_digest(t) for t in load_suite("all")}
    changed = {k for k, v in before.items() if k in now and now[k] != v}
    assert changed == {"fix-failing-test", "verify-speedup-claim"}


def test_the_step_budget_signal_was_rejected_as_the_discriminator():
    """Recorded because it is the tempting wrong answer, and it was briefly implemented.

    In the real run the four genuine challenges exhausted their step budget on 9 or 10 of 10
    attempts and the two ungradeable ones hit it 0 of 10 times -- a clean split, and useless as a
    rule. A model that writes a bad patch and stops is the single most common real capability gap
    there is, so refusing every zero that did not run out of steps refuses the archetypal
    challenge. It broke ten existing tests whose fixtures are exactly that shape, which is what
    surfaced it. Asking whether the verifier changed answers the actual question.
    """
    attempts = tuple(
        Attempt(public_passed=False, hidden_passed=None, tokens=5_000, tool_calls=3, wall_time_s=9.0, steps=4)
        for _ in range(10)
    )
    opened = open_challenge(Baseline(task_id="t", attempts=attempts), epoch=EPOCH, task_pins=PINS)
    assert opened.baseline.pass_rate == 0.0
    assert not any(a.max_steps_hit for a in opened.baseline.attempts)


# --- the packets that were actually published ---------------------------------------------------

PACKETS = sorted(Path("datasets/challenges").glob("*.json"))


@pytest.mark.skipif(not PACKETS, reason="no challenges have been opened in this checkout")
@pytest.mark.parametrize("path", PACKETS, ids=lambda p: p.stem)
def test_every_published_packet_survives_the_real_publish_screen(path):
    """Through `refuse_withheld_body`, the function the API calls, rather than a check on keys.

    A packet is a file on disk that a miner fetches over HTTP. Asserting its shape here and
    screening it there would be two rules that can drift; this is the one that decides.
    """
    refuse_withheld_body(json.loads(path.read_text(encoding="utf-8")), where=str(path))


@pytest.mark.skipif(not PACKETS, reason="no challenges have been opened in this checkout")
@pytest.mark.parametrize("path", PACKETS, ids=lambda p: p.stem)
def test_every_published_packet_is_a_work_order_and_names_its_withheld_check(path):
    """Two failures that both produce valid JSON. Without the commitment the packet cannot say
    which withheld check grades it; without the prompt no miner can do the work. The first
    version of the writer passed two task keys through a 14-key allowlist and shipped the second.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["task"]["hidden_verify_commitment"].startswith("sha256:")
    assert record["task"]["prompt"].strip()
    assert record["withheld"]["body_included"] is False
    assert record["acceptance_thresholds_included"] is False
    assert "hidden_verify" in record["withheld"]["dropped_task_keys"]


# --- the reader against the real producer, not a hand-shaped dict -------------------------------
#
# The bug this section exists for. `from_episode_log` claimed to consume what
# `runner --episodes-out` writes and did not: `JsonlEpisodeSink.append` nests the metrics under a
# "metrics" key beside `episode`, `task_id`, `setup_failed` and `disqualified`, and the reader
# looked for metric names at the top level. Every lookup missed and every default applied, so a
# real log of ten healthy episodes read as ten zero-token failures -- silently, because to `bool()`
# a missing key and a false value are the same thing.
#
# It passed its tests because the tests and the 190-episode file it was developed against were
# both flat exports. So these drive the actual sink.


def _sunk(tmp_path, metrics_rows):
    """Write episodes through the real JsonlEpisodeSink and read them back."""
    from hermesbench.metrics import EpisodeMetrics
    from hermesbench.sink import JsonlEpisodeSink, read_episodes

    class _Integrity:
        disqualified = False

    class _Result:
        def __init__(self, m):
            self.task_id = m.task_id
            self.metrics = m
            self.setup_failed = m.setup_failed
            self.integrity = _Integrity()

    path = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(path) as sink:
        for kw in metrics_rows:
            sink.append(_Result(EpisodeMetrics(**kw)))
    return list(read_episodes(path))


def _metrics_kw(**kw):
    base = dict(
        task_id="t",
        success=False,
        tool_calls=9,
        failed_calls=0,
        hit_failure=False,
        recovered=False,
        mutated=True,
        self_checked=True,
        tokens_used=78_417,
        wall_time_s=317.0,
        steps=33,
        max_steps_hit=True,
        public_passed=False,
    )
    base.update(kw)
    return base


def test_the_reader_sees_real_numbers_through_the_sinks_own_wrapper(tmp_path):
    """The regression. Before the fix this produced ten attempts of zero tokens that all read as
    failures, and `open_challenge` happily packaged them."""
    rows = _sunk(tmp_path, [_metrics_kw() for _ in range(10)])
    assert "metrics" in rows[0], "the sink nests; if this changes the reader must be revisited"

    opened, refused = from_episode_log(rows, epoch=EPOCH, task_pins=PINS)
    assert opened and not refused
    baseline = opened[0].baseline
    assert baseline.median_tokens == 78_417, "tokens must survive the wrapper"
    assert all(a.max_steps_hit for a in baseline.attempts)
    assert baseline.attempts[0].steps == 33


def test_a_flat_export_still_reads_because_both_shapes_are_real_inputs():
    rows = [_row("t") for _ in range(10)]
    opened, _ = from_episode_log(rows, epoch=EPOCH, task_pins=PINS)
    assert opened and opened[0].baseline.attempts[0].tokens == 30_000


def test_the_wrappers_setup_failed_wins_over_the_metrics_copy(tmp_path):
    """It appears in both places and they are not redundant: the wrapper is the runner's verdict
    for the episode, the metrics are what the episode measured. Taking the wrapper's keeps a
    setup failure classified as infrastructure breakage rather than as an agent failure, which is
    what `open_challenge` refuses on."""
    rows = _sunk(tmp_path, [_metrics_kw(setup_failed=True) for _ in range(10)])
    assert episode_metrics_of(rows[0])["setup_failed"] is True

    opened, refused = from_episode_log(rows, epoch=EPOCH, task_pins=PINS)
    assert not opened
    assert refused and "infrastructure breakage" in refused[0][1]


def test_normalising_a_row_with_no_metrics_key_returns_it_unchanged():
    row = {"task_id": "t", "tokens_used": 5}
    assert episode_metrics_of(row) is row


def test_the_task_id_survives_when_only_the_wrapper_carries_it():
    """`EpisodeMetrics.to_record` does emit task_id, but the wrapper is the authority on which
    task the episode belongs to -- and a row grouped under "" is a row silently dropped."""
    assert episode_metrics_of({"task_id": "real", "metrics": {"tokens_used": 1}})["task_id"] == "real"

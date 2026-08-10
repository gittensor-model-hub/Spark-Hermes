"""Repeats and intervals: what a fifteen-task suite can and cannot establish."""

import json

import pytest

from hermesbench.repeats import Interval, distinguishable, repeat_suite, wilson

# --- the interval ---------------------------------------------------------------------


def test_a_perfect_score_does_not_claim_certainty():
    """The normal approximation reports zero width at 15/15; that is the reason for Wilson."""
    interval = wilson(15, 15)
    assert interval.estimate == 1.0
    assert interval.low < 1.0
    assert interval.width > 0.1


def test_a_zero_score_does_not_claim_certainty_either():
    interval = wilson(0, 15)
    assert interval.high > 0.0 and interval.low == 0.0


def test_more_observations_narrow_the_interval():
    assert wilson(50, 100).width < wilson(5, 10).width


def test_an_interval_over_nothing_is_refused():
    with pytest.raises(ValueError, match="describes nothing"):
        wilson(0, 0)


def test_more_successes_than_observations_is_refused():
    with pytest.raises(ValueError, match="not a proportion"):
        wilson(5, 3)


def test_interval_record_is_json_safe():
    assert json.loads(json.dumps(wilson(9, 15).to_record()))["n"] == 15


# --- what fifteen tasks actually support ------------------------------------------------


def test_the_current_suite_size_cannot_separate_a_ten_point_difference():
    """The honest finding, and the reason to grow the suite rather than rerun it."""
    a, b = wilson(12, 15), wilson(10, 15)
    ok, reason = distinguishable(a, b)
    assert not ok and "too small to separate" in reason


def test_a_large_enough_gap_is_distinguishable():
    ok, reason = distinguishable(wilson(15, 15), wilson(2, 15))
    assert ok and reason == ""


def test_overlap_is_reported_conservatively():
    """Non-overlap is sufficient for a difference, not necessary."""
    assert Interval(0.5, 0.3, 0.7, 15).overlaps(Interval(0.6, 0.4, 0.8, 15))
    assert not Interval(0.9, 0.8, 1.0, 15).overlaps(Interval(0.2, 0.1, 0.3, 15))


# --- repeats ------------------------------------------------------------------------------


def test_a_stable_task_is_not_flaky():
    suite = repeat_suite(["t1"], lambda task, attempt: True, repeats=5)
    assert suite.tasks[0].passes == 5 and not suite.tasks[0].flaky


def test_a_task_that_flips_is_named_rather_than_averaged_away():
    """A suite where many tasks flip cannot separate two models however often it is rerun."""
    suite = repeat_suite(["t1"], lambda task, attempt: attempt % 2 == 0, repeats=4)
    assert suite.tasks[0].flaky
    assert [t.task_id for t in suite.flaky_tasks] == ["t1"]


def test_the_attempt_index_is_passed_through_so_a_caller_can_vary_a_seed():
    seen: list[int] = []

    def run(task, attempt):
        seen.append(attempt)
        return True

    repeat_suite(["t1"], run, repeats=3)
    assert seen == [0, 1, 2]


def test_running_zero_times_is_refused():
    with pytest.raises(ValueError, match="at least once"):
        repeat_suite(["t1"], lambda t, a: True, repeats=0)


def test_repeats_do_not_narrow_the_interval():
    """Counting every attempt as an independent observation would claim that rerunning the
    same tasks tells you more about the population of tasks."""
    once = repeat_suite(["a", "b", "c"], lambda t, n: True, repeats=1)
    many = repeat_suite(["a", "b", "c"], lambda t, n: True, repeats=20)
    assert once.interval.n == many.interval.n == 3
    assert once.interval.width == many.interval.width


def test_mean_rate_averages_over_tasks():
    suite = repeat_suite(["a", "b"], lambda t, n: t == "a", repeats=4)
    assert suite.mean_rate == 0.5


def test_suite_record_is_json_safe():
    suite = repeat_suite(["a", "b"], lambda t, n: t == "a" or n == 0, repeats=3)
    record = json.loads(json.dumps(suite.to_record()))
    assert record["repeats"] == 3 and len(record["per_task"]) == 2
    assert "b" in record["flaky_tasks"]


# --- paired comparison: the runs are already paired, so use it -------------------------------


def _passes(ids, passing):
    return {t: (t in passing) for t in ids}


IDS = [f"t{i:02d}" for i in range(15)]


def test_identical_decisions_are_not_evidence_of_a_difference():
    from hermesbench.repeats import paired, paired_distinguishable

    same = _passes(IDS, IDS[:11])
    ok, reason = paired_distinguishable(paired(same, dict(same)))
    assert not ok
    assert "not evidence that there is none" in reason


def test_the_same_rate_on_different_tasks_is_visible_only_when_paired():
    """Both models pass 11 of 15, so every unpaired interval is identical -- and eight tasks
    disagree. Only the paired view can see that they are not the same model."""
    from hermesbench.repeats import paired, wilson

    a = _passes(IDS, IDS[:11])
    b = _passes(IDS, IDS[4:])
    assert wilson(11, 15) == wilson(11, 15)  # the unpaired view cannot tell them apart
    comparison = paired(a, b)
    assert comparison.discordant == 8
    assert comparison.only_a == 4 and comparison.only_b == 4


def test_a_one_sided_split_reaches_significance_where_the_unpaired_test_cannot():
    """The whole reason to pair. Six discordant tasks all favouring one model is p=0.031,
    while the unpaired intervals at 6/15 and 0/15 still overlap -- so the same runs support
    a claim under the paired test that the unpaired one cannot make."""
    from hermesbench.repeats import distinguishable, paired, paired_distinguishable, wilson

    a = _passes(IDS, IDS[:6])
    b = _passes(IDS, [])
    comparison = paired(a, b)
    assert comparison.only_a == 6 and comparison.only_b == 0
    ok, _reason = paired_distinguishable(comparison)
    assert ok
    unpaired, reason = distinguishable(wilson(6, 15), wilson(0, 15))
    assert not unpaired and "too small to separate" in reason


def test_below_six_discordant_tasks_the_suite_forecloses_the_claim():
    """Not "we did not find a difference" but "no result on this suite could show one",
    which is a fact about the benchmark rather than about the models."""
    from hermesbench.repeats import paired, paired_distinguishable

    a = _passes(IDS, IDS[:9])
    b = _passes(IDS, IDS[:7] + IDS[9:11])
    comparison = paired(a, b)
    assert comparison.discordant < 6
    ok, reason = paired_distinguishable(comparison)
    assert not ok
    assert "however they perform" in reason


def test_a_task_in_one_result_and_not_the_other_is_refused():
    """Intersecting the keys would compute a real number over a set nobody chose."""
    import pytest

    from hermesbench.repeats import paired

    with pytest.raises(ValueError, match="appear in one result and not the other"):
        paired(_passes(IDS, IDS[:5]), _passes(IDS[:14], IDS[:5]))


def test_the_p_value_never_exceeds_one():
    """At k == n/2 the two tails overlap and a naive doubling exceeds 1.0."""
    from hermesbench.repeats import PairedComparison

    for n in range(0, 13):
        for only_a in range(n + 1):
            c = PairedComparison(both_passed=0, both_failed=0, only_a=only_a, only_b=n - only_a)
            assert 0.0 <= c.p_value <= 1.0, (n, only_a)


def test_an_exact_split_is_the_least_significant_result_possible():
    from hermesbench.repeats import PairedComparison

    assert PairedComparison(both_passed=0, both_failed=0, only_a=5, only_b=5).p_value == 1.0


def test_a_clean_sweep_of_ten_is_significant():
    from hermesbench.repeats import PairedComparison

    assert PairedComparison(both_passed=0, both_failed=0, only_a=10, only_b=0).p_value < 0.01

"""Preflight: measure what an endpoint adds to a request that our digests cannot see."""

import pytest

from hermes.preflight import (
    PROBE_LONG,
    PROBE_SHORT,
    TEMPLATE_TOKEN_BUDGET,
    EndpointProfile,
    PreflightReport,
    compare,
    count_tokens,
    probe_overhead,
    problems,
)


def _endpoint(short, long_, *, teacher_id="t", model="m"):
    """A stub endpoint billing a fixed number of prompt tokens per probe."""
    billed = {PROBE_SHORT: short, PROBE_LONG: long_}

    def complete(messages):
        return {"prompt_tokens": billed[messages[0]["content"]]}

    return probe_overhead(teacher_id=teacher_id, model=model, complete=complete, usage_of=lambda r: r["prompt_tokens"])


# --- an honest endpoint ---------------------------------------------------------------------


def test_a_clean_endpoint_reports_no_unexplained_overhead():
    """Template framing only: a handful of tokens for role markers and delimiters."""
    short = count_tokens(PROBE_SHORT) + 8
    profile = _endpoint(short, count_tokens(PROBE_LONG) + 8)
    assert profile.measured
    assert profile.unexplained == 0
    assert profile.overhead_is_fixed


def test_a_clean_field_has_nothing_to_report():
    short = count_tokens(PROBE_SHORT) + 8
    report = PreflightReport(
        profiles=[_endpoint(short, count_tokens(PROBE_LONG) + 8, teacher_id=f"t{i}") for i in range(2)]
    )
    assert problems(report) == []


# --- the measurement that started this ---------------------------------------------------------


def test_an_injected_prompt_is_detected_as_unexplained_tokens():
    """Measured live on 2026-08-09: a one-character message billed 70 prompt tokens, and the
    model reproduced instructions we never sent. Nothing in the response says so -- the only
    signal is that the arithmetic does not work."""
    profile = _endpoint(70, 84)
    assert profile.measured
    assert profile.fixed_overhead >= 69
    assert profile.unexplained > 40
    report = PreflightReport(profiles=[profile])
    assert any("neither our messages nor a chat template account for" in p for p in problems(report))


def test_the_overhead_is_recognised_as_a_fixed_prefix():
    """70 -> 84 for a sentence that is itself about 14 tokens: the extra cost is a constant
    prepended block, not per-token expansion."""
    assert _endpoint(70, 84).overhead_is_fixed


def test_an_overhead_that_grows_with_the_message_is_flagged_separately():
    """A per-token expansion means the single figure recorded for a batch will not describe a
    long trajectory, which is a different problem from a hidden prefix."""
    profile = _endpoint(70, 300)
    assert not profile.overhead_is_fixed
    assert any("grows with the message" in p for p in problems(PreflightReport(profiles=[profile])))


def test_a_field_whose_endpoints_add_different_context_is_flagged():
    """Tournament asserts a fair fight by requiring one harness digest. Both teachers here
    agree on every digest we compute and receive different amounts of hidden instruction."""
    report = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi"), _endpoint(62, 76, teacher_id="qwen")])
    assert report.overhead_differs
    assert any("cannot see this" in p for p in problems(report))


def test_matching_overheads_are_not_flagged_as_differing():
    report = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="a"), _endpoint(70, 84, teacher_id="b")])
    assert not report.overhead_differs


# --- an endpoint that cannot be measured is unknown, not clean ------------------------------------


def test_an_endpoint_that_errors_is_reported_as_unknown_rather_than_zero():
    def complete(_messages):
        raise RuntimeError("gateway saturated")

    profile = probe_overhead(teacher_id="down", model="m", complete=complete, usage_of=lambda r: r["prompt_tokens"])
    assert not profile.measured
    assert "gateway saturated" in profile.error
    assert any("unknown rather than zero" in p for p in problems(PreflightReport(profiles=[profile])))


def test_an_unmeasured_endpoint_does_not_claim_a_fair_comparison():
    """It must not count as agreeing with anyone, or a dead endpoint would silence the
    differing-context warning by having no context at all."""
    down = probe_overhead(
        teacher_id="down",
        model="m",
        complete=lambda _m: (_ for _ in ()).throw(RuntimeError("boom")),
        usage_of=lambda r: r["prompt_tokens"],
    )
    report = PreflightReport(profiles=[down, _endpoint(70, 84, teacher_id="up")])
    assert not report.overhead_differs


# --- the point of recording it: a change is visible -------------------------------------------------


def test_an_overhead_change_between_batches_is_reported():
    """An injected prompt is tolerable if it is stable and disclosed. The failure is a corpus
    whose halves were conditioned differently with nothing in either half saying so."""
    before = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi")]).to_record()
    after = PreflightReport(profiles=[_endpoint(120, 134, teacher_id="kimi")])
    changes = compare(before, after)
    assert any("prompt overhead moved 69 -> 119" in c for c in changes)


def test_an_unchanged_overhead_reports_nothing():
    before = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi")]).to_record()
    assert compare(before, PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi")])) == []


def test_a_new_or_departed_endpoint_is_reported():
    before = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi")]).to_record()
    after = PreflightReport(profiles=[_endpoint(62, 76, teacher_id="qwen")])
    changes = compare(before, after)
    assert any("was not in the previous profile" in c for c in changes)
    assert any("absent now" in c for c in changes)


# --- shape ------------------------------------------------------------------------------------------


def test_the_record_carries_what_a_reviewer_needs():
    record = PreflightReport(profiles=[_endpoint(70, 84, teacher_id="kimi")]).to_record()
    assert record["endpoints"][0]["fixed_overhead"] == 69
    assert record["max_unexplained"] == 69 - TEMPLATE_TOKEN_BUDGET
    assert record["overhead_differs_across_field"] is False


def test_probe_strings_are_short_enough_that_any_estimator_agrees():
    """The crude token count only has to be right for these two strings."""
    assert count_tokens(PROBE_SHORT) == 1
    assert 8 <= count_tokens(PROBE_LONG) <= 20


def test_a_profile_with_no_measurement_has_no_overhead_to_report():
    profile = EndpointProfile(teacher_id="t", model="m", error="never ran")
    assert profile.fixed_overhead == 0
    assert profile.unexplained == 0
    assert not profile.overhead_is_fixed


@pytest.mark.parametrize("billed", [0, -5])
def test_a_nonsense_prompt_count_does_not_produce_negative_overhead(billed):
    profile = _endpoint(billed, billed)
    assert profile.fixed_overhead >= 0
    assert profile.unexplained >= 0

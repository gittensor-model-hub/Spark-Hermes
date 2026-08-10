"""Dataset mixture: is this corpus the thing we said we were building?"""

import json

import pytest

from hermes.mixture import (
    AGENT_DATASET_V1,
    DatasetMix,
    MixtureError,
    count_by,
    plan_sample,
    validate,
)

# --- the spec ----------------------------------------------------------------------


def test_weights_must_sum_to_one():
    with pytest.raises(MixtureError, match="sum to"):
        DatasetMix(name="m", weights={"a": 0.5, "b": 0.2})


def test_negative_weights_are_rejected():
    with pytest.raises(MixtureError, match="negative weight"):
        DatasetMix(name="m", weights={"a": 1.2, "b": -0.2})


def test_an_empty_mixture_is_rejected():
    with pytest.raises(MixtureError, match="at least one bucket"):
        DatasetMix(name="m", weights={})


def test_the_shipped_agent_mix_is_well_formed():
    assert abs(sum(AGENT_DATASET_V1.weights.values()) - 1.0) < 1e-9
    assert AGENT_DATASET_V1.weights["coding"] == 0.40


def test_mix_round_trips_through_a_record():
    assert DatasetMix.from_record(AGENT_DATASET_V1.to_record()).weights == AGENT_DATASET_V1.weights


# --- validation --------------------------------------------------------------------


def test_a_corpus_matching_its_spec_is_on_spec():
    counts = {"coding": 400, "terminal": 200, "research": 200, "browser": 100, "recovery": 100}
    report = validate(counts, AGENT_DATASET_V1)
    assert report.on_spec
    assert report.total == 1000
    assert not report.starved


def test_drift_is_reported_per_bucket_not_as_one_distance():
    """One number says the corpus is 12% off without saying which half is missing."""
    counts = {"coding": 120, "terminal": 200, "research": 480, "browser": 100, "recovery": 100}
    report = validate(counts, AGENT_DATASET_V1)

    assert not report.on_spec
    by_bucket = {b.bucket: b for b in report.buckets}
    assert by_bucket["coding"].drift < 0  # starved
    assert by_bucket["research"].drift > 0  # oversupplied
    assert by_bucket["terminal"].drift == pytest.approx(0.0, abs=1e-9)


def test_starved_buckets_point_at_what_to_generate_next():
    counts = {"coding": 120, "terminal": 200, "research": 480, "browser": 100, "recovery": 100}
    assert [b.bucket for b in validate(counts, AGENT_DATASET_V1).starved] == ["coding"]


def test_shortfall_is_expressed_in_rows_not_just_a_fraction():
    counts = {"coding": 100, "terminal": 200, "research": 200, "browser": 100, "recovery": 100}
    report = validate(counts, AGENT_DATASET_V1)
    coding = next(b for b in report.buckets if b.bucket == "coding")
    # 700 rows total, 40% target = 280 wanted, 100 held.
    assert coding.target == 280
    assert coding.shortfall == 180


def test_unplanned_buckets_are_surfaced_rather_than_folded_in():
    """Unplanned data is not automatically bad, but it should be a decision."""
    counts = {"coding": 400, "terminal": 200, "research": 200, "browser": 100, "recovery": 100, "poetry": 50}
    report = validate(counts, AGENT_DATASET_V1)
    assert report.unknown_buckets == {"poetry": 50}
    assert not report.on_spec


def test_a_missing_bucket_reads_as_starved_not_absent():
    counts = {"coding": 400, "terminal": 200, "research": 200, "browser": 100}
    report = validate(counts, AGENT_DATASET_V1)
    recovery = next(b for b in report.buckets if b.bucket == "recovery")
    assert recovery.have == 0
    assert recovery in report.starved


def test_validation_of_an_empty_corpus_does_not_divide_by_zero():
    report = validate({}, AGENT_DATASET_V1)
    assert report.total == 0
    assert all(b.share == 0.0 for b in report.buckets)


def test_tolerance_absorbs_sampling_noise():
    counts = {"coding": 402, "terminal": 199, "research": 200, "browser": 99, "recovery": 100}
    assert validate(counts, AGENT_DATASET_V1).on_spec


def test_report_is_json_safe():
    report = validate({"coding": 10}, AGENT_DATASET_V1)
    assert json.loads(json.dumps(report.to_record()))["mix"] == "spark-hermes-agent-v1"


# --- sampling ----------------------------------------------------------------------


def test_a_full_supply_yields_the_requested_mix():
    available = {b: 10_000 for b in AGENT_DATASET_V1.buckets}
    plan = plan_sample(available, AGENT_DATASET_V1, 1000)
    assert plan.complete
    assert plan.draw["coding"] == 400
    assert plan.achievable == 1000


def test_a_starved_bucket_is_reported_and_never_backfilled():
    """Filling a 40% coding target from spare rows elsewhere ships a false balance."""
    available = {"coding": 50, "terminal": 10_000, "research": 10_000, "browser": 10_000, "recovery": 10_000}
    plan = plan_sample(available, AGENT_DATASET_V1, 1000)

    assert not plan.complete
    assert plan.draw["coding"] == 50
    assert plan.shortfall == {"coding": 350}
    # Nothing was drawn beyond target to compensate.
    assert plan.draw["terminal"] == 200
    assert plan.achievable == 650


def test_every_bucket_short_reports_every_shortfall():
    plan = plan_sample({b: 1 for b in AGENT_DATASET_V1.buckets}, AGENT_DATASET_V1, 1000)
    assert set(plan.shortfall) == set(AGENT_DATASET_V1.buckets)


def test_a_zero_size_plan_is_trivially_complete():
    plan = plan_sample({"coding": 5}, AGENT_DATASET_V1, 0)
    assert plan.complete and plan.achievable == 0


def test_a_negative_size_is_rejected():
    with pytest.raises(MixtureError, match="cannot be negative"):
        plan_sample({}, AGENT_DATASET_V1, -1)


def test_plan_record_is_json_safe():
    plan = plan_sample({"coding": 1}, AGENT_DATASET_V1, 10)
    assert json.loads(json.dumps(plan.to_record()))["complete"] is False


# --- counting ----------------------------------------------------------------------


def test_count_by_tallies_records_and_objects():
    from dataclasses import dataclass

    @dataclass
    class Row:
        bucket: str

    assert count_by([{"bucket": "coding"}, {"bucket": "coding"}, Row("terminal")]) == {
        "coding": 2,
        "terminal": 1,
    }


def test_unlabelled_items_are_counted_rather_than_dropped():
    """Silently dropping them would make an unlabelled corpus look perfectly balanced."""
    assert count_by([{"bucket": "coding"}, {}, {"bucket": None}]) == {"coding": 1, "": 2}

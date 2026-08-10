import math

import pytest

from eval.benchmarks import assert_fraction_scores


def test_accepts_valid_fraction_scores():
    # Ints and floats across the closed [0, 1] range are all valid.
    assert_fraction_scores({"gsm8k": 0.0, "triton": 1, "bfcl": 0.4123}, "candidate") is None


@pytest.mark.parametrize("bad_value", [None, [0.9], {"nested": 1}, (0.9,)])
def test_rejects_non_numeric_score_value(bad_value):
    # Regression: a null/list/object/tuple value used to reach float(value) and raise
    # TypeError, escaping the fail-closed ValueError contract and crashing eval.verify
    # (the CI training-track gate does not wrap verify_submission).
    with pytest.raises(ValueError, match="malformed"):
        assert_fraction_scores({"gsm8k": bad_value}, "candidate")


def test_rejects_non_numeric_string_score():
    # A non-numeric string previously surfaced a bare float() ValueError
    # ("could not convert string to float"); it now fails the fraction contract cleanly.
    with pytest.raises(ValueError, match="malformed"):
        assert_fraction_scores({"gsm8k": "high"}, "candidate")


@pytest.mark.parametrize("bad_value", [True, False])
def test_rejects_boolean_score(bad_value):
    # bool is a subclass of int, so a JSON `true` used to slip through as score 1.0.
    with pytest.raises(ValueError, match="malformed"):
        assert_fraction_scores({"gsm8k": bad_value}, "candidate")


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_score(bad_value):
    # json.loads accepts NaN/Infinity by default; these must be rejected, not scored.
    with pytest.raises(ValueError, match="malformed"):
        assert_fraction_scores({"gsm8k": bad_value}, "candidate")


def test_rejects_non_object_scores_payload():
    with pytest.raises(ValueError, match="must be a JSON object"):
        assert_fraction_scores([0.9, 0.8], "candidate")


@pytest.mark.parametrize("bad_value", [1.5, 90.0, -0.1])
def test_rejects_out_of_range_score(bad_value):
    # Preserved behaviour: a finite number outside [0, 1] (e.g. a 0-100 percentage)
    # is still rejected as out of range.
    with pytest.raises(ValueError, match=r"fractions in \[0, 1\]"):
        assert_fraction_scores({"gsm8k": bad_value}, "candidate")

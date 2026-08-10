"""Cost: an unpriced model is not a free one, and cached input is not input."""

import json

import pytest

from hermes.cost import (
    ANTHROPIC,
    OPENAI,
    Cost,
    CostError,
    Ledger,
    Price,
    PriceBook,
    UnpricedModel,
    Usage,
    cheaper,
    cost_of,
    usage_from_provider,
)

CHEAP = Price(model="cheap", input=1.0, output=2.0, cached_input=0.1, effective="2026-08")
PLAIN = Price(model="plain", input=3.0, output=6.0, effective="2026-08")


def _book() -> PriceBook:
    return PriceBook(revision="2026-08").add(CHEAP).add(PLAIN)


# --- the two provider shapes disagree, and it is silent -----------------------------


def test_openai_prompt_tokens_include_the_cached_ones():
    """Counting them again would overcharge by exactly the size of the discount."""
    usage = usage_from_provider(
        {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 800}},
        shape=OPENAI,
    )
    assert usage.input_tokens == 200
    assert usage.cached_input_tokens == 800
    assert usage.total_input == 1000  # not 1800


def test_anthropic_input_tokens_exclude_the_cached_ones():
    """Subtracting here would undercharge; the shapes are genuinely opposite."""
    usage = usage_from_provider(
        {"input_tokens": 200, "cache_read_input_tokens": 800, "output_tokens": 50},
        shape=ANTHROPIC,
    )
    assert usage.input_tokens == 200
    assert usage.cached_input_tokens == 800
    assert usage.total_input == 1000


def test_anthropic_cache_writes_are_their_own_bucket():
    usage = usage_from_provider(
        {"input_tokens": 10, "cache_creation_input_tokens": 500, "output_tokens": 5}, shape=ANTHROPIC
    )
    assert usage.cache_write_tokens == 500


def test_the_shape_must_be_declared_rather_than_sniffed():
    """An uncached request in either shape looks like the other one."""
    with pytest.raises(CostError, match="unknown provider shape"):
        usage_from_provider({"prompt_tokens": 10}, shape="guess")


def test_cached_tokens_larger_than_the_prompt_is_refused():
    """In the OpenAI shape they are a subset, so this is not that shape."""
    with pytest.raises(CostError, match="not that shape"):
        usage_from_provider({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 99}}, shape=OPENAI)


def test_an_empty_usage_object_is_zero_not_an_error():
    assert usage_from_provider({}, shape=OPENAI).total == 0


def test_missing_cache_fields_mean_nothing_was_cached():
    usage = usage_from_provider({"prompt_tokens": 100, "completion_tokens": 10}, shape=OPENAI)
    assert usage.cached_input_tokens == 0 and usage.input_tokens == 100


def test_negative_counts_are_refused():
    with pytest.raises(CostError, match="cannot be negative"):
        Usage(input_tokens=-1)


# --- an unpriced model is not a free model -----------------------------------------


def test_an_unpriced_model_is_refused_rather_than_costed_at_zero():
    """select_winner breaks ties on lowest cost; zero would win every one of them."""
    with pytest.raises(UnpricedModel, match="no price on file"):
        _book().price_of("brand-new-model")


def test_the_refusal_says_why_zero_would_be_worse():
    with pytest.raises(UnpricedModel, match="cheapest one in every comparison"):
        _book().cost("unknown", Usage(input_tokens=100))


def test_knows_reports_coverage_without_raising():
    book = _book()
    assert book.knows("cheap") and not book.knows("unknown")


# --- pricing ------------------------------------------------------------------------


def test_cached_input_is_billed_at_the_cached_rate():
    dear = cost_of(Usage(input_tokens=1_000_000), CHEAP)
    cheap_run = cost_of(Usage(cached_input_tokens=1_000_000), CHEAP)
    assert dear.total == 1.0
    assert cheap_run.total == pytest.approx(0.1)


def test_a_model_with_no_cache_discount_bills_cached_input_at_the_full_rate():
    """Defaulting to free would credit a discount the provider never gave."""
    assert cost_of(Usage(cached_input_tokens=1_000_000), PLAIN).total == 3.0


def test_components_are_reported_apart():
    """Nine-tenths cached input and the same total from fresh input mean opposite things."""
    cost = cost_of(Usage(input_tokens=1000, cached_input_tokens=9000, output_tokens=100), CHEAP)
    assert set(cost.components) == {"input", "cached_input", "cache_write", "output"}
    assert cost.components["cached_input"] > 0


def test_cost_carries_the_price_revision():
    assert _book().cost("cheap", Usage(input_tokens=10)).revision == "2026-08"


def test_negative_rates_are_refused():
    with pytest.raises(CostError, match="negative"):
        Price(model="m", input=-1.0, output=1.0)


def test_cache_hit_rate_reports_where_the_input_came_from():
    assert Usage(input_tokens=200, cached_input_tokens=800).cache_hit_rate == 0.8
    assert Usage().cache_hit_rate == 0.0


# --- comparison ---------------------------------------------------------------------


def test_costs_from_different_revisions_are_not_compared():
    """Picking the smaller would answer 'which ran when rates were lower'."""
    a = Cost(total=1.0, currency="USD", revision="2026-08")
    b = Cost(total=0.5, currency="USD", revision="2025-01")
    with pytest.raises(CostError, match="measures when the work ran"):
        cheaper(a, b)


def test_costs_in_different_currencies_are_not_compared():
    a = Cost(total=1.0, currency="USD", revision="r")
    b = Cost(total=0.5, currency="EUR", revision="r")
    with pytest.raises(CostError, match="conversion rate"):
        cheaper(a, b)


def test_the_cheaper_of_two_comparable_costs():
    a = Cost(total=1.0, currency="USD", revision="r")
    b = Cost(total=0.5, currency="USD", revision="r")
    assert cheaper(a, b) is b


# --- the ledger stops ----------------------------------------------------------------


def test_spend_accumulates_across_charges():
    ledger = Ledger(price_book=_book())
    ledger.charge("cheap", Usage(input_tokens=1_000_000))
    ledger.charge("cheap", Usage(output_tokens=1_000_000))
    assert ledger.spent == pytest.approx(3.0)
    assert ledger.usage.total == 2_000_000


def test_a_charge_that_would_exceed_the_limit_is_refused_before_it_lands():
    """A ledger never holds a total it was not allowed to reach."""
    ledger = Ledger(price_book=_book(), limit=0.5)
    with pytest.raises(CostError, match="would exceed"):
        ledger.charge("cheap", Usage(input_tokens=1_000_000))
    assert ledger.spent == 0.0


def test_an_exhausted_ledger_refuses_everything():
    ledger = Ledger(price_book=_book(), limit=1.0)
    ledger.charge("cheap", Usage(input_tokens=1_000_000))
    assert ledger.exhausted
    with pytest.raises(CostError, match="exhausted"):
        ledger.charge("cheap", Usage(input_tokens=1))


def test_would_exceed_lets_a_caller_decide_in_advance():
    ledger = Ledger(price_book=_book(), limit=0.5)
    cost = _book().cost("cheap", Usage(input_tokens=1_000_000))
    assert ledger.would_exceed(cost) is True


def test_no_limit_means_unbounded_remaining():
    ledger = Ledger(price_book=_book())
    assert ledger.remaining == float("inf") and not ledger.exhausted


def test_a_currency_mismatch_is_refused():
    book = PriceBook(revision="r").add(Price(model="eu", input=1.0, output=1.0, currency="EUR"))
    ledger = Ledger(price_book=book, currency="USD")
    with pytest.raises(CostError, match="priced in EUR"):
        ledger.charge("eu", Usage(input_tokens=10))


def test_an_unpriced_model_cannot_be_charged_silently():
    ledger = Ledger(price_book=_book(), limit=100.0)
    with pytest.raises(UnpricedModel):
        ledger.charge("unknown", Usage(input_tokens=1_000_000))
    assert ledger.spent == 0.0


def test_ledger_record_is_json_safe():
    ledger = Ledger(price_book=_book(), limit=10.0)
    ledger.charge("cheap", Usage(input_tokens=1000), label="task-1")
    record = json.loads(json.dumps(ledger.to_record()))
    assert record["entries"] == 1 and record["remaining"] < 10.0


def test_usage_record_is_json_safe():
    assert json.loads(json.dumps(Usage(input_tokens=1).to_record()))["total"] == 1


# --- the end-to-end path the repo already had --------------------------------------


def test_a_captured_provider_usage_object_prices_straight_through():
    """teacher/providers.py already records this dict on every row and nothing read it."""
    captured = {"prompt_tokens": 10_000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 9_000}}
    usage = usage_from_provider(captured, shape=OPENAI)
    cost = _book().cost("cheap", usage)
    # 1k fresh at 1.0/M + 9k cached at 0.1/M + 500 out at 2.0/M
    assert cost.total == pytest.approx(0.001 + 0.0009 + 0.001)
    assert usage.cache_hit_rate == 0.9


def test_ignoring_the_cache_split_overcharges_by_the_size_of_the_discount():
    captured = {"prompt_tokens": 10_000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 9_000}}
    correct = _book().cost("cheap", usage_from_provider(captured, shape=OPENAI))
    naive = cost_of(Usage(input_tokens=10_000), CHEAP)
    assert naive.total > correct.total * 4


# --- regressions: adversarial pass ---------------------------------------------------


def test_nan_never_reaches_the_ledger():
    """`nan < 0` is False, so a sign test alone lets it through -- and one NaN makes
    `spent >= limit` False forever, so the budget stops refusing anything."""
    with pytest.raises(CostError, match="must be finite"):
        Usage(input_tokens=float("nan"))


def test_infinite_counts_are_refused():
    with pytest.raises(CostError, match="must be finite"):
        Usage(output_tokens=float("inf"))


def test_a_nan_rate_is_refused():
    with pytest.raises(CostError, match="rates must be finite"):
        Price(model="m", input=float("nan"), output=1.0)


def test_a_ledger_ceiling_survives_a_hostile_charge():
    ledger = Ledger(price_book=_book(), limit=1.0)
    with pytest.raises(CostError):
        ledger.charge("cheap", Usage(input_tokens=float("nan")))
    ledger.charge("cheap", Usage(input_tokens=1_000_000))
    assert ledger.exhausted
    with pytest.raises(CostError, match="exhausted"):
        ledger.charge("cheap", Usage(input_tokens=1))


def test_a_usage_object_read_under_the_wrong_shape_is_refused():
    """Parsed as zeros, a real model costs 0.0 and wins the cost tie-break outright."""
    anthropic_turn = {"input_tokens": 200, "cache_read_input_tokens": 20_000, "output_tokens": 1_500}
    with pytest.raises(CostError, match="not that shape"):
        usage_from_provider(anthropic_turn, shape=OPENAI)


def test_the_mismatch_is_caught_in_both_directions():
    openai_turn = {"prompt_tokens": 1000, "completion_tokens": 50}
    with pytest.raises(CostError, match="not that shape"):
        usage_from_provider(openai_turn, shape=ANTHROPIC)


def test_a_streaming_delta_with_only_output_tokens_still_parses():
    """Anthropic's message_delta carries no input fields; refusing it would be wrong."""
    assert usage_from_provider({"output_tokens": 12}, shape=ANTHROPIC).output_tokens == 12


def test_a_non_mapping_usage_object_raises_a_typed_error():
    with pytest.raises(CostError, match="must be a mapping"):
        usage_from_provider([1, 2], shape=OPENAI)  # type: ignore[arg-type]


def test_cache_writes_without_a_rate_are_refused_rather_than_undercharged():
    """Anthropic bills 1.25x for the 5-minute TTL and 2x for the hour; the TTL is not in
    the usage object, so no single default is right."""
    with pytest.raises(CostError, match="undercharges"):
        cost_of(Usage(cache_write_tokens=1000), PLAIN)


def test_a_declared_cache_write_rate_prices_normally():
    price = Price(model="p", input=3.0, output=15.0, cache_write=3.75)
    assert cost_of(Usage(cache_write_tokens=1_000_000), price).total == pytest.approx(3.75)


def test_openai_cache_write_tokens_are_a_subset_of_the_prompt_too():
    usage = usage_from_provider(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        },
        shape=OPENAI,
    )
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (100, 600, 300)
    assert usage.total_input == 1000


def test_repricing_a_model_in_place_is_refused():
    """cheaper() refuses across revisions; that only means anything if a revision names
    one rate card."""
    book = _book()
    with pytest.raises(CostError, match="two rate cards sharing one revision"):
        book.add(Price(model="cheap", input=0.01, output=0.01))

import json

import pytest

from hermes.router import (
    DOMAINS,
    GENERAL,
    KeywordRouter,
    ModelRouter,
    RoutingDecision,
    RoutingError,
    build_prompt,
    compare,
    evaluate,
    load_suite,
    model_for,
    parse_decision,
)
from hermes.router.evaluate import RoutingDatasetError, RoutingExample, load_examples

SUITE = load_suite("v0")


# --- decision type -----------------------------------------------------------------


def test_unknown_target_is_rejected():
    with pytest.raises(RoutingError, match="unknown routing target"):
        RoutingDecision(target="database", confidence=0.9, reason="x")


def test_confidence_must_be_a_fraction():
    with pytest.raises(RoutingError, match=r"\[0, 1\]"):
        RoutingDecision(target="cuda", confidence=1.5, reason="x")


def test_decision_exposes_the_serving_model():
    assert RoutingDecision(target="cuda", confidence=0.9, reason="x").model == "Spark-Hermes-CUDA"
    assert model_for(GENERAL) == "Spark-Hermes-3.8-27B"


def test_decision_record_is_json_safe():
    d = RoutingDecision(target="swe", confidence=0.5, reason="x", scores={"swe": 1.0})
    assert json.loads(json.dumps(d.to_record()))["target"] == "swe"


# --- keyword baseline --------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Profile this Triton kernel with Nsight and fix bank conflicts", "cuda"),
        ("The UART driver on our STM32 drops bytes; debug the interrupt handler", "firmware"),
        ("ASan found a use-after-free; write a proof of concept", "cyber"),
        ("Fix the failing pytest and open a pull request", "swe"),
    ],
)
def test_keyword_router_routes_clear_domain_tasks(task, expected):
    assert KeywordRouter().route(task).target == expected


def test_keyword_router_abstains_on_an_empty_task():
    decision = KeywordRouter().route("   ")
    assert decision.abstained is True
    assert decision.target == GENERAL


def test_keyword_router_abstains_when_nothing_is_indicated():
    decision = KeywordRouter().route("Help me plan the offsite agenda")
    assert decision.abstained is True


def test_keyword_router_abstains_on_cross_domain_work():
    """A task spanning three specialists is generalist work, not the leader's work."""
    decision = KeywordRouter().route(
        "Profile the CUDA kernel, fuzz the parser, then open a pull request with regression tests"
    )
    assert decision.target == GENERAL
    assert decision.abstained is True
    assert "spans" in decision.reason


def test_keyword_router_abstains_when_two_domains_tie():
    decision = KeywordRouter(min_margin=5.0).route("Profile this Triton kernel")
    assert decision.abstained is True
    assert "too close to call" in decision.reason


def test_word_boundaries_prevent_substring_false_positives():
    """Without boundaries `spi` matches 'inspired' and `poc` matches 'process'."""
    decision = KeywordRouter().route("I was inspired by this process and its position")
    assert decision.target == GENERAL


def test_strong_terms_outweigh_repeated_weak_terms():
    """One 'cutlass' must beat a pile of 'optimize'/'benchmark'."""
    weak = "optimize optimize optimize benchmark benchmark latency throughput"
    decision = KeywordRouter().route(f"Port the CUTLASS epilogue. {weak}")
    assert decision.target == "cuda"


def test_keyword_router_is_deterministic():
    task = "Debug the FreeRTOS stack overflow"
    assert KeywordRouter().route(task).target == KeywordRouter().route(task).target


def test_scores_are_reported_for_every_domain():
    decision = KeywordRouter().route("Profile this Triton kernel with Nsight")
    assert set(decision.scores) == set(DOMAINS)


# --- model router ------------------------------------------------------------------


def _replier(payload):
    return lambda prompt: payload


def test_model_router_parses_a_clean_reply():
    router = ModelRouter(_replier('{"target": "cuda", "confidence": 0.9, "reason": "kernel work"}'))
    decision = router.route("optimize a kernel")
    assert (decision.target, decision.abstained) == ("cuda", False)


def test_model_router_tolerates_surrounding_prose():
    router = ModelRouter(_replier('Sure!\n{"target": "swe", "confidence": 0.8, "reason": "repo"}\nHope that helps'))
    assert router.route("fix the test").target == "swe"


def test_hallucinated_specialist_becomes_an_abstention():
    """Dispatching to a worker that does not exist is worse than falling back."""
    router = ModelRouter(_replier('{"target": "database", "confidence": 0.99, "reason": "sql"}'))
    decision = router.route("tune a query")
    assert decision.target == GENERAL
    assert decision.abstained is True
    assert "unknown target" in decision.reason


def test_low_confidence_specialist_choice_becomes_an_abstention():
    router = ModelRouter(_replier('{"target": "cuda", "confidence": 0.2, "reason": "maybe"}'))
    decision = router.route("something")
    assert decision.abstained is True
    assert decision.target == GENERAL


def test_low_confidence_threshold_is_configurable():
    router = ModelRouter(_replier('{"target": "cuda", "confidence": 0.5, "reason": "ok"}'), min_confidence=0.4)
    assert router.route("x").target == "cuda"


def test_deliberate_general_is_not_an_abstention():
    """'This is generalist work' is a decision; 'I don't know' is not. They differ."""
    router = ModelRouter(_replier('{"target": "general", "confidence": 0.9, "reason": "broad"}'))
    decision = router.route("plan the offsite")
    assert (decision.target, decision.abstained) == (GENERAL, False)


def test_deliberate_general_survives_a_low_confidence():
    """The confidence gate must not turn a real `general` into an abstention."""
    router = ModelRouter(_replier('{"target": "general", "confidence": 0.1, "reason": "unsure but broad"}'))
    assert router.route("x").abstained is False


@pytest.mark.parametrize(
    "reply",
    ["not json at all", "[1,2,3]", '{"target": "cuda", "confidence": "high"}', "{broken", ""],
)
def test_malformed_replies_abstain_rather_than_raise(reply):
    assert ModelRouter(_replier(reply)).route("task").abstained is True


def test_a_dead_router_degrades_to_the_generalist():
    """The router fronts every request; if it dies the work must still be routable."""

    def boom(prompt):
        raise ConnectionError("endpoint down")

    decision = ModelRouter(boom).route("optimize a kernel")
    assert decision.target == GENERAL
    assert "router call failed" in decision.reason


def test_model_router_abstains_on_empty_input_without_calling_the_model():
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return "{}"

    assert ModelRouter(spy).route("  ").abstained is True
    assert calls == []


def test_out_of_range_confidence_is_clamped_not_rejected():
    router = ModelRouter(_replier('{"target": "cuda", "confidence": 3.0, "reason": "x"}'))
    assert router.route("x").confidence == 1.0


def test_prompt_lists_every_specialist_and_the_fallback():
    prompt = build_prompt("some task")
    for key in DOMAINS:
        assert key in prompt
    assert GENERAL in prompt
    assert "some task" in prompt


def test_parse_decision_is_usable_standalone():
    assert parse_decision('{"target": "cyber", "confidence": 0.95, "reason": "cve"}').target == "cyber"


# --- dataset + evaluation ----------------------------------------------------------


def test_suite_loads_and_is_labeled():
    assert len(SUITE) >= 20
    assert all(e.gold in {*DOMAINS, GENERAL} for e in SUITE)


def test_suite_covers_every_domain_and_the_fallback():
    covered = {e.gold for e in SUITE}
    assert covered == {*DOMAINS, GENERAL}


def test_unknown_gold_label_is_rejected():
    with pytest.raises(RoutingDatasetError, match="unknown gold label"):
        RoutingExample.from_record({"task": "x", "gold": "database"})


def test_example_without_a_task_is_rejected():
    with pytest.raises(RoutingDatasetError, match="no task"):
        RoutingExample.from_record({"gold": "cuda"})


def test_bad_dataset_row_is_reported_with_its_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"task": "ok", "gold": "cuda"}) + "\n{ not json\n")
    with pytest.raises(RoutingDatasetError, match=r"bad\.jsonl:2"):
        list(load_examples(path))


def test_missing_suite_raises():
    with pytest.raises(RoutingDatasetError, match="no such routing suite"):
        load_suite("v999")


def test_keyword_baseline_never_misroutes_on_the_regression_suite():
    """The suite is fitted to the keyword lists, so this is a regression guard only.

    It asserts no *misroutes* rather than a headline accuracy: misrouting is the
    expensive error, and pinning accuracy at 1.00 would make every future dataset
    addition look like a regression.
    """
    metrics = evaluate(KeywordRouter(), SUITE)
    assert metrics.misroute_rate == 0.0


def test_evaluate_on_no_examples_does_not_divide_by_zero():
    assert evaluate(KeywordRouter(), []).examples == 0


def test_per_domain_recall_exposes_a_lazy_majority_router():
    """A router that always says 'swe' must not look good just because swe is common."""

    class AlwaysSWE:
        def route(self, task):
            return RoutingDecision(target="swe", confidence=1.0, reason="always")

    metrics = evaluate(AlwaysSWE(), SUITE)
    assert metrics.specialist_recall["swe"] == 1.0
    assert metrics.specialist_recall["cuda"] == 0.0
    assert metrics.misroute_rate > 0.5


def test_abstaining_router_has_no_misroutes():
    """Always falling back is safe-but-useless: zero misroutes, poor recall."""

    class AlwaysAbstain:
        def route(self, task):
            from hermes.router import abstain

            return abstain("never sure")

    metrics = evaluate(AlwaysAbstain(), SUITE)
    assert metrics.misroute_rate == 0.0
    assert metrics.abstention_rate == 1.0
    assert metrics.specialist_recall["cuda"] == 0.0


def test_confident_general_miss_is_counted_apart_from_abstention():
    class ConfidentGeneral:
        def route(self, task):
            return RoutingDecision(target=GENERAL, confidence=1.0, reason="claims generalist")

    metrics = evaluate(ConfidentGeneral(), SUITE)
    assert metrics.abstention_rate == 0.0
    assert metrics.confident_general_miss_rate > 0.0


def test_compare_rejects_a_candidate_that_only_matches_the_free_baseline():
    baseline = evaluate(KeywordRouter(), SUITE)
    result = compare(baseline, baseline)
    assert result["beats_baseline"] is False
    assert "does not justify its cost" in result["verdict"]


def test_compare_rejects_accuracy_bought_with_extra_misroutes():
    """Trading a cheap error for an expensive one is a regression, whatever accuracy does."""
    from hermes.router.evaluate import RouterMetrics

    baseline = RouterMetrics(0.80, 0.05, 0.15, 0.0, {}, {}, 10)
    candidate = RouterMetrics(0.85, 0.15, 0.00, 0.0, {}, {}, 10)
    result = compare(baseline, candidate)
    assert result["beats_baseline"] is False
    assert "misroutes more" in result["verdict"]


def test_compare_accepts_a_genuine_improvement():
    from hermes.router.evaluate import RouterMetrics

    baseline = RouterMetrics(0.80, 0.10, 0.10, 0.0, {}, {}, 10)
    candidate = RouterMetrics(0.90, 0.05, 0.05, 0.0, {}, {}, 10)
    result = compare(baseline, candidate)
    assert result["beats_baseline"] is True


def test_compare_record_is_json_safe():
    baseline = evaluate(KeywordRouter(), SUITE)
    assert json.loads(json.dumps(compare(baseline, baseline)))["beats_baseline"] is False

import json

import pytest

from hermes.router import (
    GENERAL,
    TIER_KEYWORD,
    TIER_MODEL,
    CascadeRouter,
    KeywordRouter,
    ModelRouter,
    RoutingDecision,
    evaluate,
    load_suite,
)
from hermes.router.base import CROSS_DOMAIN, EMPTY_TASK, NO_EVIDENCE, TOO_CLOSE

SUITE = load_suite("v0")

CLEAR_CUDA = "Profile this Triton kernel with Nsight and fix the bank conflicts"
NO_SIGNAL = "Help me plan the offsite agenda"
CROSS = "Profile the CUDA kernel, fuzz the parser, then open a pull request with regression tests"


class SpyRouter:
    """Records every task it is asked about, so escalation can be counted."""

    def __init__(self, decision=None):
        self.seen = []
        self.decision = decision or RoutingDecision(
            target="cuda", confidence=0.9, reason="spy says cuda", tier=TIER_MODEL
        )

    def route(self, task):
        self.seen.append(task)
        return self.decision


def _tiny(payload):
    """A ModelRouter backed by a canned reply."""
    return ModelRouter(lambda prompt: payload)


# --- the cheap tier keeps the easy cases ------------------------------------------


def test_confident_keyword_decision_does_not_escalate():
    """The whole point: don't pay a model to re-decide what a word match already settled."""
    spy = SpyRouter()
    decision = CascadeRouter(escalate_to=spy).route(CLEAR_CUDA)

    assert decision.target == "cuda"
    assert decision.tier == TIER_KEYWORD
    assert decision.escalated is False
    assert spy.seen == []


def test_uncertain_keyword_decision_escalates():
    spy = SpyRouter()
    decision = CascadeRouter(escalate_to=spy).route(NO_SIGNAL)

    assert spy.seen == [NO_SIGNAL]
    assert decision.escalated is True
    assert decision.tier == TIER_MODEL
    assert decision.target == "cuda"


def test_cross_domain_does_not_escalate():
    """A provably multi-domain task is already answered; a second opinion adds nothing."""
    spy = SpyRouter()
    decision = CascadeRouter(escalate_to=spy).route(CROSS)

    assert decision.reason_code == CROSS_DOMAIN
    assert decision.target == GENERAL
    assert decision.escalated is False
    assert spy.seen == []


def test_empty_task_does_not_escalate():
    spy = SpyRouter()
    decision = CascadeRouter(escalate_to=spy).route("   ")
    assert decision.reason_code == EMPTY_TASK
    assert spy.seen == []


@pytest.mark.parametrize("code", [NO_EVIDENCE, TOO_CLOSE])
def test_only_uncertainty_codes_are_escalatable(code):
    from hermes.router.base import abstain

    assert abstain("x", reason_code=code).escalatable is True


@pytest.mark.parametrize("code", [CROSS_DOMAIN, EMPTY_TASK])
def test_decision_codes_are_not_escalatable(code):
    from hermes.router.base import abstain

    assert abstain("x", reason_code=code).escalatable is False


def test_a_positive_decision_is_never_escalatable():
    assert RoutingDecision(target="cuda", confidence=0.9, reason="x").escalatable is False


def test_too_close_to_call_escalates():
    spy = SpyRouter()
    cascade = CascadeRouter(escalate_to=spy, fast=KeywordRouter(min_margin=99.0))
    cascade.route(CLEAR_CUDA)
    assert spy.seen == [CLEAR_CUDA]


# --- degradation ------------------------------------------------------------------


def test_cascade_without_an_escalation_router_degrades_to_the_free_tier():
    """No tiny router deployed yet is a configuration, not an error."""
    cascade = CascadeRouter()
    assert cascade.has_escalation is False
    assert cascade.route(CLEAR_CUDA).target == "cuda"
    assert cascade.route(NO_SIGNAL).target == GENERAL
    assert cascade.route(NO_SIGNAL).escalated is False


def test_a_dead_tiny_router_still_yields_a_routable_decision():
    def boom(prompt):
        raise ConnectionError("tiny router down")

    decision = CascadeRouter(escalate_to=ModelRouter(boom)).route(NO_SIGNAL)
    assert decision.target == GENERAL
    assert decision.abstained is True
    assert decision.escalated is True


def test_escalation_that_also_abstains_lands_on_the_generalist():
    decision = CascadeRouter(escalate_to=_tiny("not json")).route(NO_SIGNAL)
    assert decision.target == GENERAL
    assert decision.abstained is True
    assert decision.escalated is True


def test_abstained_escalation_reports_the_model_tier_that_actually_decided():
    """Logs must not credit the cheap tier with a call it did not make."""
    decision = CascadeRouter(escalate_to=_tiny("garbage")).route(NO_SIGNAL)
    assert decision.tier == TIER_MODEL


def test_escalated_decision_keeps_the_keyword_scores_for_debuggability():
    decision = CascadeRouter(escalate_to=SpyRouter()).route(NO_SIGNAL)
    assert decision.scores  # why it escalated is still visible
    assert "escalated" in decision.reason


def test_hallucinated_target_from_the_tiny_router_is_contained():
    decision = CascadeRouter(escalate_to=_tiny('{"target": "database", "confidence": 0.99, "reason": "sql"}')).route(
        NO_SIGNAL
    )
    assert decision.target == GENERAL
    assert decision.abstained is True


def test_low_confidence_from_the_tiny_router_is_contained():
    decision = CascadeRouter(escalate_to=_tiny('{"target": "cuda", "confidence": 0.1, "reason": "guessing"}')).route(
        NO_SIGNAL
    )
    assert decision.target == GENERAL


def test_repr_names_both_tiers():
    text = repr(CascadeRouter(escalate_to=SpyRouter()))
    assert "SpyRouter" in text and "KeywordRouter" in text


# --- economics --------------------------------------------------------------------


def _oracle_tiny():
    gold = {e.task: e.gold for e in SUITE}
    calls = []

    def complete(prompt):
        task = prompt.split("Task:\n", 1)[1]
        calls.append(task)
        return json.dumps({"target": gold.get(task, GENERAL), "confidence": 0.9, "reason": "tiny"})

    return ModelRouter(complete), calls


def test_cascade_pays_for_only_the_hard_minority():
    """The cascade's justification is cost: most traffic must never reach the model."""
    tiny, calls = _oracle_tiny()
    metrics = evaluate(CascadeRouter(escalate_to=tiny), SUITE)

    assert len(calls) == len(SUITE) * metrics.escalation_rate
    assert 0.0 < metrics.escalation_rate < 0.5, "free tier should handle the clear majority"


def test_escalation_rate_is_zero_without_an_escalation_router():
    assert evaluate(CascadeRouter(), SUITE).escalation_rate == 0.0


def test_cascade_does_not_misroute_more_than_the_free_tier_alone():
    tiny, _ = _oracle_tiny()
    baseline = evaluate(KeywordRouter(), SUITE)
    cascade = evaluate(CascadeRouter(escalate_to=tiny), SUITE)
    assert cascade.misroute_rate <= baseline.misroute_rate


def test_a_reckless_tiny_router_shows_up_as_misroutes_not_hidden():
    """Escalating to a bad model must degrade the visible metrics, not be absorbed."""
    reckless = SpyRouter(RoutingDecision(target="cuda", confidence=1.0, reason="always cuda", tier=TIER_MODEL))
    metrics = evaluate(CascadeRouter(escalate_to=reckless), SUITE)
    assert metrics.misroute_rate > 0.0


def test_metrics_record_is_json_safe_with_escalation():
    tiny, _ = _oracle_tiny()
    record = evaluate(CascadeRouter(escalate_to=tiny), SUITE).to_record()
    assert json.loads(json.dumps(record))["escalation_rate"] > 0

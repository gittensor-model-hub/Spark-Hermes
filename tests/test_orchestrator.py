"""The Hermes Orchestrator: what work is needed, not who does it."""

import json

import pytest

from hermes.orchestrator import (
    MAX_CONSECUTIVE_INVALID_CALLS,
    REROUTE_COMPLETED,
    REROUTE_DOMAIN_CHANGED,
    REROUTE_INVALID_TOOL_CALLS,
    REROUTE_NO_PROGRESS,
    REROUTE_REPEATED_FAILURE,
    REROUTE_TOOL_UNAVAILABLE,
    REROUTE_VERIFIER_FAILED,
    ExpertSession,
    HandoffState,
    Orchestrator,
    Plan,
    PlanError,
    Subtask,
)
from hermes.router.spec import TaskSpecError

# The worked example from the architecture: optimize a CUDA repo, submit a clean patch.
CUDA_PLAN = Plan(
    request="Optimize this CUDA repository and submit a clean patch.",
    subtasks=(
        Subtask("reproduce", "Reproduce the build and benchmark", "swe", "implement"),
        Subtask("profile", "Profile the GPU workload", "cuda", "debug", depends_on=("reproduce",)),
        Subtask("optimize", "Optimize the kernel", "cuda", "optimize", depends_on=("profile",)),
        Subtask("verify", "Run correctness tests", "cuda", "verify", depends_on=("optimize",)),
        Subtask("review", "Review the final diff", "swe", "review", depends_on=("verify",)),
    ),
)


# --- plans -------------------------------------------------------------------------


def test_plan_orders_subtasks_by_dependency():
    assert CUDA_PLAN.execution_order() == ("reproduce", "profile", "optimize", "verify", "review")


def test_plan_spans_multiple_domains():
    assert set(CUDA_PLAN.domains) == {"swe", "cuda"}


def test_a_cycle_is_rejected_while_it_is_still_a_plan():
    """A plan that cannot finish should fail before a paid run starts, not halfway."""
    with pytest.raises(PlanError, match="dependency cycle"):
        Plan(
            request="r",
            subtasks=(
                Subtask("a", "a", "swe", "implement", depends_on=("b",)),
                Subtask("b", "b", "swe", "implement", depends_on=("a",)),
            ),
        )


def test_unknown_dependency_is_rejected():
    with pytest.raises(PlanError, match="depends on unknown"):
        Plan(request="r", subtasks=(Subtask("a", "a", "swe", "implement", depends_on=("ghost",)),))


def test_duplicate_subtask_ids_are_rejected():
    with pytest.raises(PlanError, match="duplicate subtask_id"):
        Plan(request="r", subtasks=(Subtask("a", "x", "swe", "implement"), Subtask("a", "y", "swe", "debug")))


def test_empty_plan_is_rejected():
    with pytest.raises(PlanError, match="at least one subtask"):
        Plan(request="r", subtasks=())


def test_subtask_taxonomy_is_validated():
    with pytest.raises(TaskSpecError, match="unknown domain"):
        Subtask("a", "a", "astrology", "implement")
    with pytest.raises(TaskSpecError, match="unknown action"):
        Subtask("a", "a", "swe", "vibe")


def test_ready_returns_every_runnable_subtask():
    """Independent work must not be forced into a sequence the plan never required."""
    plan = Plan(
        request="r",
        subtasks=(
            Subtask("root", "root", "swe", "implement"),
            Subtask("left", "left", "swe", "debug", depends_on=("root",)),
            Subtask("right", "right", "cuda", "debug", depends_on=("root",)),
        ),
    )
    assert {s.subtask_id for s in plan.ready({"root"})} == {"left", "right"}


def test_ready_excludes_completed_and_blocked():
    assert [s.subtask_id for s in CUDA_PLAN.ready(set())] == ["reproduce"]
    assert [s.subtask_id for s in CUDA_PLAN.ready({"reproduce"})] == ["profile"]


# --- separation of concerns --------------------------------------------------------


def test_a_plan_names_no_models():
    """The orchestrator decides what work is needed; the router decides who does it.

    An orchestrator that also picks models becomes a second router with worse
    information about worker capability.
    """
    blob = json.dumps(CUDA_PLAN.to_record()).lower()
    for token in ("spark-hermes", "claude", "qwen", "kimi", "gpt-", "agent_module", "model"):
        assert token not in blob, f"plan leaked {token!r}"


def test_assignment_takes_the_module_the_router_chose():
    orchestrator = Orchestrator()
    session = orchestrator.assign(CUDA_PLAN.by_id("profile"), "spark-cuda-worker-v1")
    assert session.agent_module == "spark-cuda-worker-v1"
    assert session.domain == "cuda"


# --- sticky routing ----------------------------------------------------------------


def _session(**kw) -> ExpertSession:
    return ExpertSession(subtask_id="s", agent_module="m", domain="cuda", **kw)


def test_an_expert_keeps_the_subtask_through_ordinary_friction():
    """Changing worker mid-repair discards what the current one established."""
    session = _session()
    session.record_step(failed=True)
    assert session.should_reroute() is None


def test_repeated_failure_triggers_a_handoff():
    session = _session()
    for _ in range(2):
        session.record_step(failed=True)
    assert session.should_reroute() == REROUTE_REPEATED_FAILURE


def test_progress_clears_the_failure_streak():
    """Three tries then success is recovery -- the behaviour we want to keep."""
    session = _session()
    session.record_step(failed=True)
    session.record_step(progressed=True)
    session.record_step(failed=True)
    assert session.should_reroute() is None


def test_consecutive_invalid_tool_calls_trigger_a_handoff():
    session = _session()
    for _ in range(MAX_CONSECUTIVE_INVALID_CALLS):
        session.record_step(invalid_tool_call=True)
    assert session.should_reroute() == REROUTE_INVALID_TOOL_CALLS


def test_a_single_invalid_call_is_forgiven():
    """One is a typo; two in a row is a model that cannot drive the tool."""
    session = _session()
    session.record_step(invalid_tool_call=True)
    session.record_step(progressed=True)
    assert session.should_reroute() is None


def test_no_progress_triggers_a_handoff():
    session = _session()
    for _ in range(5):
        session.record_step()
    assert session.should_reroute() == REROUTE_NO_PROGRESS


def test_verifier_failure_after_a_claim_outranks_everything():
    """The expert said done and the verifier disagreed: its judgement is unreliable."""
    session = _session()
    assert session.should_reroute(verifier_failed_after_claim=True) == REROUTE_VERIFIER_FAILED


def test_domain_change_triggers_a_handoff():
    assert _session().should_reroute(current_domain="firmware") == REROUTE_DOMAIN_CHANGED


def test_missing_tool_triggers_a_handoff():
    assert _session().should_reroute(required_tool_missing=True) == REROUTE_TOOL_UNAVAILABLE


def test_completion_ends_the_session():
    session = _session()
    session.completed = True
    assert session.should_reroute() == REROUTE_COMPLETED


# --- handoff state -----------------------------------------------------------------


def test_handoff_carries_a_summary_not_a_transcript():
    state = HandoffState(
        subtask_id="optimize",
        current_hypothesis="memory bandwidth limited",
        files_changed=("src/attention.cu",),
        tests={"correctness": "pass", "benchmark": "pending"},
        artifacts=("profiles/ncu-before.ncu-rep",),
        next_action="benchmark the fused kernel",
    )
    record = state.to_record()
    assert record["current_hypothesis"] == "memory bandwidth limited"
    assert record["tests"]["benchmark"] == "pending"


def test_context_comes_only_from_direct_dependencies():
    """A subtask three hops downstream does not need intermediate hypotheses."""
    orchestrator = Orchestrator()
    for sid in ("reproduce", "profile", "optimize"):
        orchestrator.assign(CUDA_PLAN.by_id(sid), "m")
        orchestrator.complete(sid, HandoffState(subtask_id=sid, next_action=f"after {sid}"))

    context = orchestrator.context_for(CUDA_PLAN.by_id("verify"))
    assert [c.subtask_id for c in context] == ["optimize"]


def test_completing_an_unassigned_subtask_is_an_error():
    with pytest.raises(PlanError, match="unassigned"):
        Orchestrator().complete("profile")


# --- driving a plan ----------------------------------------------------------------


def test_orchestrator_walks_the_plan_to_completion():
    orchestrator = Orchestrator()
    executed: list[str] = []

    while not orchestrator.is_finished(CUDA_PLAN):
        ready = orchestrator.next_subtasks(CUDA_PLAN)
        assert ready, "plan stalled with work outstanding"
        for subtask in ready:
            orchestrator.assign(subtask, f"module-for-{subtask.domain}")
            executed.append(subtask.subtask_id)
            orchestrator.complete(subtask.subtask_id, HandoffState(subtask_id=subtask.subtask_id))

    assert executed == list(CUDA_PLAN.execution_order())


def test_state_record_is_json_safe():
    orchestrator = Orchestrator()
    orchestrator.assign(CUDA_PLAN.by_id("reproduce"), "spark-swe-worker-v1")
    orchestrator.complete("reproduce", HandoffState(subtask_id="reproduce", next_action="profile next"))
    assert json.loads(json.dumps(orchestrator.to_record()))["completed"] == ["reproduce"]

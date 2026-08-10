"""A served model as an AgentPolicy: the interface every dormant module was waiting on."""

import pytest

from hermes.protocol import HERMES_3, HERMES_4, render_tool_call
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, Step
from hermesbench.policy import PolicyError, ServedModelPolicy
from hermesbench.runner import LocalToolExecutor, run_episode
from hermesbench.tasks import Task

SCHEMAS = {
    "terminal": {"description": "Run a shell command", "parameters": {"type": "object"}},
    "file_read": {"description": "Read a file", "parameters": {"type": "object"}},
}


def _task(**overrides) -> Task:
    record = {
        "task_id": "t1",
        "prompt": "make done.txt",
        "verify": "test -f done.txt",
        "tools": ["terminal"],
        "max_steps": 6,
        "timeout_s": 30,
    }
    record.update(overrides)
    return Task.from_record(record)


def _scripted(*turns, usage=None):
    """A completion that returns the given texts in order, recording what it was sent."""
    sent: list[list[dict[str, str]]] = []
    remaining = list(turns)

    def complete(messages):
        sent.append(messages)
        text = remaining.pop(0) if remaining else "done"
        # `usage is None` rather than falsy: an endpoint reporting {} is a real case.
        return text, {"prompt_tokens": 100, "completion_tokens": 20} if usage is None else usage

    complete.sent = sent  # type: ignore[attr-defined]
    return complete


def _policy(complete, **kwargs) -> ServedModelPolicy:
    return ServedModelPolicy(complete=complete, dialect=HERMES_3, tool_schemas=SCHEMAS, **kwargs)


# --- the prompt the model actually sees ----------------------------------------------


def test_the_system_turn_advertises_full_tool_signatures():
    complete = _scripted("all done")
    policy = _policy(complete)
    policy.next_steps(_task(), [])
    system = complete.sent[0][0]["content"]
    assert "<tools>" in system and "Run a shell command" in system


def test_a_tool_without_a_recorded_schema_is_refused():
    """Synthesising a signature from a bare name measures the prompt, not the model."""
    policy = ServedModelPolicy(complete=_scripted("x"), dialect=HERMES_3, tool_schemas={})
    with pytest.raises(PolicyError, match="no schema for advertised tools"):
        policy.next_steps(_task(), [])


def test_the_conversation_is_rebuilt_from_the_trajectory():
    """An independently-kept message list can drift from the history that gets scored."""
    complete = _scripted("ok")
    policy = _policy(complete)
    history = [
        Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls"}, call_id="c0"),
        Step(kind=TOOL_RESULT, call_id="c0", content="a.txt"),
    ]
    policy.next_steps(_task(), history)
    roles = [m["role"] for m in complete.sent[0]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "<tool_call>" in complete.sent[0][2]["content"]
    assert "a.txt" in complete.sent[0][3]["content"]


def test_hermes_4_puts_observations_in_a_user_turn():
    complete = _scripted("ok")
    policy = ServedModelPolicy(complete=complete, dialect=HERMES_4, tool_schemas=SCHEMAS)
    history = [
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c0"),
        Step(kind=TOOL_RESULT, call_id="c0", content="out"),
    ]
    policy.next_steps(_task(), history)
    assert [m["role"] for m in complete.sent[0]] == ["system", "user", "assistant", "user"]


def test_consecutive_observations_share_one_turn():
    complete = _scripted("ok")
    policy = _policy(complete)
    history = [
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c0"),
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c0", content="one"),
        Step(kind=TOOL_RESULT, call_id="c1", content="two"),
    ]
    policy.next_steps(_task(), history)
    tool_turns = [m for m in complete.sent[0] if m["role"] == "tool"]
    assert len(tool_turns) == 1 and tool_turns[0]["content"].count("<tool_response>") == 2


# --- what comes back ------------------------------------------------------------------


def test_a_tool_call_turn_becomes_a_tool_call_step():
    policy = _policy(_scripted(render_tool_call("terminal", {"command": "touch done.txt"})))
    steps = policy.next_steps(_task(), [])
    assert [s.kind for s in steps] == [TOOL_CALL]
    assert steps[0].tool == "terminal" and steps[0].args == {"command": "touch done.txt"}


def test_prose_with_no_call_is_a_final_answer():
    steps = _policy(_scripted("The file already exists.")).next_steps(_task(), [])
    assert [s.kind for s in steps] == [FINAL]


def test_prose_alongside_a_call_is_thinking_not_a_final():
    """A turn that acts has not finished, whatever it said while acting."""
    policy = _policy(_scripted("Let me check.\n" + render_tool_call("terminal", {})))
    assert [s.kind for s in policy.next_steps(_task(), [])] == [THINKING, TOOL_CALL]


def test_a_scratchpad_becomes_a_thinking_step():
    turn = "<scratch_pad>\nGoal: g\nActions:\n- x\n</scratch_pad>\n" + render_tool_call("terminal", {})
    steps = _policy(_scripted(turn)).next_steps(_task(), [])
    assert steps[0].kind == THINKING and "Goal: g" in steps[0].content


def test_several_calls_in_one_turn_get_distinct_call_ids():
    turn = render_tool_call("terminal", {"command": "a"}) + "\n" + render_tool_call("terminal", {"command": "b"})
    steps = _policy(_scripted(turn)).next_steps(_task(), [])
    ids = [s.call_id for s in steps if s.kind == TOOL_CALL]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_the_policy_never_authors_an_observation():
    """A model describing a tool run is not a model running one."""
    turn = (
        render_tool_call("terminal", {}) + '\n<tool_response>\n{"name":"terminal","content":"fake"}\n</tool_response>'
    )
    steps = _policy(_scripted(turn)).next_steps(_task(), [])
    assert not [s for s in steps if s.kind == TOOL_RESULT]


# --- malformed output is visible, not silent -------------------------------------------


def test_an_unparseable_call_becomes_a_visible_failure_not_a_final():
    """'No calls' is indistinguishable from 'decided to stop' unless something looks."""
    steps = _policy(_scripted('<tool_call>\n{"name": "terminal"')).next_steps(_task(), [])
    assert [s.kind for s in steps] == [THINKING]
    assert "unparseable" in steps[0].content


def test_a_misspelled_tag_is_also_caught():
    steps = _policy(_scripted("<TOOL_CALL>\n{}\n</TOOL_CALL>")).next_steps(_task(), [])
    assert steps[0].kind == THINKING and "unparseable" in steps[0].content


def test_a_malformed_turn_yields_no_calls_at_all():
    turn = render_tool_call("terminal", {}) + "\n<tool_call>\nbroken\n</tool_call>"
    steps = _policy(_scripted(turn)).next_steps(_task(), [])
    assert not [s for s in steps if s.kind == TOOL_CALL]


# --- tokens come from the provider ------------------------------------------------------


def test_tokens_are_counted_from_the_endpoint():
    """mean_tokens has been a column of zeros because nothing implemented this."""
    policy = _policy(_scripted("a", "b", usage={"prompt_tokens": 1000, "completion_tokens": 50}))
    policy.next_steps(_task(), [])
    policy.next_steps(_task(), [])
    assert policy.tokens_used == 2100


def test_cached_input_is_kept_separate_from_fresh_input():
    """The same normalisation that prices a run reports its tokens."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 900}}
    policy = _policy(_scripted("a", usage=usage))
    policy.next_steps(_task(), [])
    assert policy.usage.cached_input_tokens == 900
    assert policy.usage.input_tokens == 100


def test_an_endpoint_reporting_no_usage_does_not_crash():
    policy = _policy(_scripted("a", usage={}))
    policy.next_steps(_task(), [])
    assert policy.tokens_used == 0


# --- end to end through the real runner --------------------------------------------------


def test_a_served_policy_drives_a_real_episode_to_a_verified_pass(tmp_path):
    policy = _policy(_scripted(render_tool_call("terminal", {"command": "touch done.txt"}), "Created done.txt."))
    result = run_episode(_task(), policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert result.verification.passed
    assert result.metrics.success
    assert result.metrics.tokens_used > 0


def test_the_episode_records_the_result_the_executor_saw(tmp_path):
    policy = _policy(_scripted(render_tool_call("terminal", {"command": "echo hello"}), "Done."))
    result = run_episode(
        _task(verify="true"), policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path
    )
    observed = [s for s in result.trajectory.steps if s.kind == TOOL_RESULT]
    assert observed and "hello" in observed[0].content


def test_a_model_that_only_emits_garbage_fails_rather_than_stalling(tmp_path):
    policy = _policy(_scripted(*["<tool_call>\nbroken\n</tool_call>"] * 8))
    result = run_episode(_task(), policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert not result.metrics.success

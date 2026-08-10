from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.metrics import episode_metrics, suite_metrics


def _traj(steps: list[Step]) -> AgentTrajectory:
    return AgentTrajectory(task="t", steps=tuple(steps), success=True)


def _call(call_id: str, tool: str = "terminal") -> Step:
    return Step(kind=TOOL_CALL, tool=tool, args={}, call_id=call_id)


def _result(call_id: str, ok: bool = True) -> Step:
    return Step(kind=TOOL_RESULT, call_id=call_id, content="out", ok=ok)


def test_counts_calls_and_failures():
    m = episode_metrics(
        _traj([_call("c1"), _result("c1", ok=False), _call("c2"), _result("c2"), Step(kind=FINAL)]),
        task_id="t",
        verified_success=True,
    )
    assert (m.tool_calls, m.failed_calls, m.hit_failure) == (2, 1, True)


def test_recovered_requires_a_failure_an_action_and_a_verified_success():
    """All three. A failure followed by no action is not a recovery, however it ended."""
    tried_again = _traj([_call("c1"), _result("c1", ok=False), _call("c2"), _result("c2"), Step(kind=FINAL)])
    assert episode_metrics(tried_again, task_id="t", verified_success=True).recovered is True
    assert episode_metrics(tried_again, task_id="t", verified_success=False).recovered is False


def test_a_failure_followed_by_nothing_is_not_a_recovery():
    """The old rule was `bool(failed) and verified_success`, which asked for no action at
    all: fail, declare done, pass, score as having recovered."""
    gave_up = _traj([_call("c1"), _result("c1", ok=False), Step(kind=FINAL)])
    m = episode_metrics(gave_up, task_id="t", verified_success=True)
    assert m.hit_failure is True
    assert m.recovered is False


def test_a_refusal_of_an_unavailable_tool_is_not_an_observed_failure():
    """The runner synthesises ok=False for a call to a tool the task never offered, so the
    trajectory's tools_available claim stays true. Counting those meant naming a
    nonexistent tool once -- about twenty tokens -- manufactured hit_failure, and
    succeeding afterwards then scored the episode as a recovery."""
    forged = AgentTrajectory(
        task="t",
        success=True,
        tools_available=("terminal",),
        steps=(
            _call("c0", tool="noop"),
            _result("c0", ok=False),
            _call("c1"),
            _result("c1"),
            Step(kind=FINAL),
        ),
    )
    m = episode_metrics(forged, task_id="t", verified_success=True)
    assert m.hit_failure is False
    assert m.recovered is False


def test_a_real_tool_failure_still_counts_when_the_tool_set_is_recorded():
    genuine = AgentTrajectory(
        task="t",
        success=True,
        tools_available=("terminal",),
        steps=(_call("c1"), _result("c1", ok=False), _call("c2"), _result("c2"), Step(kind=FINAL)),
    )
    m = episode_metrics(genuine, task_id="t", verified_success=True)
    assert (m.hit_failure, m.recovered) == (True, True)


def test_a_trajectory_with_no_recorded_tool_set_still_counts_failures():
    """Older rows predate `tools_available`; drop the filter rather than the failure."""
    old = _traj([_call("c1"), _result("c1", ok=False), _call("c2"), _result("c2"), Step(kind=FINAL)])
    assert old.tools_available == ()
    assert episode_metrics(old, task_id="t", verified_success=True).hit_failure is True


def test_clean_success_is_not_counted_as_a_recovery():
    clean = _traj([_call("c1"), _result("c1"), Step(kind=FINAL)])
    m = episode_metrics(clean, task_id="t", verified_success=True)
    assert m.recovered is False
    assert m.hit_failure is False


def test_success_is_the_verified_outcome_not_the_claim():
    """A confident final message over a failing check must score as a failure."""
    confident = AgentTrajectory(
        task="t",
        steps=(_call("c1"), _result("c1"), Step(kind=FINAL, content="All tests pass!")),
        success=True,
    )
    assert episode_metrics(confident, task_id="t", verified_success=False).success is False


def test_self_check_true_when_something_is_observed_after_the_last_edit():
    steps = [_call("c1", "edit"), _result("c1"), _call("c2", "terminal"), _result("c2"), Step(kind=FINAL)]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True)
    assert (m.mutated, m.self_checked) == (True, True)


def test_self_check_false_when_the_edit_is_the_last_thing_that_happens():
    steps = [_call("c1", "terminal"), _result("c1"), _call("c2", "edit"), _result("c2"), Step(kind=FINAL)]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True)
    assert (m.mutated, m.self_checked) == (True, False)


def test_read_only_episode_is_not_eligible_for_self_check():
    """Nothing to check; scoring it either way would measure task type, not behavior."""
    steps = [_call("c1", "file_read"), _result("c1"), Step(kind=FINAL)]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True)
    assert (m.mutated, m.self_checked) == (False, False)


def test_batched_call_issued_before_the_edit_does_not_count_as_a_check():
    """The result of an earlier call can land after the edit; that proves nothing.

    Agents may issue several calls before observing any of them, so position of the
    *result* is not evidence — only a call made after the last mutation counts.
    """
    steps = [
        _call("c1", "terminal"),
        _call("c2", "edit"),
        _result("c1"),
        _result("c2"),
        Step(kind=FINAL),
    ]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True)
    assert (m.mutated, m.self_checked) == (True, False)


def test_batched_call_issued_after_the_edit_does_count():
    steps = [
        _call("c1", "edit"),
        _call("c2", "terminal"),
        _result("c1"),
        _result("c2"),
        Step(kind=FINAL),
    ]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True)
    assert (m.mutated, m.self_checked) == (True, True)


def test_later_call_that_itself_failed_is_not_a_check():
    steps = [_call("c1", "edit"), _result("c1"), _call("c2", "terminal"), _result("c2", ok=False), Step(kind=FINAL)]
    assert episode_metrics(_traj(steps), task_id="t", verified_success=False).self_checked is False


def test_custom_mutating_tools_are_honored():
    steps = [_call("c1", "deploy"), _result("c1"), Step(kind=FINAL)]
    m = episode_metrics(_traj(steps), task_id="t", verified_success=True, mutating_tools=("deploy",))
    assert m.mutated is True


def test_failed_observation_after_an_edit_does_not_count_as_a_check():
    steps = [_call("c1", "edit"), _result("c1"), _call("c2", "terminal"), _result("c2", ok=False), Step(kind=FINAL)]
    assert episode_metrics(_traj(steps), task_id="t", verified_success=False).self_checked is False


def test_steps_and_thinking_are_counted():
    steps = [Step(kind=THINKING, content="hm"), _call("c1"), _result("c1"), Step(kind=FINAL)]
    assert episode_metrics(_traj(steps), task_id="t", verified_success=True).steps == 4


def _episode(**kwargs):
    ok = kwargs.pop("ok", True)
    steps = [_call("c1"), _result("c1", ok=ok)]
    # A failed first call is followed by a second: recovery now requires that the agent
    # actually acted, so a fixture meaning "hit a failure and worked on" must show one.
    if not ok:
        steps += [_call("c2"), _result("c2")]
    steps.append(Step(kind=FINAL))
    return episode_metrics(_traj(steps), task_id=kwargs.pop("task_id", "t"), **kwargs)


def test_empty_suite_scores_zero_without_dividing_by_zero():
    m = suite_metrics([])
    assert m.episodes == 0
    assert m.success_rate == 0.0


def test_suite_success_rate():
    episodes = [_episode(verified_success=True), _episode(verified_success=False)]
    assert suite_metrics(episodes).success_rate == 0.5


def test_tool_efficiency_is_pooled_not_averaged():
    """A 2-call episode must not outweigh a 40-call one."""
    small = episode_metrics(
        _traj([_call("c1"), _result("c1", ok=False), Step(kind=FINAL)]), task_id="a", verified_success=False
    )
    big_steps: list[Step] = []
    for i in range(9):
        big_steps += [_call(f"b{i}"), _result(f"b{i}")]
    big_steps.append(Step(kind=FINAL))
    big = episode_metrics(_traj(big_steps), task_id="b", verified_success=True)

    # 9 ok of 10 total calls pooled = 0.9; a per-episode mean would give 0.5.
    assert suite_metrics([small, big]).tool_efficiency == 0.9


def test_recovery_rate_only_counts_episodes_that_hit_a_failure():
    recovered = _episode(verified_success=True, ok=False)
    not_recovered = _episode(verified_success=False, ok=False)
    clean = _episode(verified_success=True)
    m = suite_metrics([recovered, not_recovered, clean])
    assert m.recovery_eligible == 2
    assert m.recovery_rate == 0.5


def test_recovery_rate_reports_zero_eligible_when_nothing_ever_failed():
    """0.0 with a 0 denominator means 'never had to', not 'never recovers'."""
    m = suite_metrics([_episode(verified_success=True)])
    assert (m.recovery_rate, m.recovery_eligible) == (0.0, 0)


def test_self_check_rate_excludes_read_only_episodes_from_the_denominator():
    checked = episode_metrics(
        _traj([_call("c1", "edit"), _result("c1"), _call("c2"), _result("c2"), Step(kind=FINAL)]),
        task_id="a",
        verified_success=True,
    )
    unchecked = episode_metrics(
        _traj([_call("c1", "edit"), _result("c1"), Step(kind=FINAL)]), task_id="b", verified_success=True
    )
    read_only = episode_metrics(
        _traj([_call("c1", "file_read"), _result("c1"), Step(kind=FINAL)]), task_id="c", verified_success=True
    )
    m = suite_metrics([checked, unchecked, read_only])
    assert m.mutation_eligible == 2
    assert m.self_check_rate == 0.5


def test_means_are_reported():
    a = _episode(verified_success=True, tokens_used=100, wall_time_s=2.0)
    b = _episode(verified_success=True, tokens_used=300, wall_time_s=4.0)
    m = suite_metrics([a, b])
    assert m.mean_tokens == 200
    assert m.mean_wall_time_s == 3.0
    assert m.mean_tool_calls == 1.0


def test_setup_failures_are_excluded_from_every_rate():
    """A broken workspace is infra breakage, not an agent that failed the task."""
    ran_ok = _episode(verified_success=True)
    broke = episode_metrics(
        _traj([_call("c1"), _result("c1"), Step(kind=FINAL)]),
        task_id="b",
        verified_success=False,
        setup_failed=True,
    )
    m = suite_metrics([ran_ok, broke])
    assert m.setup_failures == 1
    assert m.episodes == 1
    assert m.success_rate == 1.0


def test_suite_of_only_setup_failures_reports_no_episodes():
    broke = episode_metrics(
        _traj([_call("c1"), _result("c1"), Step(kind=FINAL)]),
        task_id="b",
        verified_success=False,
        setup_failed=True,
    )
    m = suite_metrics([broke])
    assert (m.episodes, m.setup_failures, m.success_rate) == (0, 1, 0.0)
    assert len(m.per_episode) == 1


def test_to_record_is_json_safe():
    import json

    record = suite_metrics([_episode(verified_success=True)]).to_record()
    assert json.loads(json.dumps(record))["episodes"] == 1


# --- capability categories ---------------------------------------------------------


def _cat_episode(task_id, category, success):
    steps = [_call("c1"), _result("c1"), Step(kind=FINAL)]
    return episode_metrics(_traj(steps), task_id=task_id, verified_success=success, category=category)


def test_success_is_reported_per_category():
    """Pooling hides the case that matters: strong at one, hopeless at another."""
    episodes = [
        _cat_episode("a", "tool_calling", True),
        _cat_episode("b", "tool_calling", True),
        _cat_episode("c", "long_horizon", False),
        _cat_episode("d", "long_horizon", False),
    ]
    metrics = suite_metrics(episodes)
    assert metrics.success_rate == 0.5  # the aggregate describes neither category
    assert metrics.category_success["tool_calling"] == 1.0
    assert metrics.category_success["long_horizon"] == 0.0


def test_category_support_is_reported_so_a_thin_category_is_visible():
    episodes = [_cat_episode("a", "tool_calling", True)] * 1 + [
        _cat_episode(f"b{i}", "terminal_agent", True) for i in range(9)
    ]
    metrics = suite_metrics(episodes)
    assert metrics.category_support == {"tool_calling": 1, "terminal_agent": 9}


def test_uncategorised_episodes_are_left_out_of_the_breakdown():
    """Tasks predating the taxonomy stay loadable and simply do not appear."""
    metrics = suite_metrics([_cat_episode("a", "", True), _cat_episode("b", "tool_calling", True)])
    assert set(metrics.category_success) == {"tool_calling"}
    assert metrics.episodes == 2


def test_category_breakdown_is_json_safe():
    import json

    record = suite_metrics([_cat_episode("a", "self_verification", True)]).to_record()
    assert json.loads(json.dumps(record))["category_success"]["self_verification"] == 1.0


# --- hidden tests / saturation -----------------------------------------------------


def _hidden_episode(task_id, public, hidden):
    steps = [_call("c1"), _result("c1"), Step(kind=FINAL)]
    return episode_metrics(
        _traj(steps),
        task_id=task_id,
        verified_success=public and hidden is not False,
        public_passed=public,
        hidden_passed=hidden,
    )


def test_passing_public_and_failing_hidden_is_overfit():
    """Learned the benchmark rather than the job."""
    assert _hidden_episode("a", True, False).overfit is True


def test_passing_both_is_not_overfit():
    assert _hidden_episode("a", True, True).overfit is False


def test_failing_public_is_not_overfit():
    """Failing outright is incompetence, not benchmark-gaming; conflating them hides both."""
    assert _hidden_episode("a", False, False).overfit is False


def test_a_task_with_no_hidden_tests_cannot_be_overfit():
    assert _hidden_episode("a", True, None).overfit is False


def test_overfit_rate_is_measured_over_public_passers_only():
    episodes = [
        _hidden_episode("a", True, False),
        _hidden_episode("b", True, True),
        _hidden_episode("c", False, False),  # never passed public; not in the denominator
    ]
    metrics = suite_metrics(episodes)
    assert metrics.overfit_rate == 0.5
    assert metrics.hidden_test_episodes == 3


def test_episodes_without_hidden_tests_are_excluded_from_the_saturation_signal():
    metrics = suite_metrics([_hidden_episode("a", True, None)])
    assert metrics.hidden_test_episodes == 0
    assert metrics.overfit_rate == 0.0


# --- Hermes conformance: the wire format is upstream and must not drift -------------------


def _served(text: str):
    from hermes.protocol import HERMES_4
    from hermesbench.policy import ServedModelPolicy

    return ServedModelPolicy(
        complete=lambda messages: (text, None),
        dialect=HERMES_4,
        tool_schemas={"terminal": {"description": "run a command", "parameters": {}}},
    )


def _one_tool_task():
    from hermesbench.tasks import Task

    return Task(task_id="t", prompt="do it", verify="true", tools=("terminal",))


def test_a_malformed_turn_is_counted_not_swallowed():
    """`steps_from_turn` records a parse failure as a THINKING step, which no metric can
    tell apart from real deliberation. Counting it is what makes conformance measurable."""
    policy = _served('<tool_call>\n{"name": "terminal", "arguments":\n</tool_call>')
    task = _one_tool_task()
    assert policy.parse_failures == 0
    policy.next_steps(task, [])
    assert policy.parse_failures == 1
    policy.next_steps(task, [])
    assert policy.parse_failures == 2


def test_a_well_formed_turn_counts_no_failure():
    policy = _served('<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>')
    steps = policy.next_steps(_one_tool_task(), [])
    assert policy.parse_failures == 0
    assert any(s.kind == TOOL_CALL for s in steps)


def test_an_abstention_is_not_a_parse_failure():
    """Answering in words when no tool fits is Hermes's documented behaviour, not a defect."""
    policy = _served("No tool is needed here; the file is already correct.")
    policy.next_steps(_one_tool_task(), [])
    assert policy.parse_failures == 0


def test_protocol_clean_is_the_floor_for_a_trainable_trajectory():
    """A well-formed call and a planning block both cost tokens, so an efficiency score
    with no conformance term rewards drifting off-protocol -- and that drift would be
    trained into the next open-weights checkpoint, breaking the format for everyone."""
    traj = _traj([_call("c1"), _result("c1"), Step(kind=FINAL)])
    clean = episode_metrics(traj, task_id="t", verified_success=True)
    drifted = episode_metrics(traj, task_id="t", verified_success=True, malformed_turns=3)
    assert clean.protocol_clean is True
    assert clean.to_record()["malformed_turns"] == 0
    assert drifted.protocol_clean is False
    assert drifted.to_record()["malformed_turns"] == 3
    assert drifted.to_record()["protocol_clean"] is False


# --- the dialect a run used is recorded, and defaults to the pinned one ------------------------
#
# Measured on the rollout host: the same task and model under `--dialect hermes-3` produced 2
# steps, 0 tool calls, 0 malformed turns and `protocol_clean: true`. The pin records
# `hermes_dialect: hermes-4` with its evidence -- the model's chat template emits `<think>` and
# never `<scratch_pad>` -- and the flag defaulted to hermes-3, so a run that forgot it instructed
# the model in a dialect it does not speak. The model complied, in prose, and every protocol
# metric read clean. Nothing in the record could say which dialect produced it.


def test_the_default_dialect_is_the_pinned_one():
    """A literal default here is a silent whole-run failure: the epoch pins the model, and the
    model's own template decides which blocks it emits."""
    from hermes.base_model import load as load_pin
    from hermesbench.runner import _pinned_dialect

    assert _pinned_dialect() == load_pin().hermes_dialect == "hermes-4"


def test_an_unknown_pinned_dialect_falls_back_rather_than_raising():
    """`--help` must work in a checkout whose pin is missing or malformed, and a broken pin is
    better reported by the run than by argument parsing."""
    from hermesbench.runner import _pinned_dialect

    assert _pinned_dialect(default="hermes-3") in {"hermes-3", "hermes-4"}


def test_the_episode_record_says_which_dialect_produced_it():
    """Beside `verify_digest`, for the same reason. "0 tool calls" means something entirely
    different under a mismatched dialect, and without this field nothing can tell which."""
    from hermesbench.metrics import EpisodeMetrics

    record = EpisodeMetrics(
        task_id="t",
        success=False,
        tool_calls=0,
        failed_calls=0,
        hit_failure=False,
        recovered=False,
        mutated=False,
        self_checked=False,
        tokens_used=1_231,
        wall_time_s=7.0,
        steps=2,
        dialect="hermes-3",
    ).to_record()
    assert record["dialect"] == "hermes-3"
    assert record["malformed_turns"] == 0, "the case that made this necessary: clean, and wrong"
    assert record["protocol_clean"] is True

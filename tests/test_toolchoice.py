"""Tool selection: right tool, right arguments, or correctly declining to call one."""

import json

import pytest

from hermes.protocol import render_tool_call, render_tool_calls
from hermesbench.toolchoice import (
    CORRECT,
    CORRECT_ABSTENTION,
    MALFORMED,
    MISSED,
    UNNECESSARY,
    WRONG_ARGUMENTS,
    WRONG_TOOL,
    CaseResult,
    ToolChoiceCase,
    ToolChoiceError,
    score_response,
    summarize,
)

TOOLS = ("python", "browser", "terminal", "calculator")


def _case(case_id="c", tool="python", args=None):
    return ToolChoiceCase(
        case_id=case_id,
        prompt="Does this optimization improve performance?",
        tools_available=TOOLS,
        expected_tool=tool,
        expected_arguments=args,
    )


def _abstain(case_id="a"):
    return ToolChoiceCase(case_id=case_id, prompt="What is 2+2?", tools_available=TOOLS)


# --- case validity -------------------------------------------------------------------


def test_a_case_with_no_tools_on_offer_is_refused():
    """With nothing to choose from, declining is the only option and measures nothing."""
    with pytest.raises(ToolChoiceError, match="measures nothing"):
        ToolChoiceCase(case_id="c", prompt="p", tools_available=())


def test_expecting_an_unadvertised_tool_is_refused():
    with pytest.raises(ToolChoiceError, match="unpassable"):
        ToolChoiceCase(case_id="c", prompt="p", tools_available=("python",), expected_tool="nsight")


def test_an_abstention_case_cannot_expect_arguments():
    with pytest.raises(ToolChoiceError, match="cannot expect arguments"):
        ToolChoiceCase(case_id="c", prompt="p", tools_available=TOOLS, expected_arguments={"x": 1})


# --- the four outcomes when a call was expected ---------------------------------------


def test_right_tool_right_arguments():
    result = score_response(_case(args={"code": "bench()"}), render_tool_call("python", {"code": "bench()"}))
    assert result.outcome == CORRECT and result.passed


def test_right_tool_wrong_arguments_is_its_own_outcome():
    """The worker understood the task and mis-specified; a different failure from not knowing."""
    result = score_response(_case(args={"code": "bench()"}), render_tool_call("python", {"code": "print(1)"}))
    assert result.outcome == WRONG_ARGUMENTS
    assert "code=" in result.detail


def test_a_missing_argument_is_named():
    result = score_response(_case(args={"code": "x"}), render_tool_call("python", {}))
    assert result.outcome == WRONG_ARGUMENTS and "missing argument" in result.detail


def test_only_the_named_arguments_are_checked():
    """Requiring exhaustive arguments fails a worker for supplying a correct default."""
    result = score_response(_case(args={"code": "x"}), render_tool_call("python", {"code": "x", "timeout": 30}))
    assert result.outcome == CORRECT


def test_wrong_tool():
    result = score_response(_case(), render_tool_call("browser", {}))
    assert result.outcome == WRONG_TOOL


def test_guessing_instead_of_measuring_is_a_missed_call():
    result = score_response(_case(), "It improves performance by about 20%.")
    assert result.outcome == MISSED


def test_hedging_across_tools_is_over_calling_not_selection():
    """A case names one decision; grading only the first call rewards firing all four."""
    result = score_response(_case(), render_tool_calls([("browser", {}), ("python", {})]))
    assert result.outcome == UNNECESSARY


def test_a_correct_call_does_not_excuse_the_ones_beside_it():
    content = render_tool_calls([("python", {}), ("browser", {}), ("terminal", {})])
    assert score_response(_case(), content).outcome == UNNECESSARY


# --- abstention ----------------------------------------------------------------------


def test_declining_when_no_tool_is_needed_is_correct():
    """Hermes's own prompt says to answer in words when the tools are not relevant."""
    result = score_response(_abstain(), "The answer is 4.")
    assert result.outcome == CORRECT_ABSTENTION and result.passed


def test_calling_a_tool_when_none_was_needed_is_over_calling():
    result = score_response(_abstain(), render_tool_call("calculator", {"expr": "2+2"}))
    assert result.outcome == UNNECESSARY and not result.passed


# --- malformed is not abstention ------------------------------------------------------


def test_malformed_output_is_graded_as_malformed_not_as_a_decline():
    result = score_response(_abstain(), "<tool_call>\n{broken\n</tool_call>")
    assert result.outcome == MALFORMED and not result.passed


def test_a_broken_call_beside_a_correct_one_does_not_score_as_correct():
    """Otherwise a model buries a broken call next to a good one and is graded on the good one."""
    content = render_tool_call("python", {}) + "\n<tool_call>\nbroken\n</tool_call>"
    assert score_response(_case(), content).outcome == MALFORMED


# --- aggregation ----------------------------------------------------------------------


def _run(pairs):
    cases = [c for c, _ in pairs]
    results = [score_response(c, r) for c, r in pairs]
    return summarize(cases, results)


def test_wrong_arguments_still_count_as_a_correct_choice():
    report = _run([(_case("c1", args={"code": "x"}), render_tool_call("python", {"code": "y"}))])
    assert report.tool_choice_accuracy == 1.0
    assert report.argument_accuracy == 0.0


def test_argument_accuracy_excludes_wrong_tool_cases():
    """Scoring arguments against a tool that was never right lets the number rise as choice degrades."""
    report = _run(
        [
            (_case("c1", args={"code": "x"}), render_tool_call("python", {"code": "x"})),
            (_case("c2", args={"code": "x"}), render_tool_call("browser", {})),
        ]
    )
    assert report.tool_choice_accuracy == 0.5
    assert report.argument_accuracy == 1.0  # 1 of 1 correctly-chosen, not 1 of 2


def test_a_suite_with_no_abstention_cases_says_it_cannot_see_over_calling():
    """A worker that calls a tool every turn would score perfectly here."""
    report = _run([(_case("c1"), render_tool_call("python", {}))])
    assert report.measures_over_calling is False
    assert report.abstention_cases == 0


def test_a_suite_with_abstention_cases_measures_over_calling():
    report = _run(
        [
            (_case("c1"), render_tool_call("python", {})),
            (_abstain("a1"), render_tool_call("calculator", {})),
            (_abstain("a2"), "4"),
        ]
    )
    assert report.measures_over_calling is True
    assert report.over_call_rate == 1 / 3  # spans the suite, not just abstention cases
    assert report.abstention_accuracy == 0.5
    assert report.tool_choice_accuracy == 1.0  # abstention cases stay out of this denominator


def test_malformed_rate_spans_the_whole_suite():
    report = _run([(_case("c1"), "<tool_call>\nx\n</tool_call>"), (_case("c2"), render_tool_call("python", {}))])
    assert report.malformed_rate == 0.5


def test_an_empty_suite_does_not_divide_by_zero():
    report = summarize([], [])
    assert report.total == 0
    assert report.tool_choice_accuracy == 0.0
    assert report.argument_accuracy == 0.0
    assert report.malformed_rate == 0.0


def test_a_result_for_an_unknown_case_is_refused():
    with pytest.raises(ToolChoiceError, match="unknown case"):
        summarize([_case("c1")], [CaseResult("ghost", CORRECT)])


def test_report_is_json_safe():
    report = _run([(_case("c1"), render_tool_call("python", {})), (_abstain("a1"), "4")])
    record = json.loads(json.dumps(report.to_record()))
    assert record["measures_over_calling"] is True
    assert record["counts"][CORRECT] == 1


# --- regressions: every defect an adversarial pass found --------------------------


def test_an_empty_completion_is_not_a_correct_abstention():
    """A crashed worker returning nothing would otherwise score 100% on an abstention suite."""
    result = score_response(_abstain(), "")
    assert result.outcome == MALFORMED and "empty completion" in result.detail


def test_whitespace_is_not_an_answer_in_natural_language():
    assert score_response(_abstain(), "  \n ").outcome == MALFORMED


def test_a_truncated_call_is_not_a_correct_abstention():
    """A generation cut off mid-call leaves no closing tag and no trace at all."""
    result = score_response(_abstain(), '<tool_call>\n{"name": "calculator", "arguments": {}}')
    assert result.outcome == MALFORMED


def test_a_planned_call_in_the_scratchpad_is_not_an_action():
    """The GOAP block is an intention; grading it as an action is the confusion it prevents."""
    content = (
        "<scratch_pad>\nGoal: bench it\nActions:\n- run it\n"
        + render_tool_call("python", {"code": "bench()"})
        + "\n</scratch_pad>\nI will not run this yet."
    )
    assert score_response(_case(args={"code": "bench()"}), content).outcome == MISSED


def test_a_correct_call_carrying_the_closing_tag_in_an_argument_still_scores_correct():
    """An agent writing code about the protocol is routine on this repo."""
    args = {"code": "print('</tool_call>')"}
    assert score_response(_case(args=args), render_tool_call("python", args)).outcome == CORRECT


def test_a_boolean_does_not_pass_for_an_integer():
    """Python calls 1 == True; JSON does not, and limit=true is a real API error."""
    result = score_response(_case(args={"limit": 1}), render_tool_call("python", {"limit": True}))
    assert result.outcome == WRONG_ARGUMENTS


def test_zero_and_false_are_distinguished_too():
    assert score_response(_case(args={"n": 0}), render_tool_call("python", {"n": False})).outcome == WRONG_ARGUMENTS


def test_a_misspelled_tag_is_not_a_decline():
    assert score_response(_abstain(), '<TOOL_CALL>\n{"name":"t","arguments":{}}\n</TOOL_CALL>').outcome == MALFORMED


def test_a_duplicate_result_is_refused():
    """It would report a one-case suite as a two-case suite."""
    with pytest.raises(ToolChoiceError, match="more than one result"):
        summarize([_case("c1")], [CaseResult("c1", CORRECT), CaseResult("c1", CORRECT)])


def test_a_missing_result_is_refused():
    """A harness that crashed on two of three cases would report the third at 100%."""
    with pytest.raises(ToolChoiceError, match="no result for"):
        summarize([_case("c1"), _case("c2")], [CaseResult("c1", CORRECT)])


def test_duplicate_case_ids_are_refused():
    with pytest.raises(ToolChoiceError, match="duplicate case id"):
        summarize([_case("c1"), _case("c1")], [CaseResult("c1", CORRECT)])


def test_an_outcome_impossible_for_its_case_is_refused():
    """Otherwise the numerator and denominator come from different populations."""
    with pytest.raises(ToolChoiceError, match="impossible for this case"):
        summarize([_abstain("a1")], [CaseResult("a1", CORRECT)])


def test_accuracy_cannot_exceed_one():
    report = summarize([_case("c1"), _abstain("a1")], [CaseResult("c1", CORRECT), CaseResult("a1", CORRECT_ABSTENTION)])
    assert report.tool_choice_accuracy <= 1.0 and report.abstention_accuracy <= 1.0


def test_an_unknown_outcome_raises_a_typed_error_not_a_key_error():
    with pytest.raises(ToolChoiceError, match="unknown outcome"):
        summarize([_case("c1")], [CaseResult("c1", "correct_abstension")])


def test_the_counts_mapping_cannot_be_mutated_behind_the_frozen_facade():
    report = _run([(_case("c1"), render_tool_call("python", {}))])
    with pytest.raises(TypeError):
        report.counts[CORRECT] = 99  # type: ignore[index]


def test_a_boolean_nested_inside_an_argument_does_not_pass_for_an_integer():
    """The same conflation one level down, where a plain == on the dicts would hide it."""
    result = score_response(_case(args={"opts": {"n": 1}}), render_tool_call("python", {"opts": {"n": True}}))
    assert result.outcome == WRONG_ARGUMENTS


def test_a_boolean_nested_in_a_list_is_caught_too():
    result = score_response(_case(args={"xs": [1, 2]}), render_tool_call("python", {"xs": [True, 2]}))
    assert result.outcome == WRONG_ARGUMENTS


def test_nested_arguments_are_exact_match_not_subset():
    """Only the top level is subset-matched; a nested extra key is a different value."""
    result = score_response(_case(args={"f": {"a": 1}}), render_tool_call("python", {"f": {"a": 1, "b": 2}}))
    assert result.outcome == WRONG_ARGUMENTS


def test_argument_accuracy_counts_only_cases_that_pinned_arguments():
    """A suite that pins nothing would otherwise publish near-perfect precision."""
    report = _run([(_case("c1"), render_tool_call("python", {}))])
    assert report.measures_arguments is False
    assert report.argument_accuracy == 0.0
    assert report.to_record()["args_checked"] == 0


def test_an_abstention_only_suite_says_it_cannot_see_tool_choice():
    """A flawless worker would otherwise publish two headline zeros with nothing marking them."""
    report = _run([(_abstain("a1"), "4"), (_abstain("a2"), "5")])
    assert report.measures_tool_choice is False
    assert report.abstention_accuracy == 1.0

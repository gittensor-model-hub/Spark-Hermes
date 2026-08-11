"""Ways to win a promotion without a better model.

Each test here is a run that scores higher and should not ship. The gate's job is not to
compute a comparison -- that part is arithmetic -- but to refuse the comparisons that are not
about the model: a serving change, a shorter run, a harness edit, a model that gives up early,
or a model that got cheap by emitting calls Hermes cannot parse.

The statistics are tested against hand-computable values rather than against the implementation:
a sign test asserted with the same expression that computes it proves the expression is stable,
not that it is right.
"""

import json
import math

import pytest

from hermes.promotion import (
    ALPHA,
    MAX_EFFICIENCY_REGRESSION,
    MAX_MALFORMED_INCREASE,
    MIN_DISCORDANT_TASKS,
    NO,
    YES,
    Episode,
    PromotionError,
    Run,
    Serving,
    binomial_sign_test,
    check_conformance,
    check_evidence,
    check_graders,
    check_serving,
    decide,
    efficiency_terms,
    load_run,
    main,
    paired_success,
)

SERVED = Serving(
    precision="bf16",
    device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    engine="vllm 0.11.2",
    temperature=0.2,
    top_p=0.95,
    max_model_len=32768,
    confidential_computing=False,
)


GRADER = "sha256:" + "1" * 64


def _episodes(
    spec,
    *,
    tokens=50_000,
    tool_calls=10,
    wall=120.0,
    malformed=0,
    max_steps=False,
    grader=GRADER,
    dialect="hermes-4",
):
    """`spec` maps task id to the successes out of a fixed ten attempts."""
    out = []
    for task, passes in spec.items():
        for i in range(10):
            success = i < passes
            out.append(
                Episode(
                    task_id=task,
                    success=success,
                    tokens=tokens,
                    tool_calls=tool_calls,
                    wall_time_s=wall,
                    malformed_turns=malformed,
                    max_steps_hit=max_steps and not success,
                    verify_digest=grader,
                    dialect=dialect,
                )
            )
    return tuple(out)


def _run(model, spec, *, serving=SERVED, **kw):
    return Run(model=model, serving=serving, episodes=_episodes(spec, **kw))


# Eight tasks, seven of which the candidate improves: enough discordant tasks to be decidable.
WEAK = {f"t{i}": 4 for i in range(8)}
STRONG = {f"t{i}": (7 if i < 7 else 4) for i in range(8)}


def test_a_real_improvement_is_promoted():
    decision = decide(_run("m0", WEAK), _run("m1", STRONG))
    assert decision.promote, decision.issues
    assert decision.success.wins == 7 and decision.success.significant


# --- the serving change wearing a model's name ------------------------------------------------


def test_a_precision_change_is_not_a_model_improvement():
    """`harness_digest` covers the prompt, the tools, the executor and the container. It does
    not cover how the weights were served, and NVFP4 against BF16 is a different model in every
    way that matters to a score."""
    nvfp4 = Serving(**{**SERVED.to_record(), "precision": "nvfp4"})
    issues = decide(_run("m0", WEAK), _run("m1", STRONG, serving=nvfp4)).issues
    assert any("not served the same way" in i and "precision" in i for i in issues)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "NVIDIA H100"),
        ("engine", "vllm 0.12.0"),
        ("temperature", 0.7),
        ("top_p", 1.0),
        ("max_model_len", 8192),
        ("confidential_computing", True),
    ],
)
def test_every_serving_field_can_refuse_the_comparison(field, value):
    other = Serving(**{**SERVED.to_record(), field: value})
    assert check_serving(_run("m0", WEAK), _run("m1", STRONG, serving=other))


def test_two_runs_that_recorded_nothing_do_not_compare_as_identical():
    """The failure a defaulted field would produce: both sides blank, so they match, so the gate
    passes on exactly the input it exists to catch."""
    blank = Serving(precision="", device="", engine="", temperature=0.0, top_p=0.0, max_model_len=0)
    assert blank.differences(blank) == [], "blank fields do compare equal, which is why they are refused"
    issues = check_serving(_run("m0", WEAK, serving=blank), _run("m1", STRONG, serving=blank))
    assert len(issues) == 2 and all("Unstated is refused" in i for i in issues)


def test_unstated_confidential_computing_is_refused_rather_than_assumed_off():
    """It encrypts host-device traffic and inflates wall time, which is one of the metrics being
    compared. `None` is not `False`."""
    unstated = Serving(**{**SERVED.to_record(), "confidential_computing": None})
    assert "confidential_computing" in unstated.unstated()


# --- the comparison that is not the same comparison ----------------------------------------------


def test_a_task_only_one_side_ran_is_refused():
    """Dropping it quietly is how a partial rerun becomes a result."""
    issues = check_evidence(_run("m0", {**WEAK, "extra": 0}), _run("m1", STRONG))
    assert any("do not cover the same tasks" in i for i in issues)


def test_different_attempt_counts_are_not_paired_observations():
    m0 = _run("m0", WEAK)
    fewer = Run(model="m1", serving=SERVED, episodes=tuple(e for e in _episodes(STRONG) if e.task_id != "t0")[:70])
    assert check_evidence(m0, fewer)


def test_setup_failures_are_excluded_from_rates_and_reported():
    """Infrastructure breakage counted as failure makes a broken container read as a worse model
    -- but it also means the two sides did not gather the same evidence, so it is not silent."""
    broken = Run(
        model="m1",
        serving=SERVED,
        episodes=_episodes(STRONG)
        + (Episode(task_id="t0", success=False, tokens=0, tool_calls=0, wall_time_s=0.0, setup_failed=True),),
    )
    assert broken.success_rate == _run("m1", STRONG).success_rate
    assert any("setup failures" in i for i in check_evidence(_run("m0", WEAK), broken))


def test_without_manifests_the_harness_check_is_reported_as_not_run():
    """The digest comparison is the thing that refuses a harness edit read as a model change. Not
    having run it is a caveat on the verdict, not a silent pass."""
    decision = decide(_run("m0", WEAK), _run("m1", STRONG))
    assert any("suite and harness digests were not checked" in n for n in decision.notes)


def test_mismatched_manifests_refuse_the_promotion():
    """Delegated to `hermes.harness.comparable` rather than reimplemented: two definitions of
    what makes runs comparable drift, and the looser one decides."""
    from hermes.harness import RunManifest, SuiteDigest, TaskResult

    def manifest(model, digest):
        suite = SuiteDigest(name="s", digest=digest, task_count=1)
        return RunManifest(model=model, suite=suite, harness="h1", results=(TaskResult(task_id="t0", passed=True),))

    m0 = Run(model="m0", serving=SERVED, episodes=_episodes(WEAK), manifest=manifest("m0", "sha256:" + "a" * 64))
    m1 = Run(model="m1", serving=SERVED, episodes=_episodes(STRONG), manifest=manifest("m1", "sha256:" + "b" * 64))
    assert any("different suites" in i for i in decide(m0, m1).issues)


def test_a_grader_change_between_the_runs_is_refused():
    """Not hypothetical. `fix-failing-test` and `verify-speedup-claim` failed ten of ten because
    their published verify scripts invoked a bare `python`, and were fixed later -- 2 of 19 tasks
    moving for a reason neither model caused, which is the size of margin this gate rules on."""
    fixed = _run("m1", STRONG, grader="sha256:" + "2" * 64)
    issues = check_graders(_run("m0", WEAK), fixed)
    assert any("the verify script changed" in i for i in issues)
    assert not decide(_run("m0", WEAK), fixed).promote


def test_unstamped_graders_are_refused_rather_than_matched():
    """Empty on both sides compares equal, so the check would pass on exactly the old logs most
    likely to predate a grader fix."""
    blank = _run("m0", WEAK, grader="")
    issues = check_graders(blank, _run("m1", STRONG, grader=""))
    assert len(issues) == 2 and all("no verify_digest" in i for i in issues)


def test_a_dialect_change_is_refused():
    """The dialect decides how tool calls are spelled, so it decides how many of them parse."""
    assert any("one wire dialect" in i for i in check_graders(_run("m0", WEAK), _run("m1", STRONG, dialect="hermes-3")))


# --- a higher number that is not an improvement ---------------------------------------------------


def test_the_sign_test_matches_values_computed_by_hand():
    """Asserted against 2 * sum(C(n,k)) / 2**n worked out on paper, not against the code."""
    assert binomial_sign_test(0, 0) == 1.0
    assert binomial_sign_test(6, 0) == pytest.approx(2 / 64)
    assert binomial_sign_test(5, 0) == pytest.approx(2 / 32)
    assert binomial_sign_test(7, 1) == pytest.approx(2 * (1 + 8) / 256)
    assert binomial_sign_test(4, 4) == pytest.approx(1.0)


def test_five_discordant_tasks_cannot_reach_significance_at_all():
    """The reason the floor is six rather than a rounder number: at five, the smallest attainable
    two-sided p is 0.0625, so a run reported as "not significant" never could have said yes."""
    assert binomial_sign_test(5, 0) > ALPHA
    assert binomial_sign_test(MIN_DISCORDANT_TASKS, 0) <= ALPHA


def test_an_underpowered_run_says_so_rather_than_reporting_a_negative():
    """ "Not significant" and "could not have been significant" send whoever reads this to
    different next steps: one is a worse model, the other is more repeats."""
    spec = {"t0": 4, "t1": 4, "t2": 4}
    better = {"t0": 8, "t1": 8, "t2": 8}
    decision = decide(_run("m0", spec), _run("m1", better))
    assert decision.success.underpowered
    assert any("underpowered rather than" in i for i in decision.issues)
    assert not decision.promote


def test_a_scattered_improvement_is_refused():
    """Four tasks better, four worse, headline up: the shape a run with identical weights
    produces about as often as not."""
    m0 = _run("m0", {f"t{i}": 5 for i in range(8)})
    m1 = _run("m1", {f"t{i}": (8 if i < 4 else 3) for i in range(8)})
    decision = decide(m0, m1)
    assert decision.success.wins == 4 and decision.success.losses == 4
    assert not decision.promote


def test_an_unchanged_task_is_a_tie_not_a_win():
    """Ties carry no information about direction, which is what makes the sign test's denominator
    the discordant count."""
    result = paired_success(_run("m0", WEAK), _run("m1", WEAK))
    assert (result.wins, result.losses, result.ties) == (0, 0, 8)
    assert result.p_value == 1.0


def test_a_flat_headline_is_refused_even_when_the_paired_test_passes():
    """Both are required. A candidate can win more tasks than it loses while solving no more
    episodes overall, and the thing being promoted is the model people run."""
    # Nine tasks gain one episode each; the tenth loses ten. Nine wins to one loss is p = 0.021,
    # and the suite solves one episode fewer than before.
    m0 = _run("m0", {**{f"t{i}": 5 for i in range(9)}, "t9": 10})
    m1 = _run("m1", {**{f"t{i}": 6 for i in range(9)}, "t9": 0})
    assert decide(m0, m1).success.significant
    assert m1.success_rate < m0.success_rate
    assert any("overall success did not improve" in i for i in decide(m0, m1).issues)


# --- efficiency, which can veto -------------------------------------------------------------------


def test_tokens_are_amortised_over_successes_not_averaged_over_episodes():
    """A model that gives up early on the tasks it would fail shows a lower mean per episode and
    costs more per task completed, which is the number anyone using it pays."""
    thorough = _run("m0", {"t0": 5}, tokens=100_000)
    quitter = Run(
        model="m1",
        serving=SERVED,
        episodes=tuple(
            Episode(task_id="t0", success=i < 5, tokens=100_000 if i < 5 else 1_000, tool_calls=10, wall_time_s=60.0)
            for i in range(10)
        ),
    )
    assert quitter.tokens_per_success < thorough.tokens_per_success
    assert sum(e.tokens for e in quitter.ran) / 10 < sum(e.tokens for e in thorough.ran) / 10


def test_a_token_regression_refuses_an_otherwise_good_promotion():
    """The whole point of a guardrail: success improved and it still does not ship."""
    # Per episode rather than per success: the candidate solves more, so its amortised cost rises
    # by less than the per-episode figure does. 50k -> 150k is +81% amortised, well over the bound.
    decision = decide(_run("m0", WEAK), _run("m1", STRONG, tokens=150_000))
    assert decision.success.significant, "the success side must pass, or this tests the wrong thing"
    assert any("tokens_per_success regressed" in i for i in decision.issues)


def test_a_regression_inside_the_tolerance_is_allowed():
    inside = int(50_000 * (1 + MAX_EFFICIENCY_REGRESSION / 2))  # amortises to less still
    assert decide(_run("m0", WEAK), _run("m1", STRONG, tokens=inside)).promote


@pytest.mark.parametrize(("kw", "metric"), [({"tool_calls": 40}, "tool_calls"), ({"wall": 400.0}, "median_wall_time")])
def test_each_efficiency_term_can_veto(kw, metric):
    issues = decide(_run("m0", WEAK), _run("m1", STRONG, **kw)).issues
    assert any(metric in i and "regressed" in i for i in issues)


def test_running_out_of_budget_without_an_answer_counts_against_the_candidate():
    """No answer, full cost. It is not visible in success rate alone once the failures are
    counted the same as any other failure."""
    m1 = _run("m1", STRONG, max_steps=True)
    assert m1.catastrophic_rate > 0
    assert any("catastrophic_rate regressed" in i for i in decide(_run("m0", WEAK), m1).issues)


def test_a_zero_baseline_is_not_treated_as_an_infinite_regression():
    """A metric that was zero cannot get proportionally worse. Dividing by it would refuse every
    promotion on a term that started perfect."""
    m0 = _run("m0", WEAK, wall=0.0)
    m1 = _run("m1", STRONG, wall=0.0)
    assert decide(m0, m1).promote


def test_a_candidate_that_solves_nothing_has_no_efficiency_to_report():
    """Zero successes makes every per-success figure infinite rather than zero -- reported as
    unknown, since a model that solved nothing is not the cheapest one in the comparison."""
    nothing = _run("m1", {f"t{i}": 0 for i in range(8)})
    assert math.isinf(nothing.tokens_per_success)
    reported = {e["metric"]: e for e in (t.to_record() for t in efficiency_terms(_run("m0", WEAK), nothing))}
    assert reported["tokens_per_success"]["candidate"] is None, "unknown, not zero"


# --- conformance, which an efficiency score would pay to lose ---------------------------------------


def test_drifting_off_protocol_is_refused_even_when_everything_else_improves():
    """A well-formed `<tool_call>` and a reasoning block both cost tokens, so a model that stops
    emitting them scores better on every other metric here."""
    m1 = _run("m1", STRONG, tokens=30_000, malformed=1)
    decision = decide(_run("m0", WEAK), m1)
    assert decision.success.significant
    assert any("malformed turns rose" in i for i in decision.issues)


def test_conformance_is_bounded_absolutely_not_proportionally():
    """At a base rate near zero a ratio is unbounded: one malformed episode against zero is an
    infinite regression, and one in fifty against one in a hundred is not twice as bad."""
    m0 = Run(model="m0", serving=SERVED, episodes=_episodes(WEAK))
    slightly = list(_episodes(STRONG))
    slightly[0] = Episode(**{**slightly[0].__dict__, "malformed_turns": 1})
    m1 = Run(model="m1", serving=SERVED, episodes=tuple(slightly))
    assert m1.malformed_rate - m0.malformed_rate <= MAX_MALFORMED_INCREASE
    assert check_conformance(m0, m1) == []


# --- reading runs off disk --------------------------------------------------------------------------


def test_a_run_file_round_trips_through_the_loader(tmp_path):
    path = tmp_path / "m0.json"
    path.write_text(
        json.dumps(
            {
                "model": "m0",
                "serving": SERVED.to_record(),
                "episodes": [
                    {"metrics": {"task_id": "t0", "success": True, "tokens_used": 5, "tool_calls": 1, "wall_time_s": 1}}
                ],
            }
        ),
        encoding="utf-8",
    )
    run = load_run(path)
    assert run.model == "m0" and run.serving == SERVED and run.episodes[0].tokens == 5


def test_an_episode_without_a_task_id_cannot_be_paired(tmp_path):
    path = tmp_path / "m0.json"
    path.write_text(json.dumps({"model": "m0", "serving": SERVED.to_record(), "episodes": [{"success": True}]}))
    with pytest.raises(PromotionError, match="cannot be paired"):
        load_run(path)


def test_a_run_with_no_episodes_is_refused():
    with pytest.raises(PromotionError, match="nothing to compare"):
        Run(model="m0", serving=SERVED, episodes=())


# --- the CLI, whose exit status is the decision --------------------------------------------------------


def _write(tmp_path, name, spec, **kw):
    path = tmp_path / f"{name}.json"
    run = _run(name, spec, **kw)
    path.write_text(
        json.dumps(
            {
                "model": name,
                "serving": SERVED.to_record(),
                "episodes": [
                    {
                        "task_id": e.task_id,
                        "success": e.success,
                        "tokens_used": e.tokens,
                        "tool_calls": e.tool_calls,
                        "wall_time_s": e.wall_time_s,
                        "malformed_turns": e.malformed_turns,
                        "max_steps_hit": e.max_steps_hit,
                        "verify_digest": e.verify_digest,
                        "dialect": e.dialect,
                    }
                    for e in run.episodes
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_cli_exits_non_zero_when_the_candidate_is_refused(tmp_path, capsys):
    """A promotion script reads this status. A refusal that exits 0 promotes anyway."""
    m0 = _write(tmp_path, "m0", WEAK)
    m1 = _write(tmp_path, "m1", WEAK)
    assert main(["--incumbent", str(m0), "--candidate", str(m1)]) == 1
    assert NO in capsys.readouterr().out


def test_the_cli_exits_zero_on_a_promotion(tmp_path, capsys):
    m0 = _write(tmp_path, "m0", WEAK)
    m1 = _write(tmp_path, "m1", STRONG)
    assert main(["--incumbent", str(m0), "--candidate", str(m1)]) == 0
    assert YES in capsys.readouterr().out


def test_the_json_record_carries_the_reasons(tmp_path, capsys):
    m0 = _write(tmp_path, "m0", WEAK)
    m1 = _write(tmp_path, "m1", WEAK)
    main(["--incumbent", str(m0), "--candidate", str(m1), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert record["verdict"] == NO and record["issues"]
    assert {e["metric"] for e in record["efficiency"]} >= {"tokens_per_success", "median_wall_time_s"}


def test_an_unreadable_run_file_is_an_error_not_a_verdict(tmp_path, capsys):
    missing = tmp_path / "nope.json"
    assert main(["--incumbent", str(missing), "--candidate", str(missing)]) == 1
    assert "not a readable run file" in capsys.readouterr().out


def test_a_run_file_can_name_the_episode_log_the_runner_wrote(tmp_path):
    """The usual case: `hermesbench` writes JSONL, and the run file adds what the log cannot know
    -- nothing in the runner is told what device it is on."""
    log = tmp_path / "m0.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "episode": i,
                    "task_id": e.task_id,
                    "setup_failed": False,
                    "metrics": {
                        "task_id": e.task_id,
                        "success": e.success,
                        "tokens_used": e.tokens,
                        "tool_calls": e.tool_calls,
                        "wall_time_s": e.wall_time_s,
                        "verify_digest": e.verify_digest,
                        "dialect": e.dialect,
                    },
                }
            )
            for i, e in enumerate(_episodes(WEAK))
        ),
        encoding="utf-8",
    )
    run_file = tmp_path / "m0.json"
    run_file.write_text(json.dumps({"model": "m0", "serving": SERVED.to_record(), "episodes_path": "m0.jsonl"}))
    run = load_run(run_file)
    assert len(run.episodes) == 80 and run.episodes[0].verify_digest == GRADER


def test_a_truncated_final_line_does_not_stop_the_log_being_read(tmp_path):
    """The sink documents an unterminated last line as the episode the run died inside."""
    log = tmp_path / "m0.jsonl"
    good = json.dumps({"metrics": {"task_id": "t0", "success": True, "tokens_used": 1, "wall_time_s": 1}})
    log.write_text(good + "\n" + '{"metrics": {"task_id"', encoding="utf-8")
    run_file = tmp_path / "m0.json"
    run_file.write_text(json.dumps({"model": "m0", "serving": SERVED.to_record(), "episodes_path": "m0.jsonl"}))
    assert len(load_run(run_file).episodes) == 1


def test_a_run_file_with_neither_episodes_nor_a_log_is_refused(tmp_path):
    run_file = tmp_path / "m0.json"
    run_file.write_text(json.dumps({"model": "m0", "serving": SERVED.to_record()}))
    with pytest.raises(PromotionError, match="neither an episodes list nor"):
        load_run(run_file)

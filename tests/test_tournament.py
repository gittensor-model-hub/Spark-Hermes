"""Teacher tournaments: fair-fight enforcement, winner selection, artifact generation."""

import json

import pytest

from hermes.router.capability import CapabilityDB
from hermes.router.manifest import RIGHTS_DENIED, RIGHTS_UNKNOWN
from hermes.router.spec import TaskSpec
from hermes.tournament import (
    PAIR_EFFICIENCY,
    REFUSED_LESS_VERIFICATION,
    REFUSED_NARROW,
    REFUSED_WORSE_EVIDENCE,
    WON_CHEAPER,
    WON_DETERMINISTIC,
    WON_EVIDENCE,
    WON_FEWER_CALLS,
    WON_ONLY_PASS,
    CandidateRun,
    Tournament,
    TournamentError,
    Verdict,
    build_artifacts,
    efficiency_margin,
    efficiency_pair,
    select_winner,
    update_capabilities,
)

PIN = "sha256:harness-abc"


def _task(**overrides) -> TaskSpec:
    record = {
        "task_id": "cuda_001",
        "prompt": "Optimize this CUDA kernel",
        "domain": ["cuda"],
        "action": ["optimize"],
        "horizon": "long",
        "verification": "benchmark",
    }
    record.update(overrides)
    return TaskSpec.from_record(record)


def _run(model, passed=True, *, speedup=0.0, deterministic=True, harness=PIN, **kw) -> CandidateRun:
    return CandidateRun(
        model=model,
        trajectory_sha256=f"sha256:{model}",
        verdict=Verdict(
            passed=passed,
            verifier="cuda_correctness_and_speed",
            deterministic=deterministic,
            evidence={"speedup": speedup} if speedup else {},
        ),
        harness_digest=harness,
        **kw,
    )


# --- the fair fight ----------------------------------------------------------------


def test_mixed_harnesses_are_refused():
    """Comparing across harness pins measures model x harness, not model capability."""
    with pytest.raises(TournamentError, match="different harnesses"):
        Tournament(
            task=_task(),
            candidates=(_run("a"), _run("b", harness="sha256:other")),
        )


def test_missing_harness_digest_is_refused():
    with pytest.raises(TournamentError, match="unverifiable"):
        Tournament(task=_task(), candidates=(_run("a", harness=""),))


def test_duplicate_model_entries_are_refused():
    with pytest.raises(TournamentError, match="entered twice"):
        Tournament(task=_task(), candidates=(_run("a"), _run("a")))


def test_empty_field_is_refused():
    with pytest.raises(TournamentError, match="no candidates"):
        Tournament(task=_task(), candidates=())


# --- winner selection --------------------------------------------------------------


def test_single_passing_candidate_wins():
    t = Tournament(task=_task(), candidates=(_run("qwen"), _run("kimi", passed=False)))
    winner, reasons = select_winner(t)
    assert winner.model == "qwen"
    assert WON_ONLY_PASS in reasons


def test_nothing_passing_yields_no_winner():
    t = Tournament(task=_task(), candidates=(_run("a", passed=False), _run("b", passed=False)))
    winner, reasons = select_winner(t)
    assert winner is None
    assert "no_candidate_passed" in reasons


def test_majority_agreement_never_decides():
    """Two agreeing but failing teachers must not outvote one correct one."""
    t = Tournament(
        task=_task(),
        candidates=(_run("a", passed=False), _run("b", passed=False), _run("lone_correct")),
    )
    winner, _ = select_winner(t)
    assert winner.model == "lone_correct"


def test_deterministic_verdict_beats_a_judged_one():
    """Proof beats opinion, however confident the judge was."""
    t = Tournament(
        task=_task(),
        candidates=(_run("judged", deterministic=False), _run("measured", deterministic=True)),
    )
    winner, reasons = select_winner(t)
    assert winner.model == "measured"
    assert WON_DETERMINISTIC in reasons


def test_stronger_measured_evidence_wins():
    t = Tournament(
        task=_task(),
        candidates=(_run("claude", speedup=1.18), _run("qwen", speedup=1.27)),
        primary_evidence="speedup",
    )
    winner, reasons = select_winner(t)
    assert winner.model == "qwen"
    assert WON_EVIDENCE in reasons


def test_fewer_tool_calls_breaks_an_evidence_tie():
    t = Tournament(
        task=_task(),
        candidates=(
            _run("flailer", speedup=1.2, tool_calls=80),
            _run("efficient", speedup=1.2, tool_calls=12),
        ),
        primary_evidence="speedup",
    )
    winner, reasons = select_winner(t)
    assert winner.model == "efficient"
    assert WON_FEWER_CALLS in reasons


def test_recovery_breaks_a_remaining_tie():
    t = Tournament(
        task=_task(),
        candidates=(
            _run("clean", speedup=1.2, tool_calls=10, cost=1.0),
            _run("recovered", speedup=1.2, tool_calls=10, cost=1.0, hit_failure=True, recovered=True),
        ),
        primary_evidence="speedup",
    )
    winner, _ = select_winner(t)
    assert winner.model == "recovered"


def test_selection_is_deterministic_on_a_total_tie():
    a = Tournament(task=_task(), candidates=(_run("zeta"), _run("alpha")))
    b = Tournament(task=_task(), candidates=(_run("alpha"), _run("zeta")))
    assert select_winner(a)[0].model == select_winner(b)[0].model == "alpha"


# --- artifacts ---------------------------------------------------------------------


def test_one_tournament_produces_sft_dpo_and_router_data():
    t = Tournament(
        task=_task(),
        candidates=(
            _run("qwen3.8-max", speedup=1.27),
            _run("claude-fable-5", speedup=1.18),
            _run("kimi-k3", passed=False),
        ),
        primary_evidence="speedup",
    )
    artifacts = build_artifacts(t)

    assert artifacts.winner.model == "qwen3.8-max"
    assert artifacts.sft_trajectory_sha256 == "sha256:qwen3.8-max"
    # DPO is success vs failure only -- not winner vs the other passing candidate.
    assert [p.rejected_model for p in artifacts.dpo_pairs] == ["kimi-k3"]
    assert len(artifacts.router_example["outcomes"]) == 3


def test_losing_trajectories_survive_in_the_router_example():
    """Losers are router, critic and DPO data; discarding them throws away the signal."""
    t = Tournament(task=_task(), candidates=(_run("w"), _run("l", passed=False)))
    outcomes = build_artifacts(t).router_example["outcomes"]
    assert {o["model"] for o in outcomes} == {"w", "l"}


def test_split_decision_is_flagged():
    """Split decisions discriminate between teachers; unanimous ones mostly don't."""
    split = Tournament(task=_task(), candidates=(_run("a"), _run("b", passed=False)))
    unanimous = Tournament(task=_task(), candidates=(_run("a"), _run("b")))
    assert build_artifacts(split).is_split_decision is True
    assert build_artifacts(unanimous).is_split_decision is False


def test_unanimous_failure_produces_router_data_but_no_training_data():
    t = Tournament(task=_task(), candidates=(_run("a", passed=False), _run("b", passed=False)))
    artifacts = build_artifacts(t)
    assert artifacts.sft_trajectory_sha256 is None
    assert artifacts.dpo_pairs == ()
    assert artifacts.router_example["winner"] is None
    assert len(artifacts.router_example["outcomes"]) == 2


@pytest.mark.parametrize("rights", [RIGHTS_DENIED, RIGHTS_UNKNOWN])
def test_a_winner_without_rights_produces_no_training_data(rights):
    """Fails closed: unknown rights are not approved rights."""
    t = Tournament(task=_task(), candidates=(_run("restricted", training_rights=rights),))
    artifacts = build_artifacts(t)
    assert artifacts.winner.model == "restricted"
    assert artifacts.sft_trajectory_sha256 is None
    assert "restricted" in artifacts.withheld_for_rights


def test_a_loser_without_rights_is_excluded_from_dpo_only():
    """Its outcome still informs the router -- that it failed is a fact about the world."""
    t = Tournament(
        task=_task(),
        candidates=(_run("ok"), _run("restricted", passed=False, training_rights=RIGHTS_DENIED)),
    )
    artifacts = build_artifacts(t)
    assert artifacts.dpo_pairs == ()
    assert "restricted" in artifacts.withheld_for_rights
    assert any(o["model"] == "restricted" for o in artifacts.router_example["outcomes"])


def test_artifacts_are_json_safe():
    t = Tournament(task=_task(), candidates=(_run("a"), _run("b", passed=False)))
    assert json.loads(json.dumps(build_artifacts(t).to_record()))["task_id"] == "cuda_001"


# --- capability feedback -----------------------------------------------------------


def test_tournaments_accumulate_into_the_capability_matrix():
    tournaments = [
        Tournament(task=_task(task_id=f"t{i}"), candidates=(_run("qwen"), _run("kimi", passed=False)))
        for i in range(10)
    ]
    db = update_capabilities(CapabilityDB(), tournaments)
    assert db.get("qwen", "cuda.optimize.long", harness=PIN).attempts == 10
    assert db.get("qwen", "cuda.optimize.long", harness=PIN).verified_successes == 10
    assert db.get("kimi", "cuda.optimize.long", harness=PIN).verified_successes == 0
    assert db.estimate("qwen", "cuda.optimize.long", harness=PIN) > db.estimate(
        "kimi", "cuda.optimize.long", harness=PIN
    )


def test_capability_updates_accumulate_across_calls():
    def batch(tag):
        return [Tournament(task=_task(task_id=tag), candidates=(_run("qwen"),))]

    db = update_capabilities(CapabilityDB(), batch("a"))
    db = update_capabilities(db, batch("b"))
    assert db.get("qwen", "cuda.optimize.long", harness=PIN).attempts == 2


def test_records_from_different_harnesses_never_merge():
    """The same guard as the fair-fight rule, one level up."""
    a = Tournament(task=_task(), candidates=(_run("qwen", harness="sha256:h1"),))
    b = Tournament(task=_task(), candidates=(_run("qwen", harness="sha256:h2"),))
    db = update_capabilities(CapabilityDB(), [a, b])
    assert db.get("qwen", "cuda.optimize.long", harness="sha256:h1").attempts == 1
    assert db.get("qwen", "cuda.optimize.long", harness="sha256:h2").attempts == 1


def test_recovery_rate_uses_only_episodes_that_hit_a_failure():
    t = Tournament(
        task=_task(),
        candidates=(
            _run("m", hit_failure=True, recovered=True),
            _run("n", hit_failure=False, recovered=False),
        ),
    )
    db = update_capabilities(CapabilityDB(), [t])
    assert db.get("m", "cuda.optimize.long", harness=PIN).recovery_rate == 1.0
    # Never hit a failure, so there was nothing to recover from.
    assert db.get("n", "cuda.optimize.long", harness=PIN).recovery_rate == 0.0


# --- an unpriced run is not a cheap one --------------------------------------------


def test_the_cost_tiebreak_does_not_run_on_a_partially_priced_field():
    """An unpriced run arriving as 0.0 used to beat every priced rival, take the SFT slot,
    and push the model it 'beat' into the DPO rejections -- for being unknown."""
    tournament = Tournament(
        task=_task(),
        candidates=(
            CandidateRun(
                model="priced",
                trajectory_sha256="a",
                verdict=Verdict(passed=True, verifier="tests"),
                harness_digest=PIN,
                tool_calls=3,
                cost=1.50,
            ),
            CandidateRun(
                model="unpriced",
                trajectory_sha256="b",
                verdict=Verdict(passed=True, verifier="tests"),
                harness_digest=PIN,
                tool_calls=3,
                cost=None,
            ),
        ),
    )
    winner, reasons = select_winner(tournament)
    assert WON_CHEAPER not in reasons
    assert winner is not None and winner.model == "priced"  # falls through to stable order


def test_cost_still_decides_when_every_candidate_has_a_price():
    tournament = Tournament(
        task=_task(),
        candidates=(
            CandidateRun(
                model="dear",
                trajectory_sha256="a",
                verdict=Verdict(passed=True, verifier="tests"),
                harness_digest=PIN,
                tool_calls=3,
                cost=9.0,
            ),
            CandidateRun(
                model="cheap",
                trajectory_sha256="b",
                verdict=Verdict(passed=True, verifier="tests"),
                harness_digest=PIN,
                tool_calls=3,
                cost=1.0,
            ),
        ),
    )
    winner, reasons = select_winner(tournament)
    assert winner is not None and winner.model == "cheap"
    assert WON_CHEAPER in reasons


def test_an_unpriced_run_serializes_as_null_not_zero():
    """A router example reporting an unpriced run as free teaches the same lie."""
    run = CandidateRun(model="m", trajectory_sha256="a", verdict=Verdict(passed=True, verifier="v"), harness_digest="h")
    assert run.to_record()["cost"] is None


# --- success-vs-success preferences, and the thing they can destroy ------------------


def _passing(model, *, calls=30, wall=100.0, cost=None, self_checked=True, evidence=None, sha=None):
    return CandidateRun(
        model=model,
        trajectory_sha256=sha or f"sha-{model}",
        verdict=Verdict(passed=True, verifier="tests", evidence=evidence or {}),
        harness_digest=PIN,
        tool_calls=calls,
        wall_time_s=wall,
        cost=cost,
        self_checked=self_checked,
    )


def _field(*candidates, primary_evidence=""):
    return Tournament(task=_task(), candidates=tuple(candidates), primary_evidence=primary_evidence)


def test_efficiency_pairs_are_off_by_default():
    """The lesson they teach is one step removed from correctness."""
    tournament = _field(_passing("lean", calls=31), _passing("verbose", calls=67))
    assert all(
        p.pair_type == WON_ONLY_PASS or p.pair_type == "success_vs_failure"
        for p in build_artifacts(tournament).dpo_pairs
    )
    assert build_artifacts(tournament).dpo_pairs == ()


def test_a_clear_efficiency_win_becomes_a_pair_with_its_margin():
    tournament = _field(_passing("lean", calls=31, wall=22.0), _passing("verbose", calls=67, wall=48.0))
    artifacts = build_artifacts(tournament, prefer_on_efficiency=True)
    pair = next(p for p in artifacts.dpo_pairs if p.pair_type == PAIR_EFFICIENCY)
    assert pair.chosen_model == "lean" and pair.rejected_model == "verbose"
    assert pair.margin is not None and pair.margin.tool_efficiency > 0.5


def test_a_pair_whose_winner_verified_less_is_refused():
    """Otherwise 'used fewer tools' and 'skipped the check' are the same gradient, and the
    pair teaches a model that already knows how to verify to stop bothering."""
    tournament = _field(
        _passing("fast", calls=10, self_checked=False),
        _passing("careful", calls=40, self_checked=True),
    )
    artifacts = build_artifacts(tournament, prefer_on_efficiency=True)
    assert not [p for p in artifacts.dpo_pairs if p.pair_type == PAIR_EFFICIENCY]
    assert ("careful", REFUSED_LESS_VERIFICATION) in artifacts.refused_pairs


def test_a_candidate_that_measured_worse_cannot_be_preferred():
    """Quicker to produce but slower to run is not the better kernel. select_winner
    already narrows on evidence, so this guards direct callers pairing two candidates
    the selector never chose between."""
    tournament = _field(
        _passing("quick", calls=10, evidence={"speedup": 1.1}),
        _passing("slow_to_write", calls=40, evidence={"speedup": 2.4}),
        primary_evidence="speedup",
    )
    pair, reason = efficiency_pair(tournament, tournament.candidates[0], tournament.candidates[1], min_margin=0.15)
    assert pair is None and reason == REFUSED_WORSE_EVIDENCE


def test_the_selector_never_hands_a_worse_measured_winner_to_the_pair_builder():
    tournament = _field(
        _passing("quick", calls=10, evidence={"speedup": 1.1}),
        _passing("strong", calls=40, evidence={"speedup": 2.4}),
        primary_evidence="speedup",
    )
    artifacts = build_artifacts(tournament, prefer_on_efficiency=True)
    assert artifacts.winner is not None and artifacts.winner.model == "strong"
    assert REFUSED_WORSE_EVIDENCE not in dict(artifacts.refused_pairs).values()


def test_a_near_tie_is_dropped_as_noise():
    """A pair asserting a difference that was not there trains confidence in it."""
    tournament = _field(_passing("a", calls=30, wall=100.0), _passing("b", calls=31, wall=101.0))
    artifacts = build_artifacts(tournament, prefer_on_efficiency=True)
    assert not [p for p in artifacts.dpo_pairs if p.pair_type == PAIR_EFFICIENCY]
    assert REFUSED_NARROW in dict(artifacts.refused_pairs).values()


def test_cost_joins_the_margin_only_when_both_runs_are_priced():
    """Comparing a priced run against an unpriced one scores the unmeasured as free."""
    both = efficiency_margin(_passing("a", cost=1.0), _passing("b", cost=4.0))
    one = efficiency_margin(_passing("a", cost=1.0), _passing("b", cost=None))
    assert both.cost is not None and "cost" in both.components
    assert one.cost is None and "cost" not in one.components


def test_an_unpriced_pair_is_not_diluted_toward_a_tie():
    """Averaging over a fixed denominator would make every unpriced comparison look even."""
    priced = efficiency_margin(
        _passing("a", calls=10, wall=10.0, cost=1.0), _passing("b", calls=40, wall=40.0, cost=4.0)
    )
    unpriced = efficiency_margin(_passing("a", calls=10, wall=10.0), _passing("b", calls=40, wall=40.0))
    assert unpriced.overall == pytest.approx(priced.overall)


def test_margin_components_are_reported_apart():
    """A pair won on evidence and a pair won on wall time are different lessons."""
    margin = efficiency_margin(
        _passing("a", calls=10, wall=10.0, evidence={"speedup": 3.0}),
        _passing("b", calls=40, wall=40.0, evidence={"speedup": 1.0}),
        primary_evidence="speedup",
    )
    assert set(margin.components) == {"evidence", "tool_efficiency", "wall_time"}
    assert margin.to_record()["overall"] > 0


def test_success_vs_failure_pairs_still_carry_no_margin():
    """The verifier decided it; there is nothing to weigh."""
    tournament = _field(
        _passing("winner"),
        CandidateRun(
            model="loser",
            trajectory_sha256="sha-loser",
            verdict=Verdict(passed=False, verifier="tests"),
            harness_digest=PIN,
        ),
    )
    pair = build_artifacts(tournament).dpo_pairs[0]
    assert pair.pair_type == "success_vs_failure" and pair.margin is None


def test_refusals_are_recorded_so_an_even_field_is_distinguishable():
    """'No efficiency pairs' and 'every candidate was a near-tie' are different facts."""
    tournament = _field(_passing("a", calls=30), _passing("b", calls=31))
    record = build_artifacts(tournament, prefer_on_efficiency=True).to_record()
    assert json.loads(json.dumps(record))["refused_pairs"]


def test_efficiency_artifacts_are_json_safe():
    tournament = _field(_passing("lean", calls=10, cost=1.0), _passing("verbose", calls=60, cost=6.0))
    record = json.loads(json.dumps(build_artifacts(tournament, prefer_on_efficiency=True).to_record()))
    pair = next(p for p in record["dpo_pairs"] if p["pair_type"] == PAIR_EFFICIENCY)
    assert "cost" in pair["margin"]["components"]

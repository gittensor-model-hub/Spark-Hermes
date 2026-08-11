"""Searching the surface space, and not being fooled by the search.

The module exists for one reason and this file has to demonstrate it: a search over candidates is
a machine for manufacturing winners. `hermesbench.run_suite` already names the shape for repeats --
"keeping only the best would turn the pass rate into a best-of-k order statistic" -- and choosing
between surfaces is the same hazard one level up.

The simulation below makes it concrete. Every candidate is drawn from an identical distribution,
so there is nothing to find, and a screening leader still appears in roughly nine searches out of
ten. Confirmation on fresh episodes removes all of them.

`run` is injected throughout: the selection handling, the ablation split and the leader rule need
no served model, and they are the parts most likely to be wrong.
"""

import random
from statistics import median

import pytest

from hermes.acceptance import Arm
from miner.search import (
    Candidate,
    SearchError,
    ablations,
    leader_of,
    read_surface,
    render,
    search,
    split_rules,
)

SKILL = """---
name: p
description: d
---

# P

1. **A.** first rule
2. **B.** second rule
3. **C.** third rule
4. **D.** fourth rule
"""


@pytest.fixture
def surface(tmp_path):
    root = tmp_path / "surface"
    (root / "skills" / "p").mkdir(parents=True)
    (root / "SOUL.md").write_text("# Operating identity\nOne call per turn.\n", encoding="utf-8")
    (root / "skills" / "p" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    return root


def _arm(passes, tokens):
    n = len(tokens)
    return Arm(passes=passes, attempts=n, tokens=tuple(tokens), tool_calls=tuple([11] * n))


def _identical_runs(seed, *, mean=78_000, cv=0.235, rate=0.4, n=10):
    """Every candidate drawn from one distribution. Any leader is the selection, not a surface."""
    rng = random.Random(seed)

    def run(candidate, where):
        tokens = [max(1, int(rng.gauss(mean, mean * cv))) for _ in range(n)]
        passes = sum(1 for _ in range(n) if rng.random() < rate)
        return _arm(passes, tokens)

    return run


# --- the ablation split ---------------------------------------------------------------------------


def test_rules_are_split_on_numbered_blocks():
    assert len(split_rules(SKILL)) == 4
    assert split_rules(SKILL)[0].startswith("1. **A.**")


def test_the_candidates_are_the_full_surface_and_one_drop_per_rule(surface):
    names = [c.name for c in ablations(read_surface(surface))]
    assert names == ["full", "drop-1", "drop-2", "drop-3", "drop-4"]


def test_each_ablation_removes_exactly_one_rule(surface):
    files = read_surface(surface)
    for candidate in ablations(files)[1:]:
        text = candidate.files["skills/p/SKILL.md"]
        assert len(split_rules(text)) == 3
        assert candidate.dropped


def test_a_single_rule_skill_is_not_ablated(tmp_path):
    """Removing the only rule leaves a skill that says nothing. That is the surface without the
    skill, which is a different experiment from ablating a rule."""
    root = tmp_path / "s"
    (root / "skills" / "p").mkdir(parents=True)
    (root / "SOUL.md").write_text("# id\n", encoding="utf-8")
    (root / "skills" / "p" / "SKILL.md").write_text("---\nname: p\ndescription: d\n---\n# P\n\n1. Only rule\n", "utf-8")
    assert [c.name for c in ablations(read_surface(root))] == ["full"]


def test_a_surface_with_no_skill_is_refused(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "SOUL.md").write_text("# id\n", encoding="utf-8")
    with pytest.raises(SearchError, match="no SKILL.md"):
        ablations(read_surface(root))


# --- the leader rule -------------------------------------------------------------------------------


def test_correctness_ranks_before_tokens():
    """Mirrors `acceptance.decide`: a candidate that is cheaper and wrong is not a candidate, so a
    combined score able to trade one for the other would rank something the gate refuses."""
    base = type("M", (), {"pass_rate": 0.4, "median_tokens": 78_000})()
    cheap_and_wrong = type("M", (), {"pass_rate": 0.2, "median_tokens": 10_000, "candidate": Candidate("a", {})})()
    correct_and_dearer = type("M", (), {"pass_rate": 0.6, "median_tokens": 90_000, "candidate": Candidate("b", {})})()
    assert leader_of([cheap_and_wrong, correct_and_dearer], base).candidate.name == "b"


def test_no_leader_when_nothing_beat_the_baseline():
    """A search that always names a winner reports the luckiest draw when there was nothing to
    find."""
    base = type("M", (), {"pass_rate": 0.8, "median_tokens": 50_000})()
    worse = type("M", (), {"pass_rate": 0.4, "median_tokens": 90_000, "candidate": Candidate("a", {})})()
    assert leader_of([worse], base) is None


# --- the trap this module exists for -----------------------------------------------------------------


def test_a_leader_appears_from_pure_noise_most_of_the_time(surface, tmp_path):
    """Every candidate identical, so there is nothing to find. A screening leader still appears in
    the large majority of searches -- which is the whole reason the screening column must not be
    read as a result."""
    leaders = 0
    trials = 30
    for seed in range(trials):
        result = search(surface=surface, run=_identical_runs(seed), workspace=tmp_path / f"w{seed}", repeats=10)
        leaders += result.leader is not None
    assert leaders > trials * 0.5, f"only {leaders}/{trials}; the simulation is not exercising the effect"


def test_confirmation_removes_the_noise_leaders(surface, tmp_path):
    """The measurement that makes the module worth its complexity: across searches over identical
    candidates, no screening leader survives an independent sample with an interval above zero."""
    survived = 0
    confirmed = 0
    for seed in range(30):
        result = search(surface=surface, run=_identical_runs(seed), workspace=tmp_path / f"w{seed}", repeats=10)
        if result.confirmed_interval is None:
            continue
        confirmed += 1
        survived += result.confirmed_interval[0] > 0
    assert confirmed > 10, "the simulation produced too few confirmations to say anything"
    assert survived == 0, f"{survived} of {confirmed} noise leaders survived confirmation"


def test_a_real_improvement_does_survive_confirmation(surface, tmp_path):
    """The other direction, or the check above is satisfied by a confirmation step that rejects
    everything. `drop-2` is genuinely cheaper here and it clears."""
    rng = random.Random(11)

    def run(candidate, where):
        cheaper = candidate.name == "drop-2"
        mean = 40_000 if cheaper else 78_000
        tokens = [max(1, int(rng.gauss(mean, mean * 0.10))) for _ in range(10)]
        return _arm(9 if cheaper else 4, tokens)

    result = search(surface=surface, run=run, workspace=tmp_path / "real", repeats=10)
    assert result.leader is not None and result.leader.candidate.name == "drop-2"
    assert result.confirmed_interval is not None and result.confirmed_interval[0] > 0
    assert "survived the search" in render(result, repeats=10)


def test_the_confirmation_uses_fresh_episodes(surface, tmp_path):
    """Re-using the screening episodes would confirm nothing: the same draw cannot be independent
    evidence about itself."""
    seen = []

    def run(candidate, where):
        seen.append((candidate.name, where.name))
        return _arm(5, [70_000] * 10) if candidate.name == "drop-1" else _arm(4, [78_000] * 10)

    search(surface=surface, run=run, workspace=tmp_path / "fresh", repeats=10)
    assert ("drop-1", "screen") in seen
    assert ("drop-1", "confirm") in seen, "the leader must be re-run in its own workspace"


# --- what the report has to say --------------------------------------------------------------------


def test_selection_pressure_is_reported(surface, tmp_path):
    """One candidate is a measurement; twenty is a tournament, and a reader needs to know which."""
    result = search(surface=surface, run=_identical_runs(3), workspace=tmp_path / "w", repeats=10)
    assert result.selection_pressure == 5
    assert result.to_record()["selection_pressure"] == 5
    assert result.to_record()["screening_numbers_are_biased_by_selection"] is True


def test_the_rendered_report_warns_about_the_screening_column(surface, tmp_path):
    result = search(surface=surface, run=_identical_runs(5), workspace=tmp_path / "w", repeats=10)
    text = render(result, repeats=10)
    assert "screening column is biased" in text
    assert "Believe the" in text


def test_episodes_spent_counts_the_confirmation_too(surface, tmp_path):
    """GPU time is the budget, and a search that reports only its screening cost understates it by
    the confirmation run."""

    def run(candidate, where):
        return _arm(9 if candidate.name == "drop-1" else 4, [40_000 if candidate.name == "drop-1" else 78_000] * 10)

    result = search(surface=surface, run=run, workspace=tmp_path / "w", repeats=10)
    assert result.confirmation is not None
    assert result.episodes_spent == 60, "5 candidates plus one confirmation, at 10 episodes each"


def test_capping_the_candidate_list_is_reported_not_silent(surface, tmp_path):
    """A search that silently truncates its own space reports "the best of the ablations" while
    having measured some of them."""
    result = search(surface=surface, run=_identical_runs(1), workspace=tmp_path / "w", repeats=10, max_candidates=3)
    assert result.selection_pressure == 3
    assert any("were not measured" in n for n in result.notes)


def test_an_empty_surface_is_refused(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SearchError, match="is empty"):
        search(surface=empty, run=lambda c, w: _arm(1, [1]), workspace=tmp_path / "w", repeats=1)


def test_the_median_helper_agrees_with_statistics(surface, tmp_path):
    result = search(surface=surface, run=_identical_runs(2), workspace=tmp_path / "w", repeats=10)
    assert result.baseline.median_tokens == median(result.baseline.arm.tokens)

"""Task identity: keeping an endless generator from flooding the arena with near-copies."""

import json

from hermesbench.identity import (
    BUCKET_SATURATED,
    EXECUTION_DUPLICATE,
    OBJECTIVE_DUPLICATE,
    CoverageMatrix,
    DuplicateIndex,
    deduplicate,
    identity_of,
    jaccard,
    normalize_objective,
)


def _task(task_id="t", prompt="Fix the failing test", verify="pytest -q", setup="echo a", tools=("terminal",)):
    return {"task_id": task_id, "prompt": prompt, "verify": verify, "setup": setup, "tools": list(tools)}


# --- identity ----------------------------------------------------------------------


def test_identical_tasks_share_every_hash():
    assert identity_of(_task()) == identity_of(_task(task_id="other-id"))


def test_a_different_verifier_changes_the_verifier_hash_only():
    a, b = identity_of(_task()), identity_of(_task(verify="pytest -x"))
    assert a.verifier_hash != b.verifier_hash
    assert a.objective_hash == b.objective_hash
    assert a.environment_hash == b.environment_hash


def test_a_different_setup_changes_the_environment_hash():
    a, b = identity_of(_task()), identity_of(_task(setup="echo b"))
    assert a.environment_hash != b.environment_hash
    assert a.task_hash != b.task_hash


def test_word_order_does_not_change_the_objective():
    """'fix the bug, run the tests' and 'run the tests, fix the bug' are one request."""
    a = identity_of(_task(prompt="Fix the bug, then run the tests"))
    b = identity_of(_task(prompt="Run the tests, and fix the bug"))
    assert a.objective_hash == b.objective_hash


def test_morphological_variants_are_not_matched():
    """No stemming: 'fixing' and 'fix' are different tokens.

    A real limitation, recorded rather than papered over -- it is one of the reasons
    lexical similarity is only a proxy and an embedder hook exists.
    """
    a = identity_of(_task(prompt="fix the parser"))
    b = identity_of(_task(prompt="fixing the parser"))
    assert a.objective_hash != b.objective_hash


def test_stopwords_are_dropped():
    assert "the" not in normalize_objective("fix the bug in the module")


def test_identity_reads_task_objects_as_well_as_records():
    from hermesbench.tasks import Task

    record = _task()
    task = Task.from_record({**record, "tools": ["terminal"]})
    assert identity_of(task).objective_hash == identity_of(record).objective_hash


def test_jaccard_edges():
    assert jaccard([], []) == 1.0
    assert jaccard(["a"], []) == 0.0
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0


# --- execution duplicates ----------------------------------------------------------


def test_the_same_verifier_and_environment_is_one_task_in_two_wordings():
    index = DuplicateIndex()
    assert index.add(_task("first")) is None
    rejection = index.add(_task("second", prompt="Completely different wording here entirely"))
    assert rejection is not None
    assert rejection.reason == EXECUTION_DUPLICATE
    assert rejection.conflicts_with == "first"


def test_a_different_environment_is_a_different_task():
    index = DuplicateIndex()
    assert index.add(_task("a")) is None
    assert index.add(_task("b", setup="echo different")) is None
    assert len(index) == 2


# --- objective near-duplicates -----------------------------------------------------


def test_shared_wording_with_different_environments_is_not_a_duplicate():
    """The bug this module was written to fix.

    Every mutation-generated task carries the same generic prompt, so prompt similarity
    alone scored 1.00 across eleven genuinely different bugs. Different broken code is
    different work however alike the request reads.
    """
    index = DuplicateIndex()
    prompt = "A single bug was introduced into pkg/lib.py. Find it and fix it."
    assert index.add(_task("m1", prompt=prompt, setup="echo bug-one", verify="v1")) is None
    assert index.add(_task("m2", prompt=prompt, setup="echo bug-two", verify="v2")) is None
    assert len(index) == 2


def test_similar_wording_in_the_same_environment_is_a_duplicate():
    index = DuplicateIndex()
    index.add(_task("a", prompt="Fix the failing unit test in the parser module", verify="v1"))
    rejection = index.add(_task("b", prompt="Fix failing unit test parser module", verify="v2"))
    assert rejection is not None
    assert rejection.reason == OBJECTIVE_DUPLICATE


def test_the_environment_qualifier_can_be_disabled():
    index = DuplicateIndex(objective_dedup_requires_same_environment=False)
    index.add(_task("a", prompt="Fix the failing unit test in the parser module", setup="one", verify="v1"))
    rejection = index.add(_task("b", prompt="Fix failing unit test parser module", setup="two", verify="v2"))
    assert rejection is not None and rejection.reason == OBJECTIVE_DUPLICATE


def test_an_embedder_replaces_lexical_similarity():
    """The hook for real semantic dedup; lexical overlap is only a proxy."""
    vectors = {"alpha": [1.0, 0.0], "beta": [1.0, 0.0], "gamma": [0.0, 1.0]}
    index = DuplicateIndex(embedder=lambda prompt: vectors[prompt.split()[0]])

    assert index.add(_task("a", prompt="alpha task", verify="v1")) is None
    # Lexically almost nothing in common, but the embedder says identical.
    assert index.add(_task("b", prompt="beta task", verify="v2")) is not None
    assert index.add(_task("c", prompt="gamma task", verify="v3")) is None


# --- coverage saturation -----------------------------------------------------------


def test_a_saturated_bucket_refuses_more_of_the_same_skill():
    """Dedup proves tasks are not copies; only the cap proves the suite is not monotone."""
    index = DuplicateIndex(coverage=CoverageMatrix(cap_per_bucket=2))
    for i in range(2):
        assert index.add(_task(f"t{i}", setup=f"env{i}", verify=f"v{i}"), category="swe", topic="offbyone") is None

    rejection = index.add(_task("t2", setup="env2", verify="v2"), category="swe", topic="offbyone")
    assert rejection is not None
    assert rejection.reason == BUCKET_SATURATED


def test_a_different_topic_is_still_accepted_when_a_neighbour_is_full():
    index = DuplicateIndex(coverage=CoverageMatrix(cap_per_bucket=1))
    index.add(_task("a", setup="e1", verify="v1"), category="swe", topic="offbyone")
    assert index.add(_task("b", setup="e2", verify="v2"), category="swe", topic="race") is None


def test_thin_buckets_point_at_where_generation_is_needed():
    coverage = CoverageMatrix(cap_per_bucket=10)
    for _ in range(8):
        coverage.record("cuda", "fusion")
    coverage.record("cuda", "occupancy")
    assert coverage.thin_buckets(minimum=5) == (("cuda", "occupancy"),)


def test_coverage_record_is_json_safe():
    coverage = CoverageMatrix()
    coverage.record("swe", "parser")
    assert json.loads(json.dumps(coverage.to_record()))["counts"]["swe/parser"] == 1


# --- the stream filter -------------------------------------------------------------


def test_deduplicate_returns_rejections_rather_than_swallowing_them():
    """A generator whose output is 90% refused has run out of distinct things to break."""
    tasks = [_task("a"), _task("b"), _task("c", setup="other")]
    kept, rejected, _ = deduplicate(tasks)
    assert [t["task_id"] for t in kept] == ["a", "c"]
    assert [tid for tid, _ in rejected] == ["b"]


def test_deduplicate_applies_the_coverage_cap():
    tasks = [_task(f"t{i}", setup=f"e{i}", verify=f"v{i}") for i in range(5)]
    kept, rejected, coverage = deduplicate(
        tasks, cap_per_bucket=3, category_of=lambda t: "swe", topic_of=lambda t: "same"
    )
    assert len(kept) == 3
    assert all(r.reason == BUCKET_SATURATED for _, r in rejected)
    assert coverage.count("swe", "same") == 3


def test_deduplicate_on_an_empty_stream():
    kept, rejected, coverage = deduplicate([])
    assert kept == [] and rejected == [] and coverage.total == 0


def test_generator_output_survives_dedup_but_is_capped_by_coverage():
    """End to end on the shape that motivated this: same prompt, different bugs."""
    prompt = "A single bug was introduced into pkg/lib.py. Find it and fix it."
    tasks = [
        {**_task(f"mut{i}", prompt=prompt, setup=f"write bug {i}", verify="python -m unittest -q"), "op": op}
        for i, op in enumerate(["flip"] * 4 + ["offset"] * 4)
    ]
    kept, rejected, coverage = deduplicate(
        tasks, cap_per_bucket=3, category_of=lambda t: "swe", topic_of=lambda t: t["op"]
    )
    assert len(kept) == 6  # 3 per operator, not 1 total
    assert all(r.reason == BUCKET_SATURATED for _, r in rejected)
    assert coverage.counts == {("swe", "flip"): 3, ("swe", "offset"): 3}

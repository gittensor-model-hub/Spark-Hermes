"""Mining a real trace for its shape, and proving none of its content came along.

The invariant under test is `assert_abstract`: no run of `VERBATIM_WINDOW` characters from a seed may
appear anywhere in its DNA. The licence of a public dataset governs what may be copied from it, and
a DNA that quietly carries a sentence of the seed produces tasks that are derivative in a way nobody
notices until someone diffs them against the source.

The other tests are about the classifier being a classifier. A field that returns the same value for
every input is not a measurement, and that failure is invisible in any single example -- it took
printing the distribution over 400 seeds to see that every one of them had come out
`repository_engineering`.
"""

from __future__ import annotations

import pytest

from hermes.taskgen.dna import DOMAINS, VERBATIM_WINDOW, DNAError, TaskDNA, assert_abstract, extract

SEED = {
    "task": "The build fails with a CMake error about a missing target; find out why and fix it.",
    "category": "Repository Tasks",
    "tools": [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "read_file"}},
    ],
    "messages": [
        {"role": "user", "content": "The build fails with a CMake error about a missing target."},
        {"role": "tool", "content": "CMake Error: No such file or directory"},
    ],
}


def test_the_shape_survives_and_the_words_do_not():
    dna = extract(SEED, dataset="d", licence="apache-2.0", index=7)
    assert dna.required_tools == ("terminal", "file_read")
    assert dna.domain == "repository_engineering", "the published category decides it"
    assert dna.source == {
        "dataset": "d",
        "licence": "apache-2.0",
        "row": 7,
        "observed_tool_calls": 4,
    }
    rendered = str(dna.to_record())
    assert "CMake error about a missing target" not in rendered


def test_a_dna_that_quotes_its_seed_is_refused():
    """The guard, exercised directly. A generator that started copying prompt text into a label would
    otherwise produce derivative tasks with nothing to catch it."""
    quoted = TaskDNA(
        domain="repository_engineering",
        skills=("debugging",),
        environment="shell_workspace",
        difficulty=3,
        horizon=(4, 8),
        # Long enough to exceed the window, and lifted verbatim from the seed below.
        failure_mode="the build fails with a CMake error about a missing target",
        required_tools=("terminal",),
        verification="script",
    )
    with pytest.raises(DNAError, match="quotes"):
        assert_abstract(quoted, SEED["task"])


def test_a_short_shared_phrase_is_not_treated_as_quotation():
    """Technical language repeats across any engineering corpus. Flagging "debugging" or "terminal"
    would make the check unusable, so the bar is a run long enough to be a sentence fragment."""
    dna = extract(SEED, dataset="d", licence="apache-2.0", index=0)
    assert_abstract(dna, "debugging the terminal build with a script")
    assert len("debugging") < VERBATIM_WINDOW


def test_a_seed_with_no_harness_tool_is_dropped_rather_than_remapped():
    """A browser trace teaches nothing a terminal can reproduce. Mapping it onto `terminal` would put
    a capability in the corpus that no seed ever demonstrated."""
    browser = dict(SEED, tools=[{"type": "function", "function": {"name": "navigate_to_url"}}])
    with pytest.raises(DNAError, match="no tool"):
        extract(browser, dataset="d", licence="apache-2.0", index=0)


def test_difficulty_comes_from_the_trace_and_not_from_a_claim():
    """A seed asserting its own difficulty is not a measurement. What the trace shows is how many
    actions it took and whether a tool actually refused."""
    short = dict(SEED, messages=[SEED["messages"][0]])
    easy = extract(short, dataset="d", licence="apache-2.0", index=0)

    long_trace = dict(
        SEED,
        messages=[SEED["messages"][0]]
        + [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash"}}]} for _ in range(20)],
    )
    hard = extract(long_trace, dataset="d", licence="apache-2.0", index=0)
    assert hard.difficulty > easy.difficulty


def test_recovery_needs_a_tool_that_actually_refused():
    """Matching "error" anywhere in a tool response marked 93% of 800 real seeds as recoveries, which
    makes the flag useless and pushed 61% of the corpus to difficulty 5."""
    mentions = dict(
        SEED,
        messages=[
            SEED["messages"][0],
            {"role": "tool", "content": "The handler logs an error when the queue drains, which is expected."},
        ],
    )
    assert "error_recovery" not in extract(mentions, dataset="d", licence="apache-2.0", index=0).skills

    refused = dict(
        SEED,
        messages=[SEED["messages"][0], {"role": "tool", "content": "bash: cmake: command not found"}],
    )
    assert "error_recovery" in extract(refused, dataset="d", licence="apache-2.0", index=0).skills


def test_the_classifier_does_not_return_one_domain_for_everything():
    """The bug this file exists to have caught. `_classify` was filtering a table of SKILL labels for
    names in DOMAINS -- an empty intersection -- so every DNA in a 400-seed sample came out
    `repository_engineering`, and a generator fed 400 identical domains produces 400 variations of
    one world."""
    seen = set()
    for category, task in (
        ("Repository Tasks", "a failing pytest module"),
        ("File Operations", "parse the csv records"),
        ("Multi-Tool", "restart the systemd service"),
    ):
        dna = extract(dict(SEED, category=category, task=task), dataset="d", licence="apache-2.0", index=0)
        seen.add(dna.domain)
    assert len(seen) > 1, f"every category produced {seen}"
    assert seen <= set(DOMAINS)


def test_an_inverted_horizon_or_an_impossible_difficulty_is_refused():
    for bad in ({"difficulty": 9}, {"horizon": (10, 2)}, {"required_tools": ()}):
        fields = {
            "domain": "repository_engineering",
            "skills": ("debugging",),
            "environment": "shell_workspace",
            "difficulty": 3,
            "horizon": (2, 6),
            "failure_mode": "off_by_one",
            "required_tools": ("terminal",),
            "verification": "script",
        }
        fields.update(bad)
        with pytest.raises(DNAError):
            TaskDNA(**fields)

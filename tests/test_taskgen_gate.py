"""The gate is the trust anchor, so what matters is that it REJECTS.

A gate that accepts everything is worse than no gate: it produces a directory of tasks with a stamp
of approval on them, and the broken ones then generate failed episodes indistinguishable from a model
that could not do the work. Every test here builds a task that is broken in exactly one way and
asserts the gate names that way.

The one acceptance test uses a task modelled on the hand-written suite -- a published check that
verifies an order-free quantity and a withheld check that verifies the order-sensitive one, so a
solution that globs and accumulates passes the first and dies on the second.
"""

from __future__ import annotations

import pytest

from hermes.taskgen.gate import ALL_CHECKS, Candidate, gate

# A workspace of three fragments that must be concatenated in the order given by each file's header,
# not by filename. `total` is order-free; `first_word` is not. Deterministic by construction: literal
# content, no clock, no randomness.
SETUP = """
set -e
mkdir -p parts
printf '# seq: 2\\nbeta\\n' > parts/a.txt
printf '# seq: 1\\nalpha\\n' > parts/b.txt
printf '# seq: 3\\ngamma\\n' > parts/c.txt
"""

# Order-free: how many words there are. `cat parts/*` in any order gets this right.
PUBLIC = """
set -e
test -f answer.txt
grep -q '^total=3$' answer.txt
"""

# Order-sensitive: which word comes first once the fragments are in header order.
WITHHELD = """
set -e
test -f answer.txt
grep -q '^first=alpha$' answer.txt
"""

REFERENCE = """
set -e
printf 'total=3\\nfirst=alpha\\n' > answer.txt
"""

# Satisfies the published check exactly and answers the order question wrongly -- which is what a
# solution that globbed the directory would produce, since `a.txt` sorts first.
CHEAT = """
set -e
printf 'total=3\\nfirst=beta\\n' > answer.txt
"""


def _candidate(**overrides) -> Candidate:
    base = {
        "task_id": "gen-example",
        "setup": SETUP,
        "verify": PUBLIC,
        "withheld_verify": WITHHELD,
        "reference_solution": REFERENCE,
        "cheat_solution": CHEAT,
    }
    base.update(overrides)
    return Candidate(**base)


def test_a_sound_task_is_accepted_and_every_check_ran():
    verdict = gate(_candidate())
    assert verdict.accepted, f"{verdict.failed_check}: {verdict.detail}"
    assert verdict.checks_run == list(ALL_CHECKS), "a task cannot be accepted on a subset of the checks"


def test_a_setup_that_fails_is_named():
    verdict = gate(_candidate(setup="set -e\nexit 3\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "setup_exits_zero"


def test_a_nondeterministic_setup_is_rejected():
    """The failure this catches appears much later and somewhere else: a withheld check pinned to
    values from a workspace that moves passes on the machine that wrote it and nowhere else."""
    verdict = gate(_candidate(setup=SETUP + "\ndate +%s%N > parts/stamp.txt\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "setup_is_deterministic"
    assert "seeded generator" in verdict.detail


def test_a_public_check_that_passes_an_untouched_workspace_is_rejected():
    """It would mark every episode a success, including the ones where the agent did nothing."""
    verdict = gate(_candidate(verify="exit 0"))
    assert not verdict.accepted
    assert verdict.failed_check == "public_fails_untouched"


def test_a_withheld_check_that_passes_an_untouched_workspace_is_rejected():
    verdict = gate(_candidate(withheld_verify="exit 0"))
    assert not verdict.accepted
    assert verdict.failed_check == "withheld_fails_untouched"


def test_a_reference_solution_that_crashes_is_rejected():
    """Without a solution that runs, the task is not known to be solvable -- and a failed episode on
    an unsolvable task is unattributable: the model may be wrong, or the task may be."""
    verdict = gate(_candidate(reference_solution="set -e\nexit 1\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "reference_solution_runs"


def test_a_reference_solution_that_does_not_satisfy_the_public_check_is_rejected():
    verdict = gate(_candidate(reference_solution="set -e\nprintf 'total=99\\nfirst=alpha\\n' > answer.txt\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "public_passes_reference"


def test_a_reference_solution_that_fails_the_withheld_check_is_rejected():
    """This is the shape that would otherwise ship a task NOBODY can pass: the published check is
    satisfiable and the withheld one is not, so every episode scores as overfit forever."""
    verdict = gate(_candidate(reference_solution="set -e\nprintf 'total=3\\nfirst=zeta\\n' > answer.txt\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "withheld_passes_reference"


def test_a_withheld_check_that_agrees_with_the_public_one_is_rejected():
    """The check that decides whether the withheld half earns its cost.

    A withheld check identical in effect to the published one adds runtime, produces an
    `overfit_rate` that is structurally zero, and creates the appearance of a second opinion where
    there is one opinion twice.
    """
    verdict = gate(_candidate(withheld_verify=PUBLIC))
    assert not verdict.accepted
    assert verdict.failed_check == "checks_disagree_on_a_cheat"
    assert "withholds nothing" in verdict.detail


def test_a_cheat_that_cannot_pass_the_public_check_is_rejected():
    """Then the task proves nothing about the withheld check either way, and the generator has to be
    told that rather than having its task quietly accepted on a check that never really ran."""
    verdict = gate(_candidate(cheat_solution="set -e\ntrue\n"))
    assert not verdict.accepted
    assert verdict.failed_check == "checks_disagree_on_a_cheat"
    assert "did not even pass the published check" in verdict.detail


def test_a_setup_that_hangs_is_killed_rather_than_holding_the_slot():
    """At three hundred tasks a generator will eventually emit a script that blocks forever, and a
    generation run that stalls on one of them is worse than one that drops it."""
    verdict = gate(_candidate(setup="sleep 600"), timeout_s=2)
    assert not verdict.accepted
    assert verdict.failed_check == "setup_exits_zero"


@pytest.mark.parametrize("check", ALL_CHECKS)
def test_every_declared_check_is_reachable(check):
    """`ALL_CHECKS` is what the acceptance test asserts against, so a name listed there but never
    appended is a check that silently does not run."""
    import inspect

    from hermes.taskgen import gate as module

    assert f'"{check}"' in inspect.getsource(module), f"{check} is declared but never used"

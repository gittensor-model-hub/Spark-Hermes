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

# The same correct answer by a visibly different route: derived from the files rather than written
# out as literals. A withheld check that only accepts REFERENCE fails this, which is the point.
ALTERNATE = """
set -e
total=$(cat parts/*.txt | grep -vc '^#')
first=$(for f in parts/*.txt; do printf '%s %s\\n' "$(sed -n 's/^# seq: //p' "$f")" "$(sed -n '2p' "$f")"; done | sort -n | head -1 | cut -d' ' -f2)
printf 'total=%s\\nfirst=%s\\n' "$total" "$first" > answer.txt
"""


# The inputs the answer is DERIVED from. An agent that may rewrite these can edit `parts/b.txt`
# until its wrong `first=` is the right one, and the published check cannot tell the difference.
# Outputs (`answer.txt`) are deliberately absent: protecting one would disqualify every correct run.
PROTECTED = ("parts/a.txt", "parts/b.txt", "parts/c.txt")


def _candidate(**overrides) -> Candidate:
    base = {
        "task_id": "gen-example",
        "setup": SETUP,
        "verify": PUBLIC,
        "withheld_verify": WITHHELD,
        "reference_solution": REFERENCE,
        "cheat_solution": CHEAT,
        "alternate_solution": ALTERNATE,
        "protected_paths": PROTECTED,
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


def test_a_withheld_check_that_grades_the_METHOD_is_rejected():
    """The hole the first accepted generated task fell into.

    Its withheld check demanded a *symlink* specifically. An agent that set an environment variable
    or copied the file would have fixed the service for real and still failed -- scoring as `overfit`
    while being correct. Checks 6 and 7 cannot see it: the reference solution came from the same
    reply as the check and satisfies it by construction, so one solution can never prove a check is
    outcome-based.
    """
    method_pinned = """
set -e
test -f answer.txt
grep -q '^first=alpha$' answer.txt
test -f .used_the_blessed_tool
"""
    blessed_reference = REFERENCE + "\ntouch .used_the_blessed_tool\n"
    verdict = gate(_candidate(withheld_verify=method_pinned, reference_solution=blessed_reference))
    assert not verdict.accepted
    assert verdict.failed_check == "withheld_accepts_a_different_method"
    assert "grades the METHOD" in verdict.detail


def test_a_task_with_no_alternate_solution_skips_the_check_rather_than_failing_it():
    """Backwards compatible on purpose: a hand-written task predates this field, and refusing those
    would make the gate unusable on the suite it was modelled on. The check is simply not in
    `checks_run`, so a reader can see it did not happen rather than assuming it passed."""
    verdict = gate(_candidate(alternate_solution=""))
    assert verdict.accepted
    assert "withheld_accepts_a_different_method" not in verdict.checks_run


def test_a_workspace_holding_a_dangling_symlink_can_still_be_gated():
    """These tasks create symlinks deliberately -- the first one this gate ever accepted turned on
    whether a config path was linked to the real file -- and a setup that leaves a DANGLING link is a
    perfectly good workspace for a task about repairing it.

    `copytree` follows links by default, so a dangling target raised and took down a generation run
    at 116 of 150 accepted: one malformed workspace ended the process rather than the attempt.
    """
    # At the top level, not inside parts/: the solutions glob parts/*.txt, and a dangling entry
    # there would break them rather than exercising the copy step this test is about.
    setup = SETUP + "\nln -s nowhere.txt dangling.txt\n"
    verdict = gate(_candidate(setup=setup))
    assert verdict.accepted, f"{verdict.failed_check}: {verdict.detail}"


def test_a_workspace_the_gate_cannot_handle_rejects_the_task_not_the_run():
    """An OSError from a generated workspace is a fact about the task. Letting it propagate ends the
    run and throws away every attempt still queued behind it."""
    from hermes.taskgen.gate import WORKSPACE_UNUSABLE

    # A setup that builds a directory the copy step cannot traverse.
    hostile = SETUP + "\nmkdir -p locked/inner\nchmod 000 locked\n"
    verdict = gate(_candidate(setup=hostile))
    assert not verdict.accepted
    assert verdict.failed_check in (WORKSPACE_UNUSABLE, "setup_is_deterministic"), verdict.failed_check


@pytest.mark.parametrize("check", ALL_CHECKS)
def test_every_declared_check_is_reachable(check):
    """`ALL_CHECKS` is what the acceptance test asserts against, so a name listed there but never
    appended is a check that silently does not run."""
    import inspect

    from hermes.taskgen import gate as module

    assert f'"{check}"' in inspect.getsource(module), f"{check} is declared but never used"


# --- protected paths ---------------------------------------------------------------


def test_a_task_that_protects_nothing_is_rejected():
    """An empty tuple is not "nothing to protect", it is the integrity check switched off.

    `hermesbench.runner` returns early when `protected_paths` is empty, so a task shipping the
    default grades an agent that rewrote its own inputs exactly the same as one that did the work.
    """
    verdict = gate(_candidate(protected_paths=()))
    assert not verdict.accepted
    assert verdict.failed_check == "protected_paths_are_sound"


def test_a_protected_path_that_does_not_exist_is_rejected():
    """A path setup never created protects nothing, and the runner would read it as a deletion."""
    verdict = gate(_candidate(protected_paths=("parts/a.txt", "parts/never-written.txt")))
    assert not verdict.accepted
    assert verdict.failed_check == "protected_paths_are_sound"
    assert "never-written" in verdict.detail


def test_protecting_an_output_the_task_asks_for_is_rejected():
    """`answer.txt` is what the agent must WRITE, so setup never creates it and it protects nothing."""
    verdict = gate(_candidate(protected_paths=("parts/a.txt", "answer.txt")))
    assert not verdict.accepted
    assert verdict.failed_check == "protected_paths_are_sound"
    assert "do not exist after setup" in verdict.detail


def test_a_task_whose_own_solution_rewrites_a_protected_path_is_rejected():
    """The failure this check exists for, and the one a task author cannot see by inspection.

    The file exists, so the presence check passes; the reference solution then rewrites it. Every
    agent solving this task the intended way would be disqualified for tampering. Same shape as the
    defect check 9 catches -- a correct solution scored as cheating -- arriving from the inputs
    rather than from the verifier.
    """
    tampering = REFERENCE + "\nprintf '# seq: 2\\nbeta\\n# touched\\n' > parts/a.txt\n"
    verdict = gate(_candidate(reference_solution=tampering))
    assert not verdict.accepted
    assert verdict.failed_check == "protected_paths_are_sound"
    assert "modifies protected path" in verdict.detail, verdict.detail


def test_protecting_the_inputs_does_not_block_a_correct_solution():
    """The positive case: inputs protected, outputs free, reference still passes everything."""
    verdict = gate(_candidate())
    assert verdict.accepted, f"{verdict.failed_check}: {verdict.detail}"
    assert "protected_paths_are_sound" in verdict.checks_run


# --- workspace escapes ---------------------------------------------------------------


def test_prose_inside_a_heredoc_is_data_not_a_command():
    """The false positive the first version of this check produced on real model output.

    A task writing a runbook that SAYS "run `ssh-add ~/.ssh/id_rsa`" is documenting a system, not
    touching the home directory. 1 of 11 model-reached generations was rejected for it."""
    from hermes.taskgen.gate import _escapes_workspace

    setup = "mkdir -p docs\ncat > docs/guide.md <<'EOF'\n## Deploy\n**Fix:** run `ssh-add ~/.ssh/id_rsa`\nsee /etc/ssh/config\nEOF\ntouch done\n"
    assert _escapes_workspace(setup) == ""


def test_an_escape_after_the_heredoc_ends_is_still_caught():
    from hermes.taskgen.gate import _escapes_workspace

    setup = "cat > a.md <<EOF\nharmless\nEOF\nmkdir -p /opt/app\n"
    assert "line 4" in _escapes_workspace(setup)


def test_home_spelled_as_a_variable_is_the_same_escape():
    """`~/` and `$HOME` reach the same place; catching one teaches the model the other."""
    from hermes.taskgen.gate import _escapes_workspace

    assert "home-relative" in _escapes_workspace("mkdir -p $HOME/stash\n")
    assert "home-relative" in _escapes_workspace("mkdir -p ${HOME}/stash\n")
    assert "home-relative" in _escapes_workspace("mkdir -p ~/stash\n")


def test_legitimate_absolute_paths_are_not_escapes():
    from hermes.taskgen.gate import _escapes_workspace

    assert (
        _escapes_workspace("#!/bin/sh\nprintf x >/dev/null\nPY=\"$(command -v python3)\"\n/usr/bin/env awk 'NR>1' f\n")
        == ""
    )


def test_a_numbered_protected_list_is_read_as_paths():
    from hermes.taskgen.synth import _bare_path

    assert _bare_path("1. data/x.csv") == "data/x.csv"
    assert _bare_path("- `data/x.csv`") == "data/x.csv"
    assert _bare_path("  * data/x.csv  ") == "data/x.csv"
    assert _bare_path("data/x.csv") == "data/x.csv"


def test_protected_paths_survive_yaml_with_a_quote_in_them(tmp_path):
    """Emitted single-quoted, so a `'` doubles and a `"` or `\\` is literal."""
    import yaml

    from hermes.taskgen.dna import TaskDNA
    from hermes.taskgen.synth import Synthesised, to_task_yaml

    dna = TaskDNA(
        domain="d",
        skills=("s",),
        environment="e",
        difficulty=3,
        horizon=(2, 4),
        failure_mode="f",
        required_tools=("terminal",),
        verification="v",
        source={},
    )
    c = _candidate(protected_paths=("it's.csv", 'q"uote.txt', "back\\slash"))
    record = yaml.safe_load(
        to_task_yaml(Synthesised(candidate=c, prompt="p", dna=dna, raw=""), commitment="sha256:x", max_steps=12)
    )
    assert tuple(record["protected_paths"]) == ("it's.csv", 'q"uote.txt', "back\\slash")

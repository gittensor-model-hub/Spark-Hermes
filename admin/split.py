"""Which tasks may be trained on, and which exist only to measure.

Everything else in this pipeline is reversible. This is not: once a task's rollouts are in the
corpus, every later number measured on that task is a number about the training set, and no
downstream check can recover the distinction. The instrument that would normally notice --
`overfit_rate` -- is itself measured on these tasks, so training on them blinds exactly the thing
that would report the problem.

## The split

**Eval** is the nineteen hand-written tasks in `hermesbench/tasks/`. They are hand-authored, they
have withheld checks with reference solutions proven to pass, and every published number for this
project is measured on them. They are never trained on.

**Train** is whatever `hermes.taskgen` produced and the gate accepted. Generated, disposable, and
regenerable -- if a training task turns out to be broken the cost is a wasted rollout, not a corrupt
baseline.

## Contamination is checked, not assumed

Generated tasks are seeded from DNA mined out of public traces, and nothing stops a generator from
independently producing something close to an eval task. `contamination` compares the two sets by
the same rule `dna.assert_abstract` uses -- a shared run of `VERBATIM_WINDOW` characters is
quotation, not coincidence -- and names the pairs rather than returning a score. A number says
"0.03 similarity"; a name says which two tasks to look at.

Prompts are compared, not setups. Two tasks that build a similar workspace and ask different
questions of it are different tasks; two that ask the same question of different bytes are the same
task wearing a hat, and the prompt is where that shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from hermes.taskgen.dna import VERBATIM_WINDOW
from hermesbench.tasks import TASKS_ROOT, load_task

# Where the hand-written suite lives. Read from disk rather than listed here: a task added to the
# suite is an eval task from the moment it exists, and a hard-coded list would silently make the
# twentieth one trainable.
EVAL_ROOT = TASKS_ROOT.resolve()


class SplitError(RuntimeError):
    """A corpus was about to be built from something that is not trainable."""


@dataclass
class Contamination:
    """Overlap found between a training task and the evaluation suite."""

    train_task: str
    eval_task: str
    shared: str

    def __str__(self) -> str:
        return (
            f"{self.train_task} shares {len(self.shared)} characters with eval task {self.eval_task}: {self.shared!r}"
        )


@dataclass
class Split:
    """The two sets, and everything that was checked about them."""

    eval_tasks: tuple[str, ...] = ()
    train_tasks: tuple[str, ...] = ()
    contaminated: list[Contamination] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "eval_tasks": list(self.eval_tasks),
            "train_tasks": list(self.train_tasks),
            "contaminated": [str(c) for c in self.contaminated],
        }


def eval_task_ids(root: Path | None = None) -> tuple[str, ...]:
    """Every task id in the hand-written suite, read from disk.

    A task file that exists is an eval task. Enumerating them here rather than maintaining a list is
    what keeps the twentieth hand-written task from being quietly trainable on the day it lands.
    """
    base = EVAL_ROOT if root is None else root
    if not base.is_dir():
        raise SplitError(f"{base} does not exist; the evaluation suite has to be readable to be protected")
    ids = []
    for path in sorted([*base.rglob("*.yaml"), *base.rglob("*.yml")]):
        ids.append(load_task(path).task_id)
    for path in sorted(base.rglob("*.jsonl")):
        ids.append(path.stem)
    if not ids:
        raise SplitError(f"no tasks found under {base}; refusing to treat an empty eval set as 'nothing to protect'")
    return tuple(dict.fromkeys(ids))


def _windows(text: str) -> set[str]:
    flat = re.sub(r"\s+", " ", text).strip().lower()
    return {flat[i : i + VERBATIM_WINDOW] for i in range(max(0, len(flat) - VERBATIM_WINDOW) + 1)}


def contamination(train: dict[str, str], evaluation: dict[str, str]) -> list[Contamination]:
    """Training tasks that quote an evaluation task, by id and by the shared text.

    Named rather than scored. A similarity number invites a threshold argument; a pair of task ids
    and the exact shared run is something an operator can open and judge in ten seconds.
    """
    found: list[Contamination] = []
    eval_windows = {task_id: _windows(prompt) for task_id, prompt in evaluation.items()}
    for train_id, prompt in train.items():
        mine = _windows(prompt)
        for eval_id, theirs in eval_windows.items():
            shared = mine & theirs
            if shared:
                found.append(Contamination(train_task=train_id, eval_task=eval_id, shared=sorted(shared)[0]))
    return found


def refuse_eval_tasks(task_ids: list[str], *, root: Path | None = None) -> None:
    """Raise if any of these is an evaluation task. Called before a corpus is written.

    A refusal rather than a filter. Dropping the eval tasks silently would produce a corpus that is
    correct and a count that is wrong, and the operator would go on believing they trained on what
    they asked for. Naming them is also the only way the mistake gets fixed upstream, where it was
    made.
    """
    held_out = set(eval_task_ids(root))
    offending = sorted(set(task_ids) & held_out)
    if offending:
        raise SplitError(
            f"{len(offending)} evaluation task(s) reached the training corpus: {', '.join(offending)}. "
            "These are the tasks every published number for this project is measured on, and "
            "`overfit_rate` -- the check that would notice -- is measured on them too, so training "
            "on them blinds the instrument that reports the problem"
        )


def build(
    *,
    train_prompts: dict[str, str],
    root: Path | None = None,
    eval_prompts: dict[str, str] | None = None,
) -> Split:
    """The split, with both guards run.

    `eval_prompts` is optional because reading every task YAML costs a parse the caller may already
    have done; when absent, only the id check runs and the split says so by leaving `contaminated`
    empty. That is a real limitation rather than a clean result, which is why the caller is the one
    who decides.
    """
    held_out = eval_task_ids(root)
    refuse_eval_tasks(list(train_prompts), root=root)
    overlaps = contamination(train_prompts, eval_prompts) if eval_prompts else []
    return Split(eval_tasks=held_out, train_tasks=tuple(sorted(train_prompts)), contaminated=overlaps)


__all__ = [
    "EVAL_ROOT",
    "Contamination",
    "Split",
    "SplitError",
    "build",
    "contamination",
    "eval_task_ids",
    "refuse_eval_tasks",
]

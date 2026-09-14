"""Mutation-based task generation: unlimited tasks with a known answer.

Hand-writing arena tasks does not scale to a thousand, and every hand-written task needs
its verifier checked in both directions by a human. Mutation generation inverts that:
take working code with a passing test suite, break it in one specific way, and the task
becomes "find and fix it". The correct answer is known by construction and the verifier
already exists.

The invariant that makes it worth anything:

    a mutant is only emitted as a task once it has been observed to FAIL the verifier.

Mutation testing calls a mutation that does not change behaviour an *equivalent mutant*.
Emitted as a task it would be worse than useless -- the suite passes on the untouched
workspace, so the agent scores a success for doing nothing, and that success flows into
SFT data and the capability matrix. `generate` therefore runs the verifier against every
candidate and discards the ones that stay green, which is the same "verify must fail on
an untouched workspace" rule the hand-written tasks are held to, applied automatically.

Operators work on the AST rather than on text. A regex that turns `<` into `<=` also
edits string literals and comments; an AST rewrite cannot, and it guarantees the mutant
still parses.
"""

from __future__ import annotations

import ast
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermesbench.verify import run_command

# --- operators ---------------------------------------------------------------------

COMPARISON_FLIPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}

ARITHMETIC_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}

FLIP_COMPARISON = "flip_comparison"
SWAP_ARITHMETIC = "swap_arithmetic"
OFFSET_CONSTANT = "offset_constant"

DEFAULT_OPERATORS = (FLIP_COMPARISON, SWAP_ARITHMETIC, OFFSET_CONSTANT)


class MutationError(ValueError):
    """Source could not be mutated."""


@dataclass(frozen=True)
class Mutant:
    """One single-point change to a source file, with its provenance."""

    operator: str
    source: str
    line: int
    description: str

    def to_record(self) -> dict[str, Any]:
        return {"operator": self.operator, "line": self.line, "description": self.description}


class _SinglePointMutator(ast.NodeTransformer):
    """Applies exactly one mutation, at the `target`-th eligible site.

    One change per mutant on purpose: a task with two independent bugs cannot tell an
    agent that fixed one from an agent that fixed neither, because the suite is red
    either way.
    """

    def __init__(self, operator: str, target: int) -> None:
        self.operator = operator
        self.target = target
        self.seen = 0
        self.applied_line = -1
        self.detail = ""

    def _hit(self) -> bool:
        hit = self.seen == self.target
        self.seen += 1
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.operator != FLIP_COMPARISON:
            return node
        for index, op in enumerate(node.ops):
            replacement = COMPARISON_FLIPS.get(type(op))
            if replacement is None:
                continue
            if self._hit():
                node.ops[index] = replacement()
                self.applied_line = getattr(node, "lineno", -1)
                self.detail = f"{type(op).__name__} -> {replacement.__name__}"
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.operator != SWAP_ARITHMETIC:
            return node
        replacement = ARITHMETIC_SWAPS.get(type(node.op))
        if replacement is not None and self._hit():
            self.applied_line = getattr(node, "lineno", -1)
            self.detail = f"{type(node.op).__name__} -> {replacement.__name__}"
            node.op = replacement()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.operator != OFFSET_CONSTANT:
            return node
        # bool is a subclass of int; flipping True/False is a different operator and
        # usually produces an obviously-broken program rather than a subtle bug.
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            return node
        if self._hit():
            self.applied_line = getattr(node, "lineno", -1)
            self.detail = f"{node.value} -> {node.value + 1}"
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node


def _site_count(tree: ast.AST, operator: str) -> int:
    counter = _SinglePointMutator(operator, target=-1)
    counter.visit(ast.parse(ast.unparse(tree)))
    return counter.seen


def iter_mutants(source: str, operators: tuple[str, ...] = DEFAULT_OPERATORS) -> Iterator[Mutant]:
    """Yield every single-point mutant of `source`, one per eligible site.

    Ordered by (operator, site) so a given source always produces the same mutants in
    the same order -- a generated suite has to be reproducible or its numbers cannot be
    compared across runs.
    """
    try:
        original = ast.parse(source)
    except SyntaxError as exc:
        raise MutationError(f"source does not parse: {exc}") from exc

    baseline = ast.unparse(original)
    for operator in operators:
        for site in range(_site_count(original, operator)):
            mutator = _SinglePointMutator(operator, target=site)
            mutated = ast.unparse(mutator.visit(ast.parse(source)))
            if mutated == baseline:
                # The rewrite produced identical code; nothing to test.
                continue
            yield Mutant(
                operator=operator,
                source=mutated,
                line=mutator.applied_line,
                description=f"{operator} at line {mutator.applied_line}: {mutator.detail}",
            )


# --- task generation ---------------------------------------------------------------

_HEREDOC = "SPARK_MUTATION_EOF"

# Directories that must never reach a generated workspace, whatever the source project holds.
#
# `.git` is the one that matters and it is not a hygiene issue, it is an answer key: a mutation
# task is "one file was broken, fix it", and a workspace carrying the project's history answers it
# with `git checkout <target>` or `git log -p`. Every mutant of that project would fall to the same
# single command, and the suite would report a capable agent.
#
# The rest are reconstructible build state. They bloat every setup script with content the task
# does not need, and a stale cache is workspace state the task never intended to pin.
EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".eggs",
        ".DS_Store",
    }
)


def _is_text(path: Path) -> bool:
    """Whether a file can be embedded in a heredoc at all.

    A setup script is text, so a binary file cannot travel in one. Before this check, the reader
    was `read_text(encoding="utf-8")` over EVERY file under the project, and the first byte that
    was not valid UTF-8 ended the whole run with a `UnicodeDecodeError` naming neither the file nor
    the reason. Any real repository trips it -- a PNG fixture, a compiled artifact, a git object.
    """
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def _setup_script(project_dir: Path, target_file: str, mutant_source: str) -> tuple[str, list[str]]:
    """Emit a setup script that rebuilds the whole broken project from scratch, and what it skipped.

    Self-contained on purpose. A task that points at a fixture directory outside itself
    stops working the moment that directory moves, and a generated suite is exactly the
    kind of thing nobody notices has silently stopped reconstructing its own workspace.

    Returns the script and the relative paths it could not embed, so a caller can report them
    rather than a suite silently describing a project that is missing files.
    """
    lines = ["set -e"]
    skipped: list[str] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(project_dir)
        # Judged on the path INSIDE the project. `path.parts` carries every ancestor too, so a
        # project checked out under a directory that happens to be named `venv` or `.git` would
        # have every one of its files excluded and rebuild an empty workspace -- silently.
        if EXCLUDED_DIRS & set(relative_path.parts):
            continue
        relative = relative_path.as_posix()
        if relative != target_file and not _is_text(path):
            # Skipped rather than fatal: one binary fixture must not cost the whole project, and a
            # named skip is something a reader can act on.
            skipped.append(relative)
            continue
        content = mutant_source if relative == target_file else path.read_text(encoding="utf-8")
        if _HEREDOC in content:
            raise MutationError(f"{relative} contains the heredoc delimiter; cannot embed it safely")
        parent = str(Path(relative).parent)
        if parent not in (".", ""):
            lines.append(f"mkdir -p {parent}")
        lines.append(f"cat > {relative} <<'{_HEREDOC}'")
        lines.append(content.rstrip("\n"))
        lines.append(_HEREDOC)
    return "\n".join(lines) + "\n", skipped


@dataclass(frozen=True)
class GeneratedTask:
    """A mutation task, already proven to fail before the fix."""

    task_id: str
    mutant: Mutant
    target_file: str
    record: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return self.record


@dataclass(frozen=True)
class GenerationReport:
    """What the generator produced, and what it threw away and why.

    `equivalent` is the number that matters. A high rate means the operators are mostly
    producing no-ops on this source, and the yield is not worth the verifier time.
    """

    emitted: int
    equivalent_discarded: int
    unparseable_discarded: int
    candidates: int

    @property
    def yield_rate(self) -> float:
        return self.emitted / self.candidates if self.candidates else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "emitted": self.emitted,
            "equivalent_discarded": self.equivalent_discarded,
            "unparseable_discarded": self.unparseable_discarded,
            "yield_rate": round(self.yield_rate, 4),
        }


def generate(
    *,
    project_dir: Path,
    target_file: str,
    verify: str,
    workspace: Path,
    task_prefix: str,
    tools: tuple[str, ...] = ("terminal", "file_read", "file_write", "python"),
    verification_tools: tuple[str, ...] = ("terminal", "python"),
    protected_paths: tuple[str, ...] = (),
    operators: tuple[str, ...] = DEFAULT_OPERATORS,
    timeout_s: int = 120,
    limit: int | None = None,
) -> tuple[list[GeneratedTask], GenerationReport]:
    """Mutate `target_file` and emit a task for every mutant the verifier catches.

    `project_dir` must be a working project: `verify` has to pass on it untouched, and
    that is asserted first. Generating "find the bug" tasks from a project that is
    already broken would produce tasks nobody can complete.
    """
    source_path = project_dir / target_file
    if not source_path.is_file():
        raise MutationError(f"no such target file: {source_path}")

    workspace.mkdir(parents=True, exist_ok=True)
    probe = workspace / "_baseline"
    if probe.exists():
        shutil.rmtree(probe)
    shutil.copytree(project_dir, probe)
    baseline = run_command(verify, cwd=probe, timeout_s=timeout_s)
    if not baseline.passed:
        raise MutationError(
            f"verify does not pass on the unmutated project ({(baseline.stdout + baseline.stderr).strip()[:200]}); "
            "mutation tasks need a known-good starting point"
        )

    original_source = source_path.read_text(encoding="utf-8")
    tasks: list[GeneratedTask] = []
    equivalent = 0
    unparseable = 0
    candidates = 0

    for index, mutant in enumerate(iter_mutants(original_source, operators)):
        if limit is not None and len(tasks) >= limit:
            break
        candidates += 1

        trial = workspace / f"_trial_{index}"
        if trial.exists():
            shutil.rmtree(trial)
        shutil.copytree(project_dir, trial)
        (trial / target_file).write_text(mutant.source, encoding="utf-8")

        result = run_command(verify, cwd=trial, timeout_s=timeout_s)
        shutil.rmtree(trial)

        if result.passed:
            # Equivalent mutant: behaviour unchanged, so the task would be solved by
            # doing nothing at all.
            equivalent += 1
            continue

        task_id = f"{task_prefix}-{mutant.operator}-{index:04d}"
        setup_script, skipped = _setup_script(project_dir, target_file, mutant.source)
        tasks.append(
            GeneratedTask(
                task_id=task_id,
                mutant=mutant,
                target_file=target_file,
                record={
                    "task_id": task_id,
                    "tags": ["swe", "debug", "generated", "mutation"],
                    "prompt": (
                        f"The test suite in this workspace is failing. A single bug was introduced "
                        f"into `{target_file}`. Find it, fix it in the source, and confirm the suite "
                        f"passes. Do not modify the tests."
                    ),
                    "tools": list(tools),
                    "verification_tools": list(verification_tools),
                    "protected_paths": list(protected_paths),
                    "timeout_s": timeout_s,
                    "max_steps": 30,
                    "setup": setup_script,
                    "verify": verify,
                    "metadata": {
                        "generated_by": "hermesbench.mutation",
                        "target_file": target_file,
                        # Recorded rather than dropped: a workspace missing a binary fixture the
                        # suite needs fails for a reason that has nothing to do with the agent, and
                        # the only way to notice is for the omission to be written down.
                        **({"skipped_binary_files": skipped} if skipped else {}),
                        **mutant.to_record(),
                    },
                },
            )
        )

    shutil.rmtree(probe, ignore_errors=True)
    return tasks, GenerationReport(
        emitted=len(tasks),
        equivalent_discarded=equivalent,
        unparseable_discarded=unparseable,
        candidates=candidates,
    )

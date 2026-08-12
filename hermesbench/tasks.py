"""HermesBench task specifications.

A task is a unit of work with an objective pass/fail check attached. The check --
`verify`, a shell command whose exit status decides success -- is the whole point: it is
what lets the harness score an agent without believing anything the agent said about
itself.

Task files are YAML, one task per file, under `hermesbench/tasks/<version>/`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS_ROOT = Path(__file__).parent / "tasks"

# Tools that change state. Used to score whether an agent looked at the consequences of
# its own edits before declaring done (see hermesbench.metrics.self_check_rate). Tasks
# override this when they offer domain tools that mutate.
DEFAULT_MUTATING_TOOLS = ("edit", "write", "patch", "apply_patch", "file_write")


class TaskError(ValueError):
    """A task specification is malformed."""


@dataclass(frozen=True)
class Checkpoint:
    """One sub-objective of a long-horizon task, checkable on its own.

    Checkpoints are what make goal drift *measurable* rather than anecdotal. A single
    end-of-episode `verify` can only say whether the agent finished; sampling each
    checkpoint repeatedly during the episode shows whether an objective it had already
    satisfied later broke while it worked on something else. That regression is the
    concrete, observable form of "goal drift" and "context corruption" -- and an agent
    that quietly undoes its own earlier work scores identically to one that never got
    there, unless you look.
    """

    checkpoint_id: str
    verify: str
    description: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any], *, origin: str = "<memory>") -> Checkpoint:
        missing = [key for key in ("checkpoint_id", "verify") if not record.get(key)]
        if missing:
            raise TaskError(f"{origin}: checkpoint is missing required field(s): {', '.join(missing)}")
        return cls(
            checkpoint_id=str(record["checkpoint_id"]),
            verify=str(record["verify"]),
            description=str(record.get("description") or ""),
        )


# What a declared shortcut is expected to do. Named constants rather than bare strings, so a
# typo in a task file is a load error instead of a shortcut that silently asserts nothing.
FAILS_PUBLIC = "fails_public"
PASSES_PUBLIC_FAILS_HIDDEN = "passes_public_fails_hidden"
SHORTCUT_EXPECTATIONS = (FAILS_PUBLIC, PASSES_PUBLIC_FAILS_HIDDEN)


@dataclass(frozen=True)
class Shortcut:
    """A plausible-lazy strategy, written down as a script and expected to be caught.

    This exists because "broken workspace fails, correct solution passes, deterministic
    across repeats" -- the obvious way to validate a task before it reaches anyone -- proves
    a task is *well-formed* and says nothing about whether it is *trivial*. Both defects
    found in this corpus by hand passed all three of those checks:

      * `tc-log-rotation-order` shipped a NOTES.md hint that made 86% of plausible orderings
        produce the graded-correct answer;
      * `lh-i18n-catalog-parity` listed its objectives so that a single forward pass in the
        prompt's own order satisfied all five checkpoints and the published check.

    Neither is visible from a fresh workspace failing and a correct solution passing, because
    both of those stayed true throughout. The only thing that catches them is executing the
    lazy strategy and finding that it wins. So each trap a task claims in a comment gets a
    script here, and the sweep turns the claim into an assertion that runs.

    `expectation` separates the two useful outcomes. `fails_public` is a shortcut the
    published check already rejects. `passes_public_fails_hidden` is the more valuable kind:
    an overfit path caught only by the withheld check, which is what `overfit_rate` measures.
    The second is only decidable where the withheld tree is present, so the sweep reports it
    unresolved rather than passing when it cannot be checked.
    """

    shortcut_id: str
    apply: str
    expectation: str = FAILS_PUBLIC
    description: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any], *, origin: str = "<memory>") -> Shortcut:
        missing = [key for key in ("shortcut_id", "apply") if not record.get(key)]
        if missing:
            raise TaskError(f"{origin}: shortcut is missing required field(s): {', '.join(missing)}")
        expectation = str(record.get("expectation") or FAILS_PUBLIC)
        if expectation not in SHORTCUT_EXPECTATIONS:
            raise TaskError(
                f"{origin}: shortcut {record['shortcut_id']!r} expectation {expectation!r} is not one of "
                f"{', '.join(SHORTCUT_EXPECTATIONS)}"
            )
        return cls(
            shortcut_id=str(record["shortcut_id"]),
            apply=str(record["apply"]),
            expectation=expectation,
            description=str(record.get("description") or ""),
        )


@dataclass(frozen=True)
class Task:
    """One HermesBench task.

    `verify` is run after the agent finishes, in the task workspace; exit code 0 means
    the task was actually accomplished. `setup` prepares that workspace beforehand.
    Neither is ever shown to the agent -- an agent that can read its own grader is not
    being graded.
    """

    task_id: str
    prompt: str
    verify: str
    tools: tuple[str, ...]
    setup: str | None = None
    timeout_s: int = 600
    max_steps: int = 40
    tags: tuple[str, ...] = ()
    mutating_tools: tuple[str, ...] = DEFAULT_MUTATING_TOOLS
    # Environment overlaid on every subprocess for this task (setup, agent tool calls,
    # and verification alike). Values go through shell-style variable expansion, so
    # `PATH: "./bin:$PATH"` prepends rather than replacing. Tasks use this to make a
    # failure mode real -- shadowing a binary, pinning a locale, clearing a token --
    # instead of merely describing one in the prompt.
    env: dict[str, str] = field(default_factory=dict)
    # Sub-objectives sampled during the episode (long-horizon tasks). Empty for the
    # short v0 tasks, whose whole objective is the single `verify`.
    checkpoints: tuple[Checkpoint, ...] = ()
    # How often to sample checkpoints, in agent steps. Each sample runs every
    # checkpoint command, so this trades measurement resolution against wall time.
    checkpoint_every: int = 5
    # Read-only tools an agent uses to check its own work. Calls to these draw on
    # `max_verification_steps` instead of `max_steps`.
    #
    # Without this split the harness contradicts itself: `self_check_rate` rewards an
    # agent for re-running the tests, while a single shared budget charges it for doing
    # so and can cut the episode off mid-verification. Open models RL-trained to spend
    # tokens on checking their work are exactly the ones that would be penalised.
    verification_tools: tuple[str, ...] = ()
    max_verification_steps: int = 40
    # Deliberation draws on its own allowance, for the same reason verification does.
    #
    # `agent_steps` used to be incremented for every step that was not a tool_result, so a THINKING
    # step cost exactly as much as an action. Measured over a 19-task run: 300 thinking steps against
    # 318 tool calls -- 49% of the budget spent on reasoning, so `max_steps: 15` meant about 7
    # actions. Every failing episode in that run ended on `step budget exhausted`, and 5 of the 9
    # capped episodes had already passed.
    #
    # That charges a model for thinking, which is worse here than it looks: the pinned model's own
    # chat template sets `Reasoning strength: high` and returns deliberation on a separate channel, so
    # it *always* emits a THINKING step before a call. It cannot spend the budget the way a
    # non-reasoning model can.
    #
    # Bounded rather than free: a model that only ever thinks has to terminate, and the loop's stall
    # detection cannot see the difference between deliberating and hanging.
    max_reasoning_steps: int = 40
    # Files the agent must not touch: tests, baselines, fixtures. Hashed before and
    # after the episode, and any change disqualifies the run outright. This is the
    # general form of "fix the source, not the test" -- a grader that greps for the
    # expected value can be satisfied by a comment, but a digest cannot.
    protected_paths: tuple[str, ...] = ()
    # Checks withheld from the public release of this task. `verify` is published with
    # the benchmark and can therefore be trained against; `hidden_verify` is kept by
    # maintainers and is what the real score depends on.
    #
    # This is the anti-saturation mechanism. Both are equally invisible to the agent at
    # run time -- the task file is never in the workspace -- so hiding buys nothing
    # against a single run. What it buys is protection for an OPEN benchmark, where a
    # published verifier is something a model can be optimised against and a withheld
    # one is not. A model that passes public and fails hidden has learned the benchmark
    # rather than the job, and that gap is reported rather than hidden.
    hidden_verify: str = ""
    # Plausible-lazy strategies this task claims to catch, written as scripts so the claim is
    # executable. A task whose header describes a trap and declares no shortcut has an
    # untested claim -- which is how both defects in this corpus shipped. See `Shortcut`.
    shortcuts: tuple[Shortcut, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hidden_tests(self) -> bool:
        """Whether the withheld check can actually be RUN here."""
        return bool(self.hidden_verify.strip())

    @property
    def hidden_verify_commitment(self) -> str:
        """The published salted digest of this task's withheld check, if any."""
        return str(self.metadata.get("hidden_verify_commitment") or "")

    @property
    def declares_hidden_tests(self) -> bool:
        """Whether a withheld check EXISTS, whether or not this checkout holds it.

        Separate from `has_hidden_tests` on purpose, and the distinction is the whole
        point of splitting the suite across a public and a private tree. A task published
        with its check redacted keeps a commitment, so it still declares one -- and a
        checkout without the private overlay must report "cannot score this" rather than
        "this task has no withheld check". Those two are the same boolean if you only ask
        `has_hidden_tests`, and collapsing them turns `overfit_rate` from unavailable into
        a confident zero, which is the more dangerous of the two answers.
        """
        return self.has_hidden_tests or bool(self.hidden_verify_commitment)

    @property
    def withheld_check_missing(self) -> bool:
        """A withheld check is declared and this checkout does not have it."""
        return bool(self.hidden_verify_commitment) and not self.has_hidden_tests

    @property
    def category(self) -> str:
        """The Hermes capability this task exercises: its first recognised tag.

        Read from tags rather than a separate field so the existing tag filter doubles
        as the category selector and old tasks stay loadable.
        """
        from hermesbench import HERMES_CATEGORIES

        for tag in self.tags:
            if tag in HERMES_CATEGORIES:
                return tag
        return ""

    @property
    def is_long_horizon(self) -> bool:
        return bool(self.checkpoints)

    @classmethod
    def from_record(cls, record: dict[str, Any], *, origin: str = "<memory>") -> Task:
        missing = [key for key in ("task_id", "prompt", "verify", "tools") if not record.get(key)]
        if missing:
            raise TaskError(f"{origin}: task is missing required field(s): {', '.join(missing)}")
        tools = tuple(str(t) for t in record["tools"])
        mutating = record.get("mutating_tools")
        raw_env = record.get("env") or {}
        if not isinstance(raw_env, dict):
            raise TaskError(f"{origin}: task {record['task_id']!r} env must be a mapping")

        raw_checkpoints = record.get("checkpoints") or []
        if not isinstance(raw_checkpoints, list):
            raise TaskError(f"{origin}: task {record['task_id']!r} checkpoints must be a list")
        checkpoints = tuple(Checkpoint.from_record(c, origin=origin) for c in raw_checkpoints)
        seen_ids: set[str] = set()
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id in seen_ids:
                # Duplicate ids would silently collapse in the pass/fail timeline, so a
                # regression on one could be masked by the other still passing.
                raise TaskError(f"{origin}: duplicate checkpoint_id {checkpoint.checkpoint_id!r}")
            seen_ids.add(checkpoint.checkpoint_id)

        checkpoint_every = int(record.get("checkpoint_every", 5))
        if checkpoints and checkpoint_every < 1:
            raise TaskError(f"{origin}: checkpoint_every must be >= 1, got {checkpoint_every}")

        raw_shortcuts = record.get("shortcuts") or []
        if not isinstance(raw_shortcuts, list):
            raise TaskError(f"{origin}: task {record['task_id']!r} shortcuts must be a list")
        shortcuts = tuple(Shortcut.from_record(s, origin=origin) for s in raw_shortcuts)
        seen_shortcuts: set[str] = set()
        for shortcut in shortcuts:
            if shortcut.shortcut_id in seen_shortcuts:
                # Duplicate ids would collapse in the sweep report, so a shortcut that started
                # winning could hide behind a same-named one that still loses.
                raise TaskError(f"{origin}: duplicate shortcut_id {shortcut.shortcut_id!r}")
            seen_shortcuts.add(shortcut.shortcut_id)

        return cls(
            task_id=str(record["task_id"]),
            prompt=str(record["prompt"]),
            verify=str(record["verify"]),
            tools=tools,
            setup=record.get("setup"),
            timeout_s=int(record.get("timeout_s", 600)),
            max_steps=int(record.get("max_steps", 40)),
            max_reasoning_steps=int(record.get("max_reasoning_steps", record.get("max_steps", 40) * 2)),
            tags=tuple(str(t) for t in record.get("tags", ())),
            # `is None`, not falsiness: an explicit `mutating_tools: []` means "nothing
            # this task offers mutates", which must not silently get the default back.
            mutating_tools=DEFAULT_MUTATING_TOOLS if mutating is None else tuple(str(t) for t in mutating),
            env={str(k): str(v) for k, v in raw_env.items()},
            checkpoints=checkpoints,
            checkpoint_every=checkpoint_every,
            verification_tools=tuple(str(t) for t in record.get("verification_tools", ())),
            protected_paths=tuple(str(p) for p in record.get("protected_paths", ())),
            hidden_verify=str(record.get("hidden_verify") or ""),
            shortcuts=shortcuts,
            max_verification_steps=int(record.get("max_verification_steps", 40)),
            metadata=record.get("metadata") or {},
        )


def load_task(path: Path) -> Task:
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TaskError(f"{path}: expected a YAML mapping, got {type(record).__name__}")
    return Task.from_record(record, origin=str(path))


def iter_tasks(version: str = "v0", root: Path | None = None) -> Iterator[Task]:
    """Yield every task in a bench version, ordered by filename for reproducibility."""
    directory = (root or TASKS_ROOT) / version
    if not directory.is_dir():
        raise TaskError(f"no such bench version: {directory}")
    # Both spellings: a `.yml` task silently vanishing from the suite is worse than
    # being strict about the extension, because the run still reports a clean result.
    paths = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    for path in paths:
        yield load_task(path)


def suite_versions(root: Path | None = None) -> list[str]:
    """Every bench version on disk, in sorted order."""
    base = root or Path(__file__).resolve().parent / "tasks"
    return sorted(d.name for d in base.iterdir() if d.is_dir() and any(d.glob("*.yaml")))


def load_suite(version: str = "v0", root: Path | None = None, tags: tuple[str, ...] = ()) -> list[Task]:
    """Load one or more bench versions, optionally filtered to tasks carrying any of `tags`.

    `version` accepts a comma-separated list or the literal `all`. A single version was the
    original shape and it cannot express the scorecard the suite advertises: v0 holds three
    tasks covering two of the four capability categories, so no single invocation could ever
    produce a four-category result. Spanning versions is not a convenience -- it is the
    difference between a partial reading and the one the categories were defined for.
    """
    requested = (
        suite_versions(root) if version.strip() == "all" else [v.strip() for v in version.split(",") if v.strip()]
    )
    if not requested:
        raise TaskError("no bench version requested")

    tasks: list[Task] = []
    for name in requested:
        tasks.extend(iter_tasks(name, root))

    # Uniqueness is checked across the whole selection *before* filtering: task_id names the
    # workspace directory in run_suite, so a collision hidden by the current tag filter
    # would surface later as two tasks sharing one workspace. Across versions this matters
    # more, not less -- v0 and v1 are free to reuse an id until they are loaded together.
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise TaskError(f"duplicate task_id {task.task_id!r} across suite {', '.join(requested)}")
        seen.add(task.task_id)

    if tags:
        wanted = set(tags)
        tasks = [t for t in tasks if wanted & set(t.tags)]
    return tasks

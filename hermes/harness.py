"""Harness and suite digests: making a benchmark claim something a stranger can check.

The repo already has the vocabulary. `Tournament` refuses a field whose candidates
disagree on `harness_digest`, because comparing runs from different harnesses measures
model x harness rather than model. `HarnessPin` names every digest that can change what
an agent observes -- system prompt, tool schemas, container image, dependency lock.

Nothing computes any of them. `harness_digest` is a string the caller passes in, so the
fair-fight invariant is enforced against a label: two genuinely different harnesses both
labelled `"h1"` compare as equal, and one harness labelled differently by two runners
compares as unequal. An invariant defended by a field nobody populates is a comment.

This module computes them, from the things that actually decide the outcome.

**What goes in is exactly what can change the result.** Include less and two different
harnesses agree; include more -- a workspace path, a timestamp, a run id -- and every run
is its own harness, the fair-fight check rejects everything, and the invariant fails
closed into uselessness. That is not the safe direction it sounds like: a check that
always refuses gets deleted.

**Withheld tests are digested, not published, and the digest is salted.** A manifest that
publishes `hidden_verify` defeats the point. A manifest that publishes its bare SHA-256
defeats it too, more slowly: withheld checks are short shell commands drawn from a small
space -- `pytest tests/test_hidden.py -q` and a few hundred neighbours -- so an unsalted
digest is a crossword, not a commitment. Salted, the digest proves two runs used the *same*
withheld check without revealing which. It does **not** prove the check is what its author
claimed; only revealing the salt does that, which is what a post-hoc audit is for. The
manifest says which of those two properties it is offering, because the difference is the
whole question a reader has.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# What a published digest promises.
PROVES_CONSISTENCY = "consistency"
PROVES_CONTENT = "content"


class HarnessError(ValueError):
    """A digest cannot be computed, or would claim more than it establishes."""


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_mapping(payload: dict[str, Any]) -> str:
    """Digest a mapping through a canonical encoding.

    Sorted keys and fixed separators, so two runs that built the same content in a
    different order agree. Without canonicalisation the digest would fingerprint dict
    insertion order, which no part of the outcome depends on.
    """
    # ASCII-escaped, matching the encoder the sealed verifier uses. With ensure_ascii=False
    # here and True there, any task id or prompt outside ASCII digests differently inside
    # the enclave than outside it, and an honest batch fails its own attested check.
    return digest_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def salted_digest(text: str, salt: str) -> str:
    """Digest withheld content under a salt.

    The salt is required and must be substantial. A withheld check is a short command from
    a small space, so an unsalted digest can simply be guessed against -- the commitment
    would be readable by anyone willing to enumerate a few hundred plausible pytest
    invocations.
    """
    if len(salt) < 16:
        raise HarnessError(
            "a withheld-check digest needs a salt of at least 16 characters; these are short "
            "commands from a small space, and an unsalted digest can be brute-forced back to "
            "its content, which is the thing being withheld"
        )
    return digest_text(f"{salt}\x00{text}")


@dataclass(frozen=True)
class TaskFingerprint:
    """The parts of a task that decide what a run of it produces.

    Deliberately not the whole record. `tags` select which tasks run and do not change what
    any of them does; `metadata` is annotation. Folding those in would mean re-labelling a
    task changed the harness, and every prior result would stop being comparable for a
    reason that has nothing to do with the agent.
    """

    task_id: str
    prompt: str
    verify: str
    setup: str = ""
    tools: tuple[str, ...] = ()
    timeout_s: int = 0
    max_steps: int = 0
    max_verification_steps: int = 0
    verification_tools: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    mutating_tools: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    checkpoints: tuple[tuple[str, str], ...] = ()
    checkpoint_every: int = 0
    # Salted digest of the withheld check, never its content.
    hidden_verify_digest: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "verify": self.verify,
            "setup": self.setup,
            # Sorted: the order a task lists its tools does not change what it permits.
            "tools": sorted(self.tools),
            "timeout_s": self.timeout_s,
            "max_steps": self.max_steps,
            "max_verification_steps": self.max_verification_steps,
            "verification_tools": sorted(self.verification_tools),
            "protected_paths": sorted(self.protected_paths),
            "mutating_tools": sorted(self.mutating_tools),
            "env": dict(self.env),
            # Checkpoints run in order, so theirs is preserved.
            "checkpoints": [list(c) for c in self.checkpoints],
            "checkpoint_every": self.checkpoint_every,
            "hidden_verify_digest": self.hidden_verify_digest,
        }

    @property
    def digest(self) -> str:
        return digest_mapping(self.to_payload())


def fingerprint_task(task: Any, *, salt: str = "") -> TaskFingerprint:
    """Fingerprint a `hermesbench.tasks.Task`.

    A task with a withheld check requires a salt: without one the fingerprint would either
    omit the check -- so two suites differing only in their withheld tests would digest
    identically, and the digest would not detect the substitution it exists to detect --
    or include a guessable digest of it.
    """
    hidden = getattr(task, "hidden_verify", "") or ""
    if hidden and not salt:
        raise HarnessError(
            f"{task.task_id}: task has a withheld check but no salt was given; omitting it would "
            "let two suites with different withheld tests digest the same, and including it "
            "unsalted would publish it"
        )
    checkpoints = tuple((c.checkpoint_id, c.verify) for c in getattr(task, "checkpoints", ()) or ())
    return TaskFingerprint(
        task_id=task.task_id,
        prompt=task.prompt,
        verify=task.verify,
        setup=getattr(task, "setup", "") or "",
        tools=tuple(getattr(task, "tools", ()) or ()),
        timeout_s=int(getattr(task, "timeout_s", 0) or 0),
        max_steps=int(getattr(task, "max_steps", 0) or 0),
        max_verification_steps=int(getattr(task, "max_verification_steps", 0) or 0),
        verification_tools=tuple(getattr(task, "verification_tools", ()) or ()),
        protected_paths=tuple(getattr(task, "protected_paths", ()) or ()),
        mutating_tools=tuple(getattr(task, "mutating_tools", ()) or ()),
        env=dict(getattr(task, "env", {}) or {}),
        checkpoints=checkpoints,
        checkpoint_every=int(getattr(task, "checkpoint_every", 0) or 0),
        hidden_verify_digest=salted_digest(hidden, salt) if hidden else "",
    )


@dataclass(frozen=True)
class SuiteDigest:
    """A content address for the exact set of tasks a claim was measured on."""

    name: str
    digest: str
    task_count: int
    task_digests: dict[str, str] = field(default_factory=dict)
    withheld_tasks: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "task_count": self.task_count,
            "withheld_tasks": self.withheld_tasks,
        }


def digest_suite(name: str, fingerprints: list[TaskFingerprint]) -> SuiteDigest:
    """Digest a suite.

    Task ids must be unique and the set is digested sorted, so two runs that loaded the
    same suite in a different order agree. Execution order can change a *result* only
    through leakage between tasks, which is a defect rather than a property of the suite.
    """
    if not fingerprints:
        raise HarnessError(f"{name}: a suite digest over no tasks would certify nothing")
    ids = [f.task_id for f in fingerprints]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise HarnessError(f"{name}: duplicate task ids {duplicates}; the suite cannot be addressed")
    per_task = {f.task_id: f.digest for f in fingerprints}
    return SuiteDigest(
        name=name,
        digest=digest_mapping({"name": name, "tasks": dict(sorted(per_task.items()))}),
        task_count=len(fingerprints),
        task_digests=per_task,
        withheld_tasks=sum(1 for f in fingerprints if f.hidden_verify_digest),
    )


def harness_digest(
    pin: Any,
    *,
    suite: SuiteDigest,
    executor: str,
    trace_schema: str = "",
    observation_limit: int = 0,
    tool_timeout_s: int = 0,
    selection_policy: Any = None,
) -> str:
    """The digest `Tournament` compares candidates on.

    Refuses an unpinned harness. `HarnessPin.is_pinned` already states what a pin needs --
    a commit, a tool-schema digest, a system-prompt digest -- and a digest computed over
    blanks is a stable, meaningful-looking value that certifies nothing, which is worse
    than no digest at all because it passes the fair-fight check.
    """
    if not getattr(pin, "is_pinned", False):
        raise HarnessError(
            "harness is not pinned (needs commit, tool_schema_digest and system_prompt_digest); "
            "digesting blanks yields a stable value that certifies nothing and would pass the "
            "fair-fight check while comparing two different harnesses"
        )
    if not executor:
        raise HarnessError(
            "harness digest needs the executor; the same tasks under a different executor are a different harness"
        )
    return digest_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "suite": suite.digest,
            "executor": executor,
            # Two harness parameters that live in code rather than in any task file, and
            # decide outcomes anyway. The observation limit *is* the agent's window on the
            # world -- double it and the agent sees different output, acts differently and
            # scores differently. The tool timeout decides which builds and installs finish.
            # Left out of the digest, either could change between two runs that compare as
            # the same harness.
            "observation_limit": observation_limit,
            "tool_timeout_s": tool_timeout_s,
            # The tie-break rules. Reordering them decides which teacher wins every
            # tournament whose leaders are incomparable, so a reordered policy is a
            # different harness and prior results become incomparable rather than
            # quietly re-rankable -- which is the point of SelectionPolicy having a digest.
            "selection_policy_digest": getattr(selection_policy, "digest", ""),
            "trace_schema": trace_schema or getattr(pin, "trace_schema", ""),
            "commit": pin.commit,
            "system_prompt_digest": pin.system_prompt_digest,
            "tool_schema_digest": pin.tool_schema_digest,
            "container_image_digest": getattr(pin, "container_image_digest", ""),
            "dependency_lock_digest": getattr(pin, "dependency_lock_digest", ""),
        }
    )


@dataclass(frozen=True)
class TaskResult:
    """One task's outcome, at the grain a reader can spot-check."""

    task_id: str
    passed: bool
    hidden_passed: bool | None = None
    disqualified: bool = False
    steps: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "hidden_passed": self.hidden_passed,
            "disqualified": self.disqualified,
            "steps": self.steps,
        }


@dataclass(frozen=True)
class RunManifest:
    """A benchmark claim, in the form that lets someone else check it.

    Per-task results are required rather than optional. An aggregate alone cannot be
    spot-checked: a reader who disagrees with 82.4% has nothing to disagree *with*, and
    cannot find the handful of tasks carrying it. Publishing the grain is most of what
    separates a reproducible claim from a screenshot.
    """

    model: str
    suite: SuiteDigest
    harness: str
    results: tuple[TaskResult, ...]
    proves: str = PROVES_CONSISTENCY
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model:
            raise HarnessError("a run manifest without a model names no subject; the number is about nothing")
        if not self.harness:
            raise HarnessError("a run manifest needs a harness digest, or its number cannot be compared to any other")
        if not self.results:
            raise HarnessError(
                "a run manifest needs per-task results; an aggregate alone cannot be spot-checked, "
                "and a reader who doubts the headline has nothing to examine"
            )
        if self.proves not in (PROVES_CONSISTENCY, PROVES_CONTENT):
            raise HarnessError(f"unknown claim strength {self.proves!r}")
        reported = {r.task_id for r in self.results}
        expected = set(self.suite.task_digests)
        if expected and reported != expected:
            # Reporting a subset while naming the whole suite is how a partial run becomes
            # a headline: the denominator says 1000 and the numerator saw 40.
            missing = sorted(expected - reported)
            extra = sorted(reported - expected)
            raise HarnessError(
                f"results do not cover the suite: missing {missing[:5]}, unexpected {extra[:5]}; "
                "a partial run published under the suite's name misstates its own denominator"
            )

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.disqualified)

    @property
    def success_rate(self) -> float:
        return self.passed / len(self.results)

    @property
    def withheld_claim(self) -> str:
        """What the withheld-check digests in this manifest actually establish."""
        if not self.suite.withheld_tasks:
            return "no withheld checks in this suite"
        if self.proves == PROVES_CONTENT:
            return "salt revealed: the withheld checks can be confirmed to be the ones claimed"
        return (
            "salted digests only: two runs can be confirmed to have used the same withheld checks, "
            "but not that those checks are the ones their author described"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "harness": self.harness,
            "suite": self.suite.to_record(),
            "proves": self.proves,
            "withheld_claim": self.withheld_claim,
            "success_rate": round(self.success_rate, 4),
            "passed": self.passed,
            "metrics": {k: round(v, 4) for k, v in sorted(self.metrics.items())},
            "notes": self.notes,
            "results": [r.to_record() for r in self.results],
        }


def comparable(a: RunManifest, b: RunManifest) -> tuple[bool, str]:
    """Whether two claims can be put side by side, and if not, why not.

    Returns a reason rather than raising, because "these are not comparable" is usually the
    most informative thing a leaderboard can say about two numbers. Silently ranking them
    is how a harness change becomes a model improvement.
    """
    if a.suite.digest != b.suite.digest:
        return False, f"different suites: {a.suite.name}@{a.suite.digest[:14]} vs {b.suite.name}@{b.suite.digest[:14]}"
    if a.harness != b.harness:
        return False, f"different harnesses: {a.harness[:14]} vs {b.harness[:14]}; the gap includes the harness"
    if a.model == b.model:
        return False, f"both manifests are {a.model}; there is nothing to compare"
    return True, ""

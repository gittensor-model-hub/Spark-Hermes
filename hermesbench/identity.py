"""Task identity: keeping an endless generator from flooding the arena with near-copies.

A generator that produces unlimited tasks is only an asset if the tasks are *different*.
`hermesbench.mutation` can emit a task per mutation site, which means one source file
yields a dozen variations of the same debugging exercise -- and a benchmark of a thousand
easy variants measures one skill a thousand times while reading as broad coverage.

Three kinds of duplicate, in descending order of how reliably they can be detected:

**Execution duplicate (exact).** Same verifier against the same environment is the same
task wearing different prose. Both are hashed, so this is a decision, not an estimate.

**Objective near-duplicate (lexical, approximate).** Two prompts asking for the same thing
in different words, *against the same environment*. The environment qualifier is
load-bearing and was learned the hard way: every mutation-generated task carries the same
generic prompt ("a bug was introduced into X, find and fix it"), so prompt similarity
alone scored 1.00 across eleven genuinely different bugs and would have discarded ten of
them. Shared wording plus different broken code is not a duplicate -- the work differs
even though the request reads the same.
This is a *proxy* and it is honest about that: it will not catch "fix the memory leak"
against "resolve the RAM leak" in two different codebases, because nothing here
understands words and their environments differ anyway. `DuplicateIndex` takes an optional
`embedder` so real semantic dedup can be dropped in without changing the callers.

**Coverage saturation.** Neither of the above catches twelve genuinely-distinct tasks that
all exercise the same narrow skill. That is what the coverage matrix is for: tasks are
counted per (category, topic) bucket and a bucket that hits its cap stops accepting new
ones, however novel each individual task looks.

The last one is the one that actually prevents the failure mode. The first two catch
copies; only the coverage matrix catches *monotony*.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# Why a candidate task was refused.
EXECUTION_DUPLICATE = "execution_duplicate"
OBJECTIVE_DUPLICATE = "objective_duplicate"
BUCKET_SATURATED = "bucket_saturated"

# Default similarity above which two objectives are treated as the same request. Chosen
# to be conservative: wrongly rejecting a genuinely new task silently shrinks the arena,
# and a false accept is visible in the coverage matrix while a false reject is not.
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Words carrying no discriminating signal in a task prompt.
_STOPWORDS = frozenset(
    """a an the this that these those and or but if then than so of in on at to for with from by
    is are was were be been being do does did done have has had you your it its as into over under
    please make sure must should will can could would may might not no yours""".split()
)

_TOKEN = re.compile(r"[a-z0-9_]+")


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def normalize_objective(text: str) -> tuple[str, ...]:
    """Lowercased content tokens, deduplicated, ordered.

    Order is dropped deliberately: "fix the bug, then run the tests" and "run the tests,
    and fix the bug" are the same request, and comparing sequences would call them
    different.

    There is no stemming, so "fix" and "fixing" remain distinct tokens. That is a real
    limitation of the lexical proxy rather than an oversight -- handling it properly means
    an embedder, which is why one can be supplied.
    """
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]
    return tuple(sorted(set(tokens)))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True)
class TaskIdentity:
    """Content-addressed identity of a task, split so near-misses are diagnosable.

    Separate hashes rather than one: knowing that two tasks share a verifier but differ in
    objective is actionable, whereas a single digest can only say "not identical".
    """

    task_hash: str
    objective_hash: str
    verifier_hash: str
    environment_hash: str
    tokens: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "objective_hash": self.objective_hash,
            "verifier_hash": self.verifier_hash,
            "environment_hash": self.environment_hash,
        }


def identity_of(task: Any) -> TaskIdentity:
    """Derive an identity from anything exposing prompt/verify/setup/tools/env.

    Accepts a `hermesbench.tasks.Task` or a plain record, so generated tasks can be
    checked before they are ever materialized as files.
    """

    def _get(name: str, default: Any = "") -> Any:
        if isinstance(task, dict):
            return task.get(name, default)
        return getattr(task, name, default)

    prompt = str(_get("prompt"))
    verify = str(_get("verify"))
    setup = str(_get("setup") or "")
    tools = tuple(sorted(str(t) for t in _get("tools", ()) or ()))
    env = _get("env", {}) or {}
    env_repr = ";".join(f"{k}={env[k]}" for k in sorted(env))

    tokens = normalize_objective(prompt)
    objective_hash = _digest(" ".join(tokens))
    verifier_hash = _digest(verify.strip())
    environment_hash = _digest(setup.strip(), ",".join(tools), env_repr)

    return TaskIdentity(
        task_hash=_digest(objective_hash, verifier_hash, environment_hash),
        objective_hash=objective_hash,
        verifier_hash=verifier_hash,
        environment_hash=environment_hash,
        tokens=tokens,
    )


@dataclass(frozen=True)
class Rejection:
    reason: str
    detail: str
    conflicts_with: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail, "conflicts_with": self.conflicts_with}


@dataclass
class CoverageMatrix:
    """How many accepted tasks sit in each (category, topic) bucket.

    The cap is what stops an endless generator producing a thousand variations of one
    exercise. Without it, dedup only guarantees the tasks are not *copies* -- it says
    nothing about whether the suite measures more than one skill.
    """

    cap_per_bucket: int = 25
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def count(self, category: str, topic: str) -> int:
        return self.counts.get((category, topic), 0)

    def saturated(self, category: str, topic: str) -> bool:
        return self.count(category, topic) >= self.cap_per_bucket

    def record(self, category: str, topic: str) -> None:
        self.counts[(category, topic)] = self.count(category, topic) + 1

    @property
    def buckets(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def thin_buckets(self, minimum: int = 5) -> tuple[tuple[str, str], ...]:
        """Buckets with too few tasks -- where generation effort is actually needed."""
        return tuple(sorted(k for k, v in self.counts.items() if v < minimum))

    def to_record(self) -> dict[str, Any]:
        return {
            "cap_per_bucket": self.cap_per_bucket,
            "buckets": self.buckets,
            "total": self.total,
            "counts": {f"{c}/{t}": n for (c, t), n in sorted(self.counts.items())},
        }


class DuplicateIndex:
    """Accepts or refuses candidate tasks, and explains every refusal.

    `embedder` is the hook for real semantic dedup: any callable turning a prompt into a
    vector. Absent one, similarity is lexical overlap, which is a weaker check and is
    documented as such rather than presented as understanding.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        coverage: CoverageMatrix | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        objective_dedup_requires_same_environment: bool = True,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        # Two tasks whose environments differ are different work, however alike the
        # prompts read. Turning this off is for suites whose prompts carry the whole
        # objective and whose setups are incidental.
        self.objective_dedup_requires_same_environment = objective_dedup_requires_same_environment
        self.coverage = coverage or CoverageMatrix()
        self.embedder = embedder
        self._by_execution: dict[tuple[str, str], str] = {}
        self._accepted: list[tuple[str, TaskIdentity, list[float] | None]] = []

    def check(self, task: Any, *, category: str = "", topic: str = "") -> Rejection | None:
        """Why this task cannot be added, or None if it can."""
        identity = identity_of(task)

        execution_key = (identity.verifier_hash, identity.environment_hash)
        clash = self._by_execution.get(execution_key)
        if clash is not None:
            return Rejection(
                EXECUTION_DUPLICATE,
                "same verifier against the same environment; the prose differs, the task does not",
                clash,
            )

        vector = self.embedder(_prompt_of(task)) if self.embedder else None
        for existing_id, existing, existing_vector in self._accepted:
            if (
                self.objective_dedup_requires_same_environment
                and existing.environment_hash != identity.environment_hash
            ):
                continue
            score = (
                _cosine(vector, existing_vector)
                if vector is not None and existing_vector is not None
                else jaccard(identity.tokens, existing.tokens)
            )
            if score >= self.similarity_threshold:
                return Rejection(OBJECTIVE_DUPLICATE, f"objective similarity {score:.2f}", existing_id)

        if category and topic and self.coverage.saturated(category, topic):
            return Rejection(
                BUCKET_SATURATED,
                f"{category}/{topic} already holds {self.coverage.count(category, topic)} tasks "
                f"(cap {self.coverage.cap_per_bucket}); more here measures the same skill again",
            )

        return None

    def add(self, task: Any, *, category: str = "", topic: str = "") -> Rejection | None:
        """Accept a task into the index, or return why it was refused."""
        rejection = self.check(task, category=category, topic=topic)
        if rejection is not None:
            return rejection

        identity = identity_of(task)
        task_id = str(task.get("task_id") if isinstance(task, dict) else getattr(task, "task_id", ""))
        self._by_execution[(identity.verifier_hash, identity.environment_hash)] = task_id
        self._accepted.append((task_id, identity, self.embedder(_prompt_of(task)) if self.embedder else None))
        if category and topic:
            self.coverage.record(category, topic)
        return None

    def __len__(self) -> int:
        return len(self._accepted)

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        return tuple(task_id for task_id, _, _ in self._accepted)


def _prompt_of(task: Any) -> str:
    return str(task.get("prompt", "") if isinstance(task, dict) else getattr(task, "prompt", ""))


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def deduplicate(
    tasks: Iterable[Any],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    cap_per_bucket: int = 25,
    category_of: Callable[[Any], str] | None = None,
    topic_of: Callable[[Any], str] | None = None,
    embedder: Callable[[str], list[float]] | None = None,
) -> tuple[list[Any], list[tuple[str, Rejection]], CoverageMatrix]:
    """Filter a stream of candidate tasks. Returns (kept, rejected, coverage).

    Rejections are returned rather than logged and dropped: a generator whose output is
    90% refused is telling you its operators have run out of distinct things to break,
    and that is worth seeing.
    """
    index = DuplicateIndex(
        similarity_threshold=similarity_threshold,
        coverage=CoverageMatrix(cap_per_bucket=cap_per_bucket),
        embedder=embedder,
    )
    kept: list[Any] = []
    rejected: list[tuple[str, Rejection]] = []

    for task in tasks:
        category = category_of(task) if category_of else ""
        topic = topic_of(task) if topic_of else ""
        rejection = index.add(task, category=category, topic=topic)
        if rejection is None:
            kept.append(task)
        else:
            task_id = str(task.get("task_id") if isinstance(task, dict) else getattr(task, "task_id", ""))
            rejected.append((task_id, rejection))

    return kept, rejected, index.coverage

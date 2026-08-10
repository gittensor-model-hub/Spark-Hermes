"""The pinned teacher registry: which models generate the corpus, and what may be done with it.

Three models were chosen as dataset teachers. Checking each against its provider's own
documentation turned up constraints that decide the design, so they are encoded here as
data the pipeline reads rather than as a note somebody has to remember.

**Rights are a decision, recorded here, not a judgement this module makes.** The field is
Qwen 3.8 Max and DeepSeek V4 Pro, both `RIGHTS_APPROVED` on the maintainer's licence
review -- Qwen on 2026-08-08, DeepSeek on 2026-08-09. `rights_basis` records who decided and on what, because "approved" with no
provenance is indistinguishable from "nobody checked", and the two have very different
consequences a year later when someone asks where the corpus came from.

Claude Fable 5 is declared and deliberately outside the field. Anthropic prohibits using
outputs as training targets absent written permission and names general-purpose
text-generation models specifically, so it is `RIGHTS_DENIED` until that permission
exists. It is kept rather than deleted because the declaration is the record of why it is
absent, and because `build_artifacts` can already carry a rights-denied teacher as router
evidence without letting it become a trained token -- so adding it back for capability
comparison is one line and stays safe.

**Rights attach to an artefact, not to a model.** Kimi K3 is declared twice and the two
entries disagree, deliberately. The open-weights licence grants dealing in the weights
without restriction and reserves nothing over outputs; Moonshot's OpenPlatform terms, which
govern the hosted API instead, bar building models that could compete with the service. Same
model, two agreements, opposite answers. `kimi-k3` was declared `RIGHTS_APPROVED` on a review
of the licence while pointing at the API -- the review was real and it was of the wrong
document. `rights_conflicts()` now reports any model declared under more than one set of
rights, because arriving at that state deliberately is correct and arriving at it by accident
is how the corpus acquires a row nobody may train on.

**Rights and reproducibility are separate axes, and only one of them is now settled.**
Neither hosted endpoint in the field can be pinned: Alibaba publishes no dated snapshot for
`qwen3.8-max`, and Moonshot dropped the dated convention it used for K2. Both ids can shift
under a corpus that was already generated, and nothing in the row would show it. Kimi K3 is
open weights, so `kimi_k3_self_hosted(revision=...)` is the only configuration here that a
row could be regenerated from. `check_field` says so on every run rather than once in a
docstring.

**Logprobs are measured per endpoint, and the first measurement here was wrong in both
directions.** The docs say Kimi K3 supports logprobs and Qwen's `-max` series does not. An
earlier probe recorded that the gateway returned an empty object for both, and the registry
was written from it. Re-probed against the live gateway on 2026-08-09:

    qwen3.8-max      logprobs present, 9 tokens for a 9-token answer, top_logprobs populated,
                     and the stream reconstructs `content` exactly
    kimi-k3          no `logprobs` key at all -- absent, not empty
    kimi-k2.7-code   logprobs present, 58 tokens for the same 9-token answer, and the stream
                     reconstructs the *reasoning*: "The user wants me to count from 1 to 5..."

So `qwen3.8-max` is a logit teacher today and `kimi-k3` is not, which is the opposite of
both the docs and the previous entry.

The third row is why `logprobs` became two fields. A boolean would call qwen and
kimi-k2.7-code equally capable while their streams describe different text, and a corpus
mixing them would align one model's answer distribution against another's private thinking.
Nothing errors: both are well-formed `logprobs.content` arrays. `logprob_scope` records what
the stream is aligned to, and `may_distil_tokens` requires `LOGPROB_CONTENT`.

Reproducibility is still unsolved and is now the *only* thing blocking a replayable logit
corpus: `qwen3.8-max` publishes no dated snapshot, so its logprobs are usable and its rows
are not regenerable. Self-hosting open weights remains the answer to that half.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hermes.cost import ANTHROPIC, OPENAI
from hermes.router.manifest import RIGHTS_APPROVED, RIGHTS_DENIED, RIGHTS_UNKNOWN

# How a teacher's identity is fixed across a dataset's lifetime.
PIN_IMMUTABLE_ID = "immutable_id"  # the provider guarantees the id names fixed weights
PIN_OPEN_WEIGHTS = "open_weights"  # we hold a revision and serve it ourselves
PIN_NONE = "none"  # the id can shift underneath us

PIN_KINDS = (PIN_IMMUTABLE_ID, PIN_OPEN_WEIGHTS, PIN_NONE)

# What an endpoint's logprobs are actually aligned to. A boolean is not enough, which was
# measured rather than reasoned: on one gateway `qwen3.8-max` returns a logprob stream that
# reconstructs the assistant's content exactly, while `kimi-k2.7-code` returns one that
# reconstructs its *reasoning* -- 58 tokens of "The user wants me to count from 1 to 5..."
# for a nine-token answer. Both are well-formed `logprobs.content` arrays and neither errors.
# Mixing them would align one model's answer distribution against another's private thinking,
# and the corpus would train on it without a single failure anywhere.
LOGPROB_NONE = "none"  # the field is absent or empty
LOGPROB_CONTENT = "content"  # the stream reconstructs the assistant message
LOGPROB_FULL_STREAM = "full_stream"  # the stream includes reasoning tokens too

LOGPROB_SCOPES = (LOGPROB_NONE, LOGPROB_CONTENT, LOGPROB_FULL_STREAM)


class TeacherError(ValueError):
    """A teacher declaration is malformed or is being used outside its rights."""


@dataclass(frozen=True)
class Teacher:
    """One pinned teacher, with the constraints that govern its output."""

    teacher_id: str
    model: str
    base_url: str
    usage_shape: str
    pin: str
    training_rights: str
    logprobs: bool = False
    # Set whenever `logprobs` is True. Defaulting it to CONTENT would make the dangerous
    # case -- a stream that silently covers reasoning -- the one you get by saying nothing.
    logprob_scope: str = LOGPROB_NONE
    # Set only for PIN_OPEN_WEIGHTS: the exact revision we serve. A repo name alone is a
    # moving target, which is the thing pinning exists to prevent.
    weights_repo: str = ""
    weights_revision: str = ""
    # Who decided the rights, and on what. "approved" with no provenance is
    # indistinguishable from "nobody looked", and the difference matters when a corpus is
    # audited long after the person who checked has moved on.
    rights_basis: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.teacher_id or not self.model:
            raise TeacherError("a teacher needs an id and a model")
        if self.pin not in PIN_KINDS:
            raise TeacherError(f"{self.teacher_id}: unknown pin kind {self.pin!r}; expected one of {list(PIN_KINDS)}")
        if self.training_rights not in (RIGHTS_APPROVED, RIGHTS_DENIED, RIGHTS_UNKNOWN):
            raise TeacherError(f"{self.teacher_id}: unknown training_rights {self.training_rights!r}")
        if self.usage_shape not in (OPENAI, ANTHROPIC):
            raise TeacherError(f"{self.teacher_id}: unknown usage shape {self.usage_shape!r}")
        if self.training_rights == RIGHTS_APPROVED and not self.rights_basis:
            raise TeacherError(
                f"{self.teacher_id}: approved training rights with no recorded basis; an approval "
                "nobody signed is indistinguishable from an approval nobody made"
            )
        if self.logprob_scope not in LOGPROB_SCOPES:
            raise TeacherError(f"{self.teacher_id}: unknown logprob_scope {self.logprob_scope!r}")
        if self.logprobs and self.logprob_scope == LOGPROB_NONE:
            raise TeacherError(
                f"{self.teacher_id}: declares logprobs with no scope. Two endpoints that both "
                "'support logprobs' can align them to different token streams -- one to the "
                "assistant's answer, one to its reasoning -- and mixing them trains on the "
                "difference without erroring anywhere"
            )
        if not self.logprobs and self.logprob_scope != LOGPROB_NONE:
            raise TeacherError(f"{self.teacher_id}: declares a logprob scope but no logprobs")
        if self.pin == PIN_OPEN_WEIGHTS and not (self.weights_repo and self.weights_revision):
            # A repo without a revision is not a pin. It is a bookmark.
            raise TeacherError(
                f"{self.teacher_id}: open-weights pin needs both a repo and an exact revision; "
                "a repo name alone moves whenever the publisher pushes"
            )

    @property
    def endpoint(self) -> str:
        """Where this teacher is actually served.

        `SPARK_TEACHER_BASE_URL` overrides every declared base_url. The declared value names
        the model's own home; a deployment routes through whatever gateway holds the key,
        and one key rarely works at two hosts. Keeping them apart is not a convenience --
        a registry pointing at the model's home while the key belongs to a gateway produces
        a 401 that retries three times and reads exactly like a busy provider.
        """
        return os.environ.get("SPARK_TEACHER_BASE_URL", "").rstrip("/") or self.base_url

    @property
    def may_train(self) -> bool:
        """Fails closed: unknown rights are not approved rights."""
        return self.training_rights == RIGHTS_APPROVED

    @property
    def reproducible(self) -> bool:
        """Whether a corpus row from this teacher could be regenerated from its identity."""
        return self.pin in (PIN_IMMUTABLE_ID, PIN_OPEN_WEIGHTS)

    @property
    def may_distil_tokens(self) -> bool:
        """Token-level distillation needs logprobs over the answer, and the right to train.

        `LOGPROB_FULL_STREAM` is excluded deliberately. A stream covering reasoning tokens is
        a real distribution over real tokens -- it is simply a distribution over different
        text than the one the student is being taught to emit.
        """
        return self.logprobs and self.logprob_scope == LOGPROB_CONTENT and self.may_train

    def to_record(self) -> dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "usage_shape": self.usage_shape,
            "pin": self.pin,
            "weights_repo": self.weights_repo,
            "weights_revision": self.weights_revision,
            "training_rights": self.training_rights,
            "rights_basis": self.rights_basis,
            "may_train": self.may_train,
            "reproducible": self.reproducible,
            "logprobs": self.logprobs,
            "may_distil_tokens": self.may_distil_tokens,
            "notes": self.notes,
        }


QWEN_38_MAX = Teacher(
    teacher_id="qwen3.8-max",
    model="qwen3.8-max",
    # The model's own home. A deployment routing through a gateway sets
    # SPARK_TEACHER_BASE_URL; see `Teacher.endpoint`.
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    usage_shape=OPENAI,
    pin=PIN_NONE,
    training_rights=RIGHTS_APPROVED,
    rights_basis="maintainer licence review, 2026-08-08",
    logprobs=True,
    logprob_scope=LOGPROB_CONTENT,
    notes=(
        "No dated snapshot is published, so the id can shift underneath a corpus. Logprobs are "
        "documented but restricted to the plus/turbo snapshots and the open models -- not the -max "
        "series. Training from outputs was cleared by the maintainer licence review of "
        "2026-08-08 (see rights_basis); the pin, not the rights, is the open risk here."
    ),
)

CLAUDE_FABLE_5 = Teacher(
    teacher_id="claude-fable-5",
    model="claude-fable-5",
    base_url="https://api.anthropic.com/v1",
    usage_shape=ANTHROPIC,
    pin=PIN_IMMUTABLE_ID,
    training_rights=RIGHTS_DENIED,
    logprobs=False,
    notes=(
        "Anthropic prohibits using outputs as training targets without written permission and names "
        "general-purpose text-generation models specifically, which is what Spark-Hermes is. Router "
        "and capability evidence only -- whether it solved a task is a fact about the task. Its id is "
        "the strongest pin of the three (fixed weights for the id's lifetime), which is worth having "
        "for a reference baseline even though the trajectory cannot be trained on. No logprobs "
        "parameter exists, and the raw chain of thought is never returned."
    ),
)

DEEPSEEK_V4_PRO = Teacher(
    teacher_id="deepseek-v4-pro",
    model="deepseek-v4-pro",
    base_url="https://yunwu.ai/v1",
    usage_shape=OPENAI,
    pin=PIN_NONE,
    training_rights=RIGHTS_APPROVED,
    rights_basis="maintainer licence review, 2026-08-09",
    logprobs=False,
    notes=(
        "Replaces Kimi K3 in the field, which the gateway does not serve at all -- its newest "
        "Kimi is kimi-k2.7-code. Measured over three tasks against qwen3.8-max: won two of them "
        "outright, roughly a fifth of the cost, and self-checked on both wins. The gateway "
        "accepts logprobs and returns none, so this is a trajectory teacher. No dated snapshot "
        "is published, so it cannot be pinned."
    ),
)

KIMI_K3 = Teacher(
    teacher_id="kimi-k3",
    model="kimi-k3",
    base_url="https://api.moonshot.ai/v1",
    usage_shape=OPENAI,
    pin=PIN_NONE,
    # Denied, and this entry is the record of a mistake worth not repeating. It was
    # RIGHTS_APPROVED on "maintainer licence review, 2026-08-08" -- but the licence reviewed
    # was the open-weights licence, which governs the weights and not this endpoint. The
    # hosted service is governed by Moonshot's OpenPlatform terms instead, and those bar
    # developing models with potential competitive possibilities with the service. The two
    # paths serve the same model and are trivially confusable, which is exactly why the
    # rights axis has to name the artefact it reviewed rather than the model.
    training_rights=RIGHTS_DENIED,
    rights_basis=(
        "Moonshot OpenPlatform terms (hosted API), read 2026-08-09: prohibits developing models "
        "with potential competitive possibilities with the service. The open-weights licence does "
        "not govern this endpoint -- see kimi_k3_self_hosted for the path that it does govern."
    ),
    logprobs=False,
    notes=(
        "The hosted id publishes no snapshot and Moonshot dropped the dated convention it used for "
        "K2, so this entry is not reproducible either. Use kimi_k3_self_hosted. Moonshot's terms "
        "also permit them to train on submitted content by default, which is a confidentiality "
        "question as much as a rights one -- task prompts are the asset.\n\n"
        "No logprobs, established rather than assumed: eight request shapes (logprobs alone, with "
        "top_logprobs 1/5/20, top_logprobs alone, echo, explicit stream=False) plus streaming all "
        "returned HTTP 200 with the `logprobs` key ABSENT -- not empty, absent. The same request "
        "shape against qwen3.8-max in the same session returned nine tokens, so the request is "
        "well-formed and this model path drops it."
    ),
)


def kimi_k3_self_hosted(*, revision: str, base_url: str = "http://localhost:8000/v1") -> Teacher:
    """The only configuration that is reproducible, trainable, and can supply logprobs.

    A factory rather than a constant, because it cannot be declared without a revision and
    there is no revision anyone has chosen yet. Shipping it as a constant would mean either
    inventing a SHA -- the exact failure this module exists to prevent -- or shipping an
    invalid object that raises on import. Requiring the caller to supply the pin makes the
    one trainable teacher the one you had to think about.

    Self-hosting also removes the confidentiality problem: Moonshot's terms permit them to
    train on submitted content by default, and task prompts are the asset.
    """
    if not revision:
        raise TeacherError(
            "a self-hosted teacher needs the exact weights revision; without one the corpus records "
            "which repo it came from but not which weights"
        )
    return Teacher(
        teacher_id="kimi-k3-pinned",
        model="kimi-k3",
        base_url=base_url,
        usage_shape=OPENAI,
        pin=PIN_OPEN_WEIGHTS,
        training_rights=RIGHTS_APPROVED,
        rights_basis=(
            "moonshotai/Kimi-K3 LICENSE, full text read 2026-08-09. SELF-HOSTED WEIGHTS ONLY, not "
            "api.moonshot.ai. The grant is to deal in the Software without restriction, expressly "
            "including run, deploy and fine-tune, subject to a closed list of five conditions, none "
            "of which mentions outputs or downstream training; and 'Software' is defined as weights, "
            "config and code, which does not reach outputs, so no right over outputs is reserved. "
            "The licence never uses the word distil -- this rests on the closed condition list and "
            "the absence of a reservation, not on an express permission."
        ),
        logprobs=True,
        logprob_scope=LOGPROB_CONTENT,
        weights_repo="moonshotai/Kimi-K3",
        weights_revision=revision,
        notes=(
            "Pinned open weights served locally. The only entry here that may be trained on, "
            "reproduced from its identity, and used for token-level distillation.\n\n"
            "Two forward conditions, neither binding today, both cheaper to know now:\n"
            "  - The licence's internal-use exemption covers use that does not make the software, "
            "its outputs, or its capabilities available to third parties. Publishing a model or a "
            "corpus forfeits it, so releasing is the moment it stops applying -- not a later scale.\n"
            "  - A revenue threshold over any consecutive twelve months triggers a duty to reach "
            "terms with Moonshot BEFORE commercial use. For a subnet earning emissions that is a "
            "forward tripwire rather than a hypothetical, and the duty lands before the use.\n\n"
            "The licence disclaims non-infringement and carries no indemnity, which is worth "
            "recording separately from the grant: our permission from Moonshot is not a warranty "
            "against anyone else's claim over these weights."
        ),
    )


# The declared field: two teachers, both cleared to train on. Kimi K3 was the second, and
# the reason it is not here changed once it was measured properly rather than looked up.
#
# It was dropped on the belief that the gateway did not serve it -- `/v1/models` lists 401
# models and the newest Kimi among them is `kimi-k2.7-code`. Probed directly on 2026-08-09,
# `kimi-k3` answers correctly, and so does `kimi-k3-thinking`; neither appears in the list,
# while `kimi-k3-0805`, `kimi-k3-preview`, `moonshotai/Kimi-K3` and `kimi-latest` all return
# 503 "no available channel". So the model list is not an inventory on this gateway: absence
# from it proves nothing, and a slug can serve without appearing. Anything concluded here
# from a listing rather than a request should be re-checked with a request.
#
# It stays out of the field for the reason established since: the hosted endpoint is governed
# by terms that deny training, and it returns no logprobs by any of eight request shapes or
# streaming. Fable 5 is excluded on rights. Both remain declared above, because a registry
# that only lists what is currently used loses the record of what was considered and why it
# is not here.
TEACHER_FIELD_V1 = (QWEN_38_MAX, DEEPSEEK_V4_PRO)

REGISTRY: dict[str, Teacher] = {t.teacher_id: t for t in (QWEN_38_MAX, DEEPSEEK_V4_PRO, CLAUDE_FABLE_5, KIMI_K3)}


def get(teacher_id: str) -> Teacher:
    teacher = REGISTRY.get(teacher_id)
    if teacher is None:
        raise TeacherError(f"unknown teacher {teacher_id!r}; declared: {sorted(REGISTRY)}")
    return teacher


def trainable(field_: tuple[Teacher, ...] = TEACHER_FIELD_V1) -> tuple[Teacher, ...]:
    """The teachers whose trajectories may become SFT or DPO rows."""
    return tuple(t for t in field_ if t.may_train)


def token_teachers(field_: tuple[Teacher, ...] = TEACHER_FIELD_V1) -> tuple[Teacher, ...]:
    """The teachers that can supply logprobs for on-policy distillation."""
    return tuple(t for t in field_ if t.may_distil_tokens)


def audit(field_: tuple[Teacher, ...] = TEACHER_FIELD_V1) -> dict[str, Any]:
    """What a corpus built from this field would and would not be.

    Reported before generation rather than discovered after. A field with no trainable
    teacher produces router evidence and no training data, which is a legitimate run and a
    very expensive surprise.
    """
    trainables = trainable(field_)
    return {
        "teachers": [t.teacher_id for t in field_],
        "trainable": [t.teacher_id for t in trainables],
        "rights_denied": [t.teacher_id for t in field_ if t.training_rights == RIGHTS_DENIED],
        "rights_unknown": [t.teacher_id for t in field_ if t.training_rights == RIGHTS_UNKNOWN],
        "not_reproducible": [t.teacher_id for t in field_ if not t.reproducible],
        "token_distillation": [t.teacher_id for t in token_teachers(field_)],
        "yields_training_data": bool(trainables),
        "fully_reproducible": all(t.reproducible for t in field_),
    }


def check_field(field_: tuple[Teacher, ...]) -> list[str]:
    """Problems with a teacher field, worst first. Empty means it is fit to generate with.

    Returns problems rather than raising, because a field can be usable while imperfect --
    generating router evidence from unpinnable teachers is a reasonable thing to do
    knowingly, and an exception would make it impossible rather than deliberate.
    """
    problems: list[str] = []
    if len(field_) < 2:
        problems.append("a tournament needs at least two teachers; one teacher is a generator, not a comparison")
    if not trainable(field_):
        problems.append(
            "no teacher in this field may be trained on; the run produces router and capability "
            "evidence but no SFT or DPO rows"
        )
    unpinned = [t.teacher_id for t in field_ if not t.reproducible]
    if unpinned:
        problems.append(
            f"{unpinned} cannot be pinned, so rows generated from them cannot be regenerated from "
            "their identity; the corpus is reproducible only in the sense that the file still exists"
        )
    missing_revision = [t.teacher_id for t in field_ if t.pin == PIN_OPEN_WEIGHTS and not t.weights_revision]
    if missing_revision:
        problems.append(f"{missing_revision} declare open weights with no revision")
    scopes = {t.logprob_scope for t in field_ if t.logprobs}
    if len(scopes) > 1:
        by_scope = {
            scope: sorted(t.teacher_id for t in field_ if t.logprobs and t.logprob_scope == scope)
            for scope in sorted(scopes)
        }
        problems.append(
            f"this field returns logprobs aligned to different token streams ({by_scope}); a "
            "distillation corpus built from it would put one model's answer distribution beside "
            "another's reasoning, and both are well-formed so nothing would fail"
        )
    unusable = [t.teacher_id for t in field_ if t.logprobs and t.logprob_scope == LOGPROB_FULL_STREAM]
    if unusable:
        problems.append(
            f"{unusable} return logprobs over the full stream including reasoning; the distribution "
            "is real but it is over different text than the student is taught to emit"
        )
    return problems


def rights_conflicts(teachers: Any = None) -> list[str]:
    """Where one model is declared under two sets of rights.

    The same weights served two ways are two artefacts under two agreements: a hosted
    endpoint is governed by its provider's service terms, self-hosted weights by their
    licence, and the two can differ in exactly the direction that matters. Kimi K3 is the
    live example -- its open-weights licence permits fine-tuning and reserves nothing over
    outputs, while the hosted API's terms bar building competing models, and reviewing the
    first while calling the second is how `kimi-k3` came to be declared trainable.

    Reported rather than refused. Two rights for one model is the correct state here; what
    is dangerous is arriving at it by accident, so this makes the divergence something a
    reader has to have seen.
    """
    from collections import defaultdict

    by_model: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for teacher in REGISTRY.values() if teachers is None else teachers:
        by_model[teacher.model][teacher.training_rights].append(teacher.teacher_id)
    notes = []
    for model, rights in sorted(by_model.items()):
        if len(rights) > 1:
            detail = "; ".join(f"{status}: {sorted(ids)}" for status, ids in sorted(rights.items()))
            notes.append(
                f"{model!r} is declared under more than one set of rights ({detail}). Check that each "
                "entry's rights_basis names the artefact it reviewed -- a weights licence does not "
                "govern a hosted endpoint, and the reverse."
            )
    return notes

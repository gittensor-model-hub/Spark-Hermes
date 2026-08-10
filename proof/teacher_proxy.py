"""Signed evidence that a teacher turn came through the validator's route.

A named teacher is not a controlled teacher. Measured on 2026-08-09: a one-character message
with no system message billed 70 prompt tokens on one endpoint and 62 on another, and the
model reproduced instructions we never sent. Two runs can therefore agree on every digest
this repository computes -- harness, suite, selection policy -- while evaluating different
effective agents, because the conditioning arrived from outside everything we digest.

The proxy closes that by making the route itself evidence. Four signed documents:

    TeacherEpochContract   the validator fixes the teacher, route, harness and budgets
    ProxyLease             a miner is scoped to tasks, teachers, a strategy and a budget
    TeacherTurnEnvelope    one model call, hash-chained to the previous one
    TeacherEpisodeReceipt  the sealed set of turns for one attempt

**This module verifies; it does not sign.** Signing belongs to the proxy and the validator,
which hold keys. A gate that could mint the evidence it checks would be checking itself.

## What the chain is for, precisely

Each envelope names its predecessor by digest. That single field is what refuses:

    reordering            turn 3 claiming turn 1's parent
    a dropped middle turn the expensive call a miner would rather not show
    cross-task replay     a good answer spliced from another task
    cross-strategy splice one strategy's turns presented as another's
    a reused episode      the same signed work claimed twice

None of those are caught by verifying signatures alone: every spliced turn is individually
authentic. It is the *sequence* that is forged, so the sequence is what has to be bound.

## What it does not prove, and must therefore say

Proxy evidence establishes that the recorded turns took the declared route. It cannot
establish that they were the *only* model calls the miner made -- a miner may consult
another model privately and use the advice to compose these requests. That is a different
control (egress restriction, or validator-executed strategies) and it gets a different
field. `AssuranceError` is raised on any document claiming more than its mechanism supports,
because a provenance claim that overstates itself is worse than none: it is trusted.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from hermes.harness import digest_mapping

# Reused deliberately, not reimplemented. `digest_mapping` is already the encoder the sealed
# verifier uses -- sorted keys, compact separators, ensure_ascii=True -- and the reason it is
# pinned that way is a real defect: with ensure_ascii=False outside the enclave and True
# inside, any non-ASCII task id digested differently in the two places and an honest batch
# failed its own attested check. A second canonical encoder here would reintroduce it.

SCHEMA_EPOCH = "spark-teacher-epoch-v1"
SCHEMA_LEASE = "spark-proxy-lease-v1"
SCHEMA_TURN = "spark-teacher-turn-v1"
SCHEMA_EPISODE = "spark-teacher-episode-v1"

# What a run's provenance is actually worth, weakest first. Ordered so a comparison means
# something: a row may never claim an assurance above the evidence it carries.
UNPROVEN = "unproven"  # no proxy evidence; the teacher is a claim
PROXY_BOUND = "proxy_bound"  # every declared turn took the validator's route
EXCLUSIVE = "exclusive"  # and no other model was reachable (egress-restricted)

ASSURANCE_ORDER = (UNPROVEN, PROXY_BOUND, EXCLUSIVE)


class ProxyEvidenceError(ValueError):
    """Signed proxy evidence is malformed, unverifiable, or does not chain."""


class AssuranceError(ProxyEvidenceError):
    """A document claims more provenance than its mechanism can support."""


def envelope_digest(document: dict[str, Any]) -> str:
    """The digest a successor commits to.

    Computed over the document WITHOUT its own `envelope_digest` and `signature`, because a
    document cannot contain its own digest and a signature over a field that includes the
    signature is not a thing. Everything else is covered.
    """
    body = {k: v for k, v in document.items() if k not in ("envelope_digest", "signature")}
    return digest_mapping(body)


def verify_signature(document: dict[str, Any], *, public_key_base64: str) -> None:
    """Ed25519 over the canonical body. Raises on anything short of a valid signature."""
    signature = document.get("signature")
    if not isinstance(signature, dict):
        raise ProxyEvidenceError("document carries no signature")
    if signature.get("algorithm") != "Ed25519":
        raise ProxyEvidenceError(f"unexpected signature algorithm {signature.get('algorithm')!r}")
    try:
        raw = base64.b64decode(str(signature.get("value", "")), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ProxyEvidenceError(f"signature is not valid base64: {exc}") from exc
    if len(raw) != 64:
        raise ProxyEvidenceError(f"ed25519 signature is {len(raw)} bytes, expected 64")

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_base64, validate=True))
    # Signed over the canonical digest string rather than a re-serialisation of the body, so
    # signer and verifier cannot disagree about encoding -- the one failure this module's
    # choice of `digest_mapping` exists to prevent.
    try:
        key.verify(raw, envelope_digest(document).encode("utf-8"))
    except InvalidSignature as exc:
        raise ProxyEvidenceError("signature does not verify against the declared key") from exc


def genesis_parent(
    *,
    epoch_contract_digest: str,
    lease_digest: str,
    task_id: str,
    strategy_digest: str,
    attempt_id: str,
    episode_nonce: str,
) -> str:
    """What the first turn of an episode commits to.

    Every field that makes this episode *this* episode. Without a genesis binding, turn 0 of
    one attempt is interchangeable with turn 0 of another -- the chain would refuse a spliced
    middle and accept a spliced beginning.
    """
    return digest_mapping(
        {
            "epoch_contract_digest": epoch_contract_digest,
            "lease_digest": lease_digest,
            "task_id": task_id,
            "strategy_digest": strategy_digest,
            "attempt_id": attempt_id,
            "episode_nonce": episode_nonce,
        }
    )


@dataclass(frozen=True)
class ChainCheck:
    """What a verified episode establishes, and what it still does not."""

    episode_id: str
    task_id: str
    strategy_digest: str
    teacher_id: str
    turns: int
    assurance: str
    conditioning_profile_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "strategy_digest": self.strategy_digest,
            "teacher_id": self.teacher_id,
            "turns": self.turns,
            "generation_assurance": self.assurance,
            "conditioning_profile_digest": self.conditioning_profile_digest,
            # Stated on every verified episode rather than left to a reader's assumption.
            "off_proxy_assistance_excluded": self.assurance == EXCLUSIVE,
            "provider_weights_revision_known": False,
            "generation_replayable": False,
            "semantic_correctness_proven_by_proxy": False,
        }


def verify_chain(
    envelopes: list[dict[str, Any]],
    *,
    public_key_base64: str,
    genesis: str,
    expect_task_id: str,
    expect_strategy_digest: str,
) -> ChainCheck:
    """Verify one episode's turns as a sequence, not as a bag of signatures.

    Every individual turn in a spliced episode is authentic; that is what makes splicing the
    interesting attack. So each check below is about the relation between turns, and the
    per-turn signature is only the precondition.
    """
    if not envelopes:
        raise ProxyEvidenceError("an episode with no turns proves nothing about generation")

    parent = genesis
    seen_nonces: set[str] = set()
    profiles: set[str] = set()
    teachers: set[str] = set()

    for index, document in enumerate(envelopes):
        where = f"turn {index}"
        if document.get("schema_version") != SCHEMA_TURN:
            raise ProxyEvidenceError(f"{where}: not a {SCHEMA_TURN} document")
        verify_signature(document, public_key_base64=public_key_base64)

        episode = document.get("episode") or {}
        identity = document.get("identity") or {}
        teacher = document.get("teacher") or {}

        # Contiguity. A gap is a dropped turn, and the turn a miner drops is the expensive or
        # unsuccessful one -- exactly the evidence that makes a budget claim honest.
        if episode.get("turn_index") != index:
            raise ProxyEvidenceError(
                f"{where}: declares turn_index {episode.get('turn_index')!r}; a gap or reorder means "
                "a turn was dropped or moved, and the dropped one is the one worth hiding"
            )
        if episode.get("parent_envelope_digest") != parent:
            raise ProxyEvidenceError(
                f"{where}: parent digest does not match the preceding turn; this sequence was "
                "assembled rather than produced"
            )
        if episode.get("task_id") != expect_task_id:
            raise ProxyEvidenceError(
                f"{where}: task {episode.get('task_id')!r} is not the task this episode claims "
                f"({expect_task_id!r}); a good answer was spliced in from elsewhere"
            )
        if identity.get("strategy_digest") != expect_strategy_digest:
            raise ProxyEvidenceError(
                f"{where}: strategy digest differs from the episode's; one strategy's turns are "
                "being presented as another's"
            )

        nonce = str(episode.get("request_nonce") or "")
        if not nonce:
            raise ProxyEvidenceError(f"{where}: no request nonce, so a replayed turn is undetectable")
        if nonce in seen_nonces:
            raise ProxyEvidenceError(f"{where}: request nonce reused within the episode")
        seen_nonces.add(nonce)

        # Usage that is unknown must stay unknown. A missing count silently read as zero would
        # make the most expensive run look like the cheapest, which is the direction the
        # budget reward points -- the same failure `PriceBook.price_of` raises on.
        usage = document.get("usage") or {}
        for field in ("input_tokens", "output_tokens"):
            if not isinstance(usage.get(field), int):
                raise ProxyEvidenceError(
                    f"{where}: {field} is {usage.get(field)!r}. Unknown usage must remain unknown; "
                    "treated as zero it makes an expensive run the cheapest one in every comparison"
                )

        profiles.add(str(teacher.get("conditioning_profile_digest") or ""))
        teachers.add(str(teacher.get("teacher_id") or ""))
        parent = envelope_digest(document)

    if len(teachers) != 1:
        raise ProxyEvidenceError(f"episode spans teachers {sorted(teachers)}; one episode is one teacher")
    if len(profiles) != 1:
        # The profile is the measured conditioning of the route. A change mid-episode means
        # the endpoint's hidden prompt moved under the run, and the turns either side of it
        # are not comparable -- which is precisely what the harness digest cannot see.
        raise ProxyEvidenceError(
            f"episode spans conditioning profiles {sorted(profiles)}; the route's hidden conditioning "
            "changed mid-episode, so these turns did not face the same effective agent"
        )

    assurance = str((envelopes[-1].get("assurance") or {}).get("generation_assurance") or UNPROVEN)
    check_assurance(envelopes[-1])
    return ChainCheck(
        episode_id=str((envelopes[0].get("episode") or {}).get("episode_id") or ""),
        task_id=expect_task_id,
        strategy_digest=expect_strategy_digest,
        teacher_id=next(iter(teachers)),
        turns=len(envelopes),
        assurance=assurance,
        conditioning_profile_digest=next(iter(profiles)),
    )


def check_assurance(document: dict[str, Any]) -> None:
    """Refuse a claim the mechanism cannot support.

    A proxy sees the calls that came through it. It cannot see the ones that did not, so
    `exclusive` is not a claim proxy evidence alone can make -- it needs egress restriction or
    validator-executed strategies. Allowing the higher label on proxy-only evidence would
    convert "signed API calls" into "these were the only API calls", which is the specific
    overstatement this whole design exists to avoid.
    """
    assurance = document.get("assurance") or {}
    claimed = str(assurance.get("generation_assurance") or UNPROVEN)
    if claimed not in ASSURANCE_ORDER:
        raise AssuranceError(f"unknown generation_assurance {claimed!r}; expected one of {list(ASSURANCE_ORDER)}")
    if claimed == EXCLUSIVE and not assurance.get("off_proxy_assistance_excluded"):
        raise AssuranceError(
            "claims exclusive assurance without off_proxy_assistance_excluded; a proxy records the "
            "calls that reached it and cannot observe the ones that did not"
        )
    if assurance.get("off_proxy_assistance_excluded") and claimed != EXCLUSIVE:
        raise AssuranceError(
            f"claims off-proxy assistance is excluded at assurance {claimed!r}; that exclusion is what "
            "exclusive means, and asserting it at a lower level makes the level meaningless"
        )
    if assurance.get("provider_weights_revision_known"):
        raise AssuranceError(
            "claims the provider's weights revision is known; no hosted teacher in the registry "
            "publishes a dated snapshot, so a row asserting it is asserting something unverifiable"
        )


def verify_episode(
    receipt: dict[str, Any],
    envelopes: list[dict[str, Any]],
    *,
    public_key_base64: str,
    genesis: str,
) -> ChainCheck:
    """Verify a sealed episode receipt against the turns it claims to cover.

    The receipt is checked against the envelopes rather than trusted: it is signed by the same
    proxy, so on its own it says only that the proxy asserted something. What makes it
    evidence is that its declared turn digests are exactly the turns presented, in order.
    """
    if receipt.get("schema_version") != SCHEMA_EPISODE:
        raise ProxyEvidenceError(f"not a {SCHEMA_EPISODE} document")
    verify_signature(receipt, public_key_base64=public_key_base64)
    if receipt.get("status") != "sealed":
        raise ProxyEvidenceError(f"episode status is {receipt.get('status')!r}; an unsealed episode may still grow")

    check = verify_chain(
        envelopes,
        public_key_base64=public_key_base64,
        genesis=genesis,
        expect_task_id=str(receipt.get("task_id") or ""),
        expect_strategy_digest=str(receipt.get("strategy_digest") or ""),
    )

    declared = list(receipt.get("turn_envelope_digests") or [])
    actual = [envelope_digest(e) for e in envelopes]
    if declared != actual:
        # Order-sensitive on purpose. A set comparison would accept the same turns shuffled,
        # and shuffling is one of the things the chain exists to refuse.
        raise ProxyEvidenceError(
            f"receipt declares {len(declared)} turn digests and the presented sequence is {len(actual)}; "
            "the receipt does not cover exactly these turns in this order"
        )
    if receipt.get("model_call_count") != len(envelopes):
        raise ProxyEvidenceError(
            f"receipt claims {receipt.get('model_call_count')!r} model calls against {len(envelopes)} turns; "
            "an undercount hides a call that was made"
        )
    return check


__all__ = [
    "ASSURANCE_ORDER",
    "EXCLUSIVE",
    "PROXY_BOUND",
    "SCHEMA_EPISODE",
    "SCHEMA_EPOCH",
    "SCHEMA_LEASE",
    "SCHEMA_TURN",
    "UNPROVEN",
    "AssuranceError",
    "ChainCheck",
    "ProxyEvidenceError",
    "check_assurance",
    "envelope_digest",
    "genesis_parent",
    "verify_chain",
    "verify_episode",
    "verify_signature",
]

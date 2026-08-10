"""Signed teacher evidence: every spliced turn is authentic, so the sequence is the test."""

import base64
import copy

import pytest

from proof.teacher_proxy import (
    EXCLUSIVE,
    PROXY_BOUND,
    SCHEMA_EPISODE,
    SCHEMA_TURN,
    UNPROVEN,
    AssuranceError,
    ProxyEvidenceError,
    check_assurance,
    envelope_digest,
    genesis_parent,
    verify_chain,
    verify_episode,
    verify_signature,
)

EPOCH = "sha256:" + "e1" * 32
LEASE = "sha256:" + "1e" * 32
STRATEGY = "sha256:" + "5a" * 32
HARNESS = "sha256:" + "ab" * 32
PROFILE = "sha256:" + "c0" * 32


def _signer():
    """A throwaway proxy key. The module verifies and never signs, so tests supply both."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()

    def sign(document):
        document = copy.deepcopy(document)
        digest = envelope_digest(document)
        document["envelope_digest"] = digest
        document["signature"] = {
            "algorithm": "Ed25519",
            "key_id": "spark-proxy-test",
            "value": base64.b64encode(private.sign(digest.encode("utf-8"))).decode(),
        }
        return document

    return sign, public


def _turn(
    index,
    parent,
    *,
    task_id="task-1842",
    strategy=STRATEGY,
    teacher="qwen3.8-max",
    profile=PROFILE,
    assurance=PROXY_BOUND,
):
    return {
        "schema_version": SCHEMA_TURN,
        "identity": {
            "epoch_contract_digest": EPOCH,
            "lease_digest": LEASE,
            "miner_id": "hotkey-alpha",
            "strategy_digest": strategy,
        },
        "episode": {
            "task_id": task_id,
            "attempt_id": "attempt-01",
            "episode_id": "episode-7",
            "turn_index": index,
            "request_nonce": f"nonce-{task_id}-{index}",
            "parent_envelope_digest": parent,
        },
        "teacher": {
            "teacher_id": teacher,
            "upstream_model": teacher,
            "conditioning_profile_digest": profile,
            "generation_pin_status": "unavailable",
        },
        "harness": {"harness_digest": HARNESS, "protocol_dialect": "hermes4"},
        "request": {"canonical_request_sha256": f"sha256:{index:064d}"},
        "response": {"status": "success", "canonical_response_sha256": f"sha256:{index + 500:064d}"},
        "usage": {"input_tokens": 1000 + index, "output_tokens": 100 + index},
        "assurance": {"generation_assurance": assurance, "off_proxy_assistance_excluded": assurance == EXCLUSIVE},
    }


def _episode(sign, n=4, **turn_kwargs):
    """A well-formed chain of n turns, plus its genesis and sealed receipt."""
    genesis = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id=turn_kwargs.get("task_id", "task-1842"),
        strategy_digest=turn_kwargs.get("strategy", STRATEGY),
        attempt_id="attempt-01",
        episode_nonce="nonce-episode-7",
    )
    envelopes, parent = [], genesis
    for i in range(n):
        signed = sign(_turn(i, parent, **turn_kwargs))
        envelopes.append(signed)
        parent = envelope_digest(signed)
    receipt = sign(
        {
            "schema_version": SCHEMA_EPISODE,
            "episode_id": "episode-7",
            "task_id": turn_kwargs.get("task_id", "task-1842"),
            "strategy_digest": turn_kwargs.get("strategy", STRATEGY),
            "teacher_id": turn_kwargs.get("teacher", "qwen3.8-max"),
            "turn_envelope_digests": [envelope_digest(e) for e in envelopes],
            "model_call_count": n,
            "status": "sealed",
        }
    )
    return genesis, envelopes, receipt


# --- the happy path ---------------------------------------------------------------------------


def test_a_well_formed_episode_verifies():
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    check = verify_episode(receipt, envelopes, public_key_base64=public, genesis=genesis)
    assert check.turns == 4
    assert check.assurance == PROXY_BOUND
    assert check.teacher_id == "qwen3.8-max"


def test_the_verified_record_states_what_is_not_proven():
    """A provenance claim that overstates itself is worse than none, because it is trusted."""
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    record = verify_episode(receipt, envelopes, public_key_base64=public, genesis=genesis).to_record()
    assert record["off_proxy_assistance_excluded"] is False
    assert record["provider_weights_revision_known"] is False
    assert record["generation_replayable"] is False
    assert record["semantic_correctness_proven_by_proxy"] is False


# --- signature attacks -------------------------------------------------------------------------


def test_a_single_bit_change_breaks_the_signature():
    sign, public = _signer()
    _genesis, envelopes, _receipt = _episode(sign, n=1)
    tampered = copy.deepcopy(envelopes[0])
    tampered["usage"]["output_tokens"] += 1
    with pytest.raises(ProxyEvidenceError, match="does not verify"):
        verify_signature(tampered, public_key_base64=public)


def test_another_proxys_key_does_not_verify():
    sign, _public = _signer()
    _sign2, other_public = _signer()
    _genesis, envelopes, _receipt = _episode(sign, n=1)
    with pytest.raises(ProxyEvidenceError, match="does not verify"):
        verify_signature(envelopes[0], public_key_base64=other_public)


def test_an_unsigned_document_is_refused():
    sign, public = _signer()
    _genesis, envelopes, _receipt = _episode(sign, n=1)
    naked = {k: v for k, v in envelopes[0].items() if k != "signature"}
    with pytest.raises(ProxyEvidenceError, match="no signature"):
        verify_signature(naked, public_key_base64=public)


# --- sequence attacks: every turn below is individually authentic ---------------------------------


def test_reordering_two_turns_is_refused():
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    swapped = [envelopes[0], envelopes[2], envelopes[1], envelopes[3]]
    with pytest.raises(ProxyEvidenceError, match="gap or reorder"):
        verify_episode(receipt, swapped, public_key_base64=public, genesis=genesis)


def test_dropping_a_middle_turn_is_refused():
    """The turn a miner drops is the expensive or unsuccessful one, which is exactly the
    evidence that makes a budget claim honest."""
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    without_middle = [envelopes[0], envelopes[1], envelopes[3]]
    with pytest.raises(ProxyEvidenceError, match="gap or reorder"):
        verify_episode(receipt, without_middle, public_key_base64=public, genesis=genesis)


def test_a_turn_spliced_from_another_task_is_refused():
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    _g2, other, _r2 = _episode(sign, task_id="task-9999")
    spliced = [*envelopes[:3], other[3]]
    with pytest.raises(ProxyEvidenceError, match="spliced in from elsewhere|parent digest"):
        verify_episode(receipt, spliced, public_key_base64=public, genesis=genesis)


def test_a_turn_from_another_strategy_is_refused():
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign)
    _g2, other, _r2 = _episode(sign, strategy="sha256:" + "99" * 32)
    spliced = [*envelopes[:3], other[3]]
    with pytest.raises(ProxyEvidenceError, match="strategy digest differs|parent digest"):
        verify_episode(receipt, spliced, public_key_base64=public, genesis=genesis)


def test_a_whole_valid_episode_replayed_onto_another_task_is_refused():
    """Isolates the task binding from the chain check. Every turn here is authentic and the
    chain is internally perfect -- it is simply an episode about a different task, offered as
    the answer to this one. Without the per-turn task_id the sequence alone would accept it."""
    sign, public = _signer()
    genesis, envelopes, _receipt = _episode(sign, task_id="task-9999")
    with pytest.raises(ProxyEvidenceError, match="spliced in from elsewhere"):
        verify_chain(
            envelopes,
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


def test_a_whole_valid_episode_claimed_for_another_strategy_is_refused():
    """Isolates the strategy binding. This is how a miner would sell one strategy's work as a
    new version's: the turns are real, the chain holds, only the label moved."""
    sign, public = _signer()
    other = "sha256:" + "99" * 32
    genesis, envelopes, _receipt = _episode(sign, strategy=other)
    with pytest.raises(ProxyEvidenceError, match="strategy digest differs"):
        verify_chain(
            envelopes,
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


def test_an_episode_replayed_against_a_different_genesis_is_refused():
    """Without a genesis binding, turn 0 of one attempt is interchangeable with turn 0 of
    another -- the chain would refuse a spliced middle and accept a spliced beginning."""
    sign, public = _signer()
    _genesis, envelopes, receipt = _episode(sign)
    elsewhere = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id="task-1842",
        strategy_digest=STRATEGY,
        attempt_id="attempt-02",
        episode_nonce="nonce-episode-8",
    )
    with pytest.raises(ProxyEvidenceError, match="parent digest"):
        verify_episode(receipt, envelopes, public_key_base64=public, genesis=elsewhere)


def test_a_repeated_nonce_within_an_episode_is_refused():
    sign, public = _signer()
    genesis = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id="task-1842",
        strategy_digest=STRATEGY,
        attempt_id="attempt-01",
        episode_nonce="nonce-episode-7",
    )
    first = sign(_turn(0, genesis))
    second_body = _turn(1, envelope_digest(first))
    second_body["episode"]["request_nonce"] = "nonce-task-1842-0"
    envelopes = [first, sign(second_body)]
    with pytest.raises(ProxyEvidenceError, match="nonce reused"):
        verify_chain(
            envelopes,
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


def test_an_empty_episode_proves_nothing():
    sign, public = _signer()
    genesis, _envelopes, _receipt = _episode(sign)
    with pytest.raises(ProxyEvidenceError, match="proves nothing"):
        verify_chain(
            [], public_key_base64=public, genesis=genesis, expect_task_id="task-1842", expect_strategy_digest=STRATEGY
        )


# --- receipt attacks ---------------------------------------------------------------------------


def test_a_receipt_covering_a_longer_run_than_it_presents_is_refused():
    """The receipt is signed by the same proxy, so on its own it says only that the proxy
    asserted something. What makes it evidence is that it covers exactly these turns."""
    sign, public = _signer()
    genesis, envelopes, receipt = _episode(sign, n=4)
    shortened = sign(
        {
            **{k: v for k, v in receipt.items() if k not in ("envelope_digest", "signature")},
            "turn_envelope_digests": [envelope_digest(e) for e in envelopes[:2]],
            "model_call_count": 4,
        }
    )
    with pytest.raises(ProxyEvidenceError, match="does not cover exactly these turns"):
        verify_episode(shortened, envelopes, public_key_base64=public, genesis=genesis)


def test_an_undercounted_call_count_is_refused():
    """An undercount hides a call that was made, which is a budget claim built on a lie."""
    sign, public = _signer()
    genesis, envelopes, _receipt = _episode(sign)
    lying = sign(
        {
            "schema_version": SCHEMA_EPISODE,
            "episode_id": "episode-7",
            "task_id": "task-1842",
            "strategy_digest": STRATEGY,
            "teacher_id": "qwen3.8-max",
            "turn_envelope_digests": [envelope_digest(e) for e in envelopes],
            "model_call_count": 2,
            "status": "sealed",
        }
    )
    with pytest.raises(ProxyEvidenceError, match="model calls against"):
        verify_episode(lying, envelopes, public_key_base64=public, genesis=genesis)


def test_an_unsealed_episode_is_refused():
    sign, public = _signer()
    genesis, envelopes, _receipt = _episode(sign)
    open_receipt = sign(
        {
            "schema_version": SCHEMA_EPISODE,
            "episode_id": "episode-7",
            "task_id": "task-1842",
            "strategy_digest": STRATEGY,
            "teacher_id": "qwen3.8-max",
            "turn_envelope_digests": [envelope_digest(e) for e in envelopes],
            "model_call_count": len(envelopes),
            "status": "open",
        }
    )
    with pytest.raises(ProxyEvidenceError, match="may still grow"):
        verify_episode(open_receipt, envelopes, public_key_base64=public, genesis=genesis)


# --- route and conditioning --------------------------------------------------------------------


def test_an_episode_spanning_two_teachers_is_refused():
    sign, public = _signer()
    genesis = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id="task-1842",
        strategy_digest=STRATEGY,
        attempt_id="attempt-01",
        episode_nonce="nonce-episode-7",
    )
    first = sign(_turn(0, genesis))
    second = sign(_turn(1, envelope_digest(first), teacher="deepseek-v4-pro"))
    with pytest.raises(ProxyEvidenceError, match="one episode is one teacher"):
        verify_chain(
            [first, second],
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


def test_a_conditioning_profile_change_mid_episode_is_refused():
    """The measured hidden conditioning of the route moved under the run. The harness digest
    cannot see it -- that is the whole reason the profile is bound into every turn."""
    sign, public = _signer()
    genesis = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id="task-1842",
        strategy_digest=STRATEGY,
        attempt_id="attempt-01",
        episode_nonce="nonce-episode-7",
    )
    first = sign(_turn(0, genesis))
    second = sign(_turn(1, envelope_digest(first), profile="sha256:" + "ff" * 32))
    with pytest.raises(ProxyEvidenceError, match="conditioning profile"):
        verify_chain(
            [first, second],
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


# --- unknown usage must stay unknown -------------------------------------------------------------


@pytest.mark.parametrize("missing", ["input_tokens", "output_tokens"])
def test_absent_usage_is_refused_rather_than_read_as_zero(missing):
    """Zero would make the most expensive run the cheapest in every comparison, which is the
    direction the budget reward points."""
    sign, public = _signer()
    genesis = genesis_parent(
        epoch_contract_digest=EPOCH,
        lease_digest=LEASE,
        task_id="task-1842",
        strategy_digest=STRATEGY,
        attempt_id="attempt-01",
        episode_nonce="nonce-episode-7",
    )
    body = _turn(0, genesis)
    del body["usage"][missing]
    with pytest.raises(ProxyEvidenceError, match="Unknown usage must remain unknown"):
        verify_chain(
            [sign(body)],
            public_key_base64=public,
            genesis=genesis,
            expect_task_id="task-1842",
            expect_strategy_digest=STRATEGY,
        )


# --- assurance may never exceed the mechanism ------------------------------------------------------


def test_proxy_evidence_may_not_claim_exclusive_assurance():
    """A proxy records the calls that reached it and cannot observe the ones that did not."""
    with pytest.raises(AssuranceError, match="cannot observe the ones that did not"):
        check_assurance({"assurance": {"generation_assurance": EXCLUSIVE, "off_proxy_assistance_excluded": False}})


def test_exclusion_cannot_be_asserted_below_the_exclusive_level():
    with pytest.raises(AssuranceError, match="makes the level meaningless"):
        check_assurance({"assurance": {"generation_assurance": PROXY_BOUND, "off_proxy_assistance_excluded": True}})


def test_no_row_may_claim_a_known_provider_revision():
    """No hosted teacher in the registry publishes a dated snapshot."""
    with pytest.raises(AssuranceError, match="unverifiable"):
        check_assurance({"assurance": {"generation_assurance": PROXY_BOUND, "provider_weights_revision_known": True}})


def test_an_unknown_assurance_level_is_refused():
    with pytest.raises(AssuranceError, match="unknown generation_assurance"):
        check_assurance({"assurance": {"generation_assurance": "totally_fine"}})


def test_the_default_is_unproven():
    check_assurance({"assurance": {"generation_assurance": UNPROVEN}})


# --- the canonical encoder is the one the enclave uses -----------------------------------------------


def test_non_ascii_digests_identically_to_the_sealed_encoder():
    """The defect this reuses `digest_mapping` to avoid: with ensure_ascii=False outside the
    enclave and True inside, a non-ASCII task id digested differently in the two places and
    an honest batch failed its own attested check."""
    from hermes.harness import digest_mapping

    body = {"task_id": "café-über-任务", "n": 1}
    assert envelope_digest(body) == digest_mapping(body)


def test_the_digest_excludes_the_signature_and_its_own_field():
    """A document cannot contain its own digest, and a signature over a field containing the
    signature is not a thing."""
    sign, _public = _signer()
    genesis, envelopes, _receipt = _episode(sign, n=1)
    signed = envelopes[0]
    body = {k: v for k, v in signed.items() if k not in ("envelope_digest", "signature")}
    assert envelope_digest(signed) == envelope_digest(body) == signed["envelope_digest"]


# --- exclusive has to name the control it rests on --------------------------------------------


def _assurance(**kw):
    return {"assurance": kw}


def test_exclusive_asserted_twice_is_refused():
    """The gap this closes. Before, `exclusive` was reachable by asserting it and then
    asserting the flag that means it: the two agreed with each other and nothing asked what
    excluded the off-proxy route. A boolean that concludes the thing it is meant to evidence
    is the shape of claim this module exists to refuse."""
    from proof.teacher_proxy import EXCLUSIVE, AssuranceError, check_assurance

    with pytest.raises(AssuranceError, match="without naming what excluded"):
        check_assurance(_assurance(generation_assurance=EXCLUSIVE, off_proxy_assistance_excluded=True))


def test_exclusive_with_a_named_mechanism_is_accepted():
    from proof.teacher_proxy import EGRESS_RESTRICTION, EXCLUSIVE, check_assurance

    check_assurance(
        _assurance(
            generation_assurance=EXCLUSIVE,
            off_proxy_assistance_excluded=True,
            off_proxy_assistance_excluded_by=EGRESS_RESTRICTION,
        )
    )


def test_an_unrecognised_mechanism_is_refused_rather_than_downgraded():
    """An unreviewable mechanism is not a weaker claim than a reviewable one; a reader cannot
    go and check `off_proxy_assistance_excluded_by: trust_me`."""
    from proof.teacher_proxy import EXCLUSIVE, AssuranceError, check_assurance

    with pytest.raises(AssuranceError, match="unknown exclusion mechanism"):
        check_assurance(
            _assurance(
                generation_assurance=EXCLUSIVE,
                off_proxy_assistance_excluded=True,
                off_proxy_assistance_excluded_by="trust_me",
            )
        )


def test_naming_a_mechanism_below_exclusive_is_refused():
    """Either the run was exclusive and the level understates it, or it was not and the
    mechanism implies it. Both are worth refusing rather than silently accepting."""
    from proof.teacher_proxy import PROXY_BOUND, VALIDATOR_EXECUTED, AssuranceError, check_assurance

    with pytest.raises(AssuranceError, match="names an off-proxy exclusion mechanism"):
        check_assurance(
            _assurance(
                generation_assurance=PROXY_BOUND,
                off_proxy_assistance_excluded_by=VALIDATOR_EXECUTED,
            )
        )


def test_naming_the_mechanism_does_not_claim_to_verify_it():
    """Stated in the source so nobody reads this check as proof of egress restriction. No
    function reading a document can confirm a network was restricted."""
    import inspect

    from proof import teacher_proxy

    source = inspect.getsource(teacher_proxy)
    assert "Naming the mechanism does not verify it" in source

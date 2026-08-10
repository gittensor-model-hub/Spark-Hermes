"""The round lifecycle: publish, take submissions, freeze -- and leak no score before the freeze."""

import json
from dataclasses import fields

import pytest

from hermes.challenge import Attempt, Baseline, open_challenge
from hermes.harness import derive_task_salt, salted_digest
from hermes.round import (
    ACCEPTED,
    FROZEN,
    GRADED,
    LATE,
    LEDGER_FIELDS,
    MALFORMED,
    OPEN,
    PUBLIC_VIEW_FIELDS,
    RECEIPT_FIELDS,
    REFUSED,
    SETTLED,
    VERDICT_WORDS,
    FreezeToken,
    Receipt,
    Registry,
    Round,
    RoundError,
    RoundStateError,
    ScoreLeakError,
    Verdict,
    open_round,
    screen_public_payload,
)

EPOCH = {"model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "harness_digest": "a" * 64}

MASTER = "a-master-salt-long-enough-to-be-accepted"
HIDDEN_BODY = "pytest tests/test_hidden_ordering.py -q"

# Paths the miner contract actually allows. Using real ones keeps the no-leak tests honest:
# a submission that was refused for its paths never reaches the interesting code.
GOOD_PATHS = ["SOUL.md", "skills/planning/SKILL.md"]
DIGEST = "sha256:" + "0" * 64


def _commitment(task_id):
    """The commitment a released task would carry, computed the way the pipeline computes it."""
    return salted_digest(HIDDEN_BODY, derive_task_salt(MASTER, task_id))


def _challenge(task_id="tc-log-rotation-order", *, commit=True, extra=None):
    """A confirmed baseline failure, packaged, with a withheld-check commitment on the task.

    `extra` smuggles fields into `task_pins` so the tests can confirm the round does not
    republish what the challenge packet strips.
    """
    attempts = tuple(
        Attempt(public_passed=False, hidden_passed=None, tokens=10_000, tool_calls=20, wall_time_s=30.0, steps=8)
        for _ in range(10)
    )
    pins = {"task_id": task_id, "prompt": "rotate the logs in order", "verify": "pytest tests/test_order.py -q"}
    if commit:
        pins["hidden_verify_commitment"] = _commitment(task_id)
    pins.update(extra or {})
    return open_challenge(Baseline(task_id=task_id, attempts=attempts), epoch=EPOCH, task_pins=pins)


def _round(**kw):
    kw.setdefault("challenge", _challenge())
    kw.setdefault("round_id", "r-1")
    kw.setdefault("opened_at", 1000.0)
    kw.setdefault("deadline", 2000.0)
    return open_round(**kw)


def _submitted(r, miner="miner-a", at=1100.0, paths=None, digest=DIGEST):
    return r.submit(miner, paths=list(paths or GOOD_PATHS), payload_digest=digest, received_at=at)


def _through_grading(r, *, passed=True, now=2100.0):
    """Drive a round with one submission all the way to GRADED."""
    token = r.freeze(now)
    for miner in r.submissions:
        r.record_verdict(r.token(), miner, passed=passed)
    r.grade(now + 10)
    return token


# --- 1. the state machine, and refusals that name where the round actually is ----------------


def test_the_lifecycle_runs_open_frozen_graded_settled():
    r = _round()
    assert r.state == OPEN
    _submitted(r)
    r.freeze(2000.0)
    assert r.state == FROZEN
    r.record_verdict(r.token(), "miner-a", passed=True)
    r.grade(2010.0)
    assert r.state == GRADED
    r.settle(2020.0)
    assert r.state == SETTLED


def test_an_illegal_transition_names_the_state_the_round_is_in():
    """A refusal that says only 'illegal transition' sends an operator to read the table. The
    state is the one fact they need and the one fact they cannot see from outside."""
    r = _round()
    with pytest.raises(RoundStateError, match="while the round is OPEN"):
        r.grade(1500.0)
    with pytest.raises(RoundStateError, match="while the round is OPEN"):
        r.settle(1500.0)


def test_settling_cannot_skip_grading():
    """SETTLED reveals the per-task salt. Reachable from FROZEN it would open the withheld
    check for a round that recorded no verdict at all."""
    r = _round()
    _submitted(r)
    r.freeze(2000.0)
    with pytest.raises(RoundStateError, match="while the round is FROZEN"):
        r.settle(2010.0)


def test_a_graded_round_cannot_be_reopened_by_freezing_again():
    """Idempotence is for a retry, not for reopening a round whose verdicts are recorded."""
    r = _round()
    _submitted(r)
    _through_grading(r)
    with pytest.raises(RoundStateError, match="while the round is GRADED"):
        r.freeze(2200.0)


def test_a_settled_round_is_final():
    r = _round()
    _submitted(r)
    _through_grading(r)
    r.settle(2200.0)
    with pytest.raises(RoundStateError, match="while the round is SETTLED"):
        r.grade(2300.0)
    with pytest.raises(RoundStateError, match="while the round is SETTLED"):
        r.settle(2300.0)


def test_a_round_whose_deadline_has_already_passed_is_refused_at_publication():
    """Discovered otherwise by the miners who could not submit, and by then the epoch is spent."""
    with pytest.raises(RoundError, match="every submission"):
        _round(deadline=900.0)


# --- 2. the round publishes the public half of the challenge, and reuses the packet ----------


def test_the_round_publishes_the_challenge_packet_verbatim():
    """Reimplementing the public/withheld split would give the repo two places to keep the
    withheld list correct, and only one of them would get updated."""
    c = _challenge()
    r = _round(challenge=c)
    assert r.public_view()["challenge"] == c.to_record()


def test_the_published_round_carries_the_commitment_and_never_the_body():
    r = _round()
    view = r.public_view()
    assert view["challenge"]["withheld"]["hidden_verify_commitment"] == _commitment("tc-log-rotation-order")
    assert view["challenge"]["withheld"]["body_included"] is False
    assert HIDDEN_BODY not in json.dumps(view)


def test_a_withheld_body_smuggled_through_task_pins_is_not_republished():
    """The packet's allowlist drops it and names the drop. The round asserts the outcome rather
    than trusting that the upstream allowlist still covers every withheld field."""
    r = _round(challenge=_challenge(extra={"hidden_verify": HIDDEN_BODY}))
    view = r.public_view()
    assert HIDDEN_BODY not in json.dumps(view)
    assert "hidden_verify" in view["challenge"]["withheld"]["dropped_task_keys"]


def test_a_round_records_whether_overfit_is_measurable_at_all():
    """A task with no withheld check is legitimate, but a round over one cannot produce an
    overfit_rate. Published as a fact so nothing downstream reads one from it."""
    assert _round().public_view()["withheld_check_committed"] is True
    assert _round(challenge=_challenge(commit=False)).public_view()["withheld_check_committed"] is False


# --- 3. replacement while open, and only the last one counts ---------------------------------


def test_a_miner_can_replace_a_submission_while_the_round_is_open():
    r = _round()
    first = _submitted(r, at=1100.0)
    second = _submitted(r, at=1200.0, digest="sha256:" + "1" * 64)
    assert first.revision == 1 and first.replaced_previous is False
    assert second.revision == 2 and second.replaced_previous is True
    assert second.replaced_digest == first.submission_digest


def test_only_the_last_submission_per_miner_stands():
    r = _round()
    _submitted(r, at=1100.0, paths=["SOUL.md"])
    _submitted(r, at=1200.0, paths=["skills/planning/SKILL.md"])
    standing = r.submissions["miner-a"]
    assert list(standing.paths) == ["skills/planning/SKILL.md"]
    assert len(r.submissions) == 1
    assert r.receipts[-1].standing_digest == standing.digest


def test_the_replacement_is_recorded_rather_than_being_invisible():
    r = _round()
    _submitted(r, at=1100.0)
    _submitted(r, at=1200.0, digest="sha256:" + "1" * 64)
    _submitted(r, at=1300.0, digest="sha256:" + "2" * 64)
    assert r.replacements("miner-a") == 2
    assert r.to_record()["replacements"] == 2
    assert r.to_record()["standing_submissions"][0]["revision"] == 3
    # Every attempt is kept: "my upload went through" and "no record of it" must not both hold.
    assert len(r.receipts) == 3


def test_a_malformed_upload_does_not_displace_a_standing_submission():
    """The obvious store-then-validate order loses the round for a miner whose v1 was fine and
    whose v2 archive was truncated: 'the last one counts' would count nothing."""
    r = _round()
    good = _submitted(r, at=1100.0)
    broken = r.submit("miner-a", paths=[], payload_digest="", received_at=1200.0)
    assert broken.outcome == MALFORMED
    assert r.submissions["miner-a"].digest == good.submission_digest
    assert broken.standing_digest == good.submission_digest


def test_a_refused_upload_does_not_displace_a_standing_submission():
    r = _round()
    good = _submitted(r, at=1100.0)
    refused = _submitted(r, at=1200.0, paths=["skills/planning/scripts/do_everything.sh"])
    assert refused.outcome == REFUSED
    assert r.submissions["miner-a"].digest == good.submission_digest


def test_a_time_the_round_cannot_order_is_refused_by_name():
    """Admission, replacement and the freeze all rest on comparing these, so the useful
    complaint is 'the round cannot order this event' rather than a comparison TypeError from
    two layers down."""
    r = _round()
    with pytest.raises(RoundError, match="cannot order"):
        _submitted(r, at="1100")
    with pytest.raises(RoundError, match="cannot order"):
        r.freeze(None)


def test_a_clock_that_runs_backwards_is_refused_rather_than_sorted():
    """Two workers with skewed clocks: the replacement stamped a second earlier arrives second,
    wins the 'last' comparison, and the miner is graded on the version they withdrew."""
    r = _round()
    _submitted(r, at=1200.0)
    with pytest.raises(RoundError, match="earlier than"):
        _submitted(r, at=1100.0)


# --- 4. NO SCORE MAY LEAK BEFORE FREEZE ------------------------------------------------------


def test_a_receipt_has_no_field_a_verdict_could_live_in():
    """The load-bearing structural claim. A Decision from hermes.acceptance cannot be smuggled
    back to a miner through a receipt because there is nowhere in the type to put it."""
    declared = {f.name for f in fields(Receipt)}
    assert declared == set(RECEIPT_FIELDS)
    assert not declared & VERDICT_WORDS


def test_the_receipt_field_allowlist_is_a_tripwire_and_not_decoration(monkeypatch):
    """The realistic failure is a maintainer adding a correctness field to Receipt. The
    allowlist is only worth having if the mismatch is loud, so the check runs at import.

    Driven by narrowing the allowlist rather than by widening the dataclass: the effect on the
    comparison is the same, and mutating the real class would leave the module broken for every
    later test in the file."""
    import hermes.round as mod

    mod._check_receipt_fields()  # the shipped Receipt agrees with the shipped allowlist

    monkeypatch.setattr(mod, "RECEIPT_FIELDS", frozenset(RECEIPT_FIELDS - {"outcome"}))
    with pytest.raises(ScoreLeakError, match=r"added \['outcome'\]"):
        mod._check_receipt_fields()


def test_every_receipt_outcome_reports_the_envelope_and_nothing_about_correctness():
    r = _round()
    accepted = _submitted(r, "miner-a", at=1100.0)
    malformed = r.submit("miner-b", paths="SOUL.md", payload_digest=DIGEST, received_at=1150.0)
    refused = _submitted(r, "miner-c", at=1200.0, paths=["mcp.json"])
    late = _submitted(r, "miner-d", at=2500.0)
    assert [x.outcome for x in (accepted, malformed, refused, late)] == [ACCEPTED, MALFORMED, REFUSED, LATE]
    for receipt in (accepted, malformed, refused, late):
        assert set(receipt.to_record()) <= set(RECEIPT_FIELDS)
        assert not set(receipt.to_record()) & VERDICT_WORDS


def test_a_malformed_upload_is_distinguishable_from_a_refused_one():
    """Without this a miner with a broken tarball spends the round rewriting a strategy that
    was fine -- which is the reason submit returns anything at all."""
    r = _round()
    malformed = r.submit("miner-a", paths=["SOUL.md"], payload_digest="not-a-digest", received_at=1100.0)
    refused = _submitted(r, "miner-b", paths=["run_agent.py"])
    assert malformed.outcome != refused.outcome
    assert "sha256" in malformed.problems[0]
    assert "run_agent.py" in refused.problems[0]


def test_a_verdict_cannot_be_constructed_without_a_freeze_token():
    """Correctness is computable the instant a submission lands -- the validator holds the
    withheld check. The only thing keeping it out of an open round is that the result has
    nowhere to live until freeze() mints the token."""
    with pytest.raises(RoundStateError, match="FreezeToken"):
        Verdict(token=None, miner="miner-a", submission_digest=DIGEST, passed=True)
    with pytest.raises(RoundStateError, match="FreezeToken"):
        Verdict(token="frozen", miner="miner-a", submission_digest=DIGEST, passed=True)


def test_an_open_round_has_no_token_to_mint_a_verdict_with():
    r = _round()
    _submitted(r)
    with pytest.raises(RoundStateError, match="no freeze token"):
        r.token()


def test_recording_a_verdict_is_refused_while_the_round_is_open():
    r = _round()
    _submitted(r)
    r.freeze(2000.0)
    token = r.token()
    fresh = _round(round_id="r-2")
    _submitted(fresh)
    with pytest.raises(RoundStateError, match="while the round is OPEN"):
        fresh.record_verdict(token, "miner-a", passed=True)


def test_a_forged_token_with_the_right_values_is_still_refused():
    """Every field of the token is in the round's public record, so a value comparison would
    let anyone holding that record mint the capability."""
    r = _round()
    _submitted(r)
    real = r.freeze(2000.0)
    forged = FreezeToken(
        round_id=r.round_id,
        frozen_at=real.frozen_at,
        deadline_used=real.deadline_used,
        seal_digest=real.seal_digest,
    )
    assert forged == r.token()  # equal by value
    with pytest.raises(RoundStateError, match="not this round's freeze token"):
        r.record_verdict(forged, "miner-a", passed=True)


def test_reading_verdicts_is_refused_while_the_round_is_open():
    """A caller expecting an empty dict here is a caller about to publish one."""
    r = _round()
    with pytest.raises(RoundStateError, match="not readable while the round is OPEN"):
        _ = r.verdicts


def test_no_payload_of_an_open_round_carries_a_verdict_shaped_field():
    r = _round()
    _submitted(r, "miner-a")
    _submitted(r, "miner-b", at=1150.0)
    for payload in (r.public_view(), r.to_record()):
        assert "verdicts" not in payload
        assert payload["no_score_before_freeze"] is True
        assert set(payload) <= PUBLIC_VIEW_FIELDS | LEDGER_FIELDS | {"challenge"}
    assert r.to_record()["verdicts_withheld_until"] == GRADED


def test_the_validator_ledger_withholds_verdicts_until_the_round_is_graded():
    """Making only the miner-facing payload safe leaves the ledger safe by operator
    discipline, and the ledger is the file that gets copied into a status page."""
    r = _round()
    _submitted(r)
    r.freeze(2000.0)
    r.record_verdict(r.token(), "miner-a", passed=False)
    ledger = r.to_record()
    assert "verdicts" not in ledger
    assert ledger["verdicts_recorded"] == 1
    assert set(ledger) <= PUBLIC_VIEW_FIELDS | LEDGER_FIELDS | {"challenge"}
    # The verdict exists as an object; it is simply not reachable through any payload.
    assert r.verdicts["miner-a"].passed is False


def test_verdicts_appear_only_once_the_round_is_graded():
    r = _round()
    _submitted(r)
    _through_grading(r, passed=False)
    view = r.public_view()
    assert view["verdicts"] == [
        {"miner": "miner-a", "submission_digest": r.submissions["miner-a"].digest, "passed": False, "notes": ""}
    ]


def test_the_screen_refuses_a_verdict_shaped_field_added_to_the_round_metadata():
    """The second net, behind the allowlist: it catches a field somebody added deliberately,
    updated the allowlist for, and named wrongly."""
    with pytest.raises(ScoreLeakError, match="not in the published field set"):
        screen_public_payload({"round_id": "r-1", "hidden_passed": True}, allowed=frozenset({"round_id"}), where="x")
    with pytest.raises(ScoreLeakError, match="verdict-shaped"):
        screen_public_payload(
            {"round_id": "r-1", "hidden_passed": True},
            allowed=frozenset({"round_id", "hidden_passed"}),
            where="x",
        )


def test_the_screen_refuses_the_withheld_body_at_any_depth():
    with pytest.raises(ScoreLeakError, match="withheld half"):
        screen_public_payload(
            {"round_id": "r-1", "frozen": {"debug": [{"hidden_verify": HIDDEN_BODY}]}},
            allowed=frozenset({"round_id", "frozen"}),
            where="x",
        )


# --- 5. the freeze: idempotent, and it records the cutoff it actually used --------------------


def test_freezing_twice_returns_the_same_record_and_moves_nothing():
    """Transports retry. A second freeze that recomputed the cutoff from a later `now` would
    widen the admission window on a retry and admit uploads the first freeze excluded."""
    r = _round()
    _submitted(r)
    first = r.freeze(2000.0)
    second = r.freeze(9999.0)
    assert second is first
    assert second.deadline_used == 2000.0
    assert second.frozen_at == 2000.0
    assert r.state == FROZEN


def test_the_freeze_records_the_cutoff_and_the_moment_separately():
    """The timer fires late in every real deployment. Collapsed into one number, 'was my
    12:00:01 upload in?' is unanswerable from the record."""
    r = _round()
    frozen = r.freeze(2040.0)
    assert frozen.deadline_used == 2000.0
    assert frozen.frozen_at == 2040.0
    assert frozen.overran_s == 40.0
    assert frozen.early is False


def test_a_validator_running_late_does_not_quietly_widen_the_window():
    """Admission is decided by the announced deadline, not by when freeze() got called --
    otherwise which miners got in depends on scheduling jitter nobody can audit."""
    r = _round()
    late = _submitted(r, at=2000.5)
    assert late.outcome == LATE
    assert r.submissions == {}
    frozen = r.freeze(2100.0)
    assert frozen.sealed_submissions == 0


def test_freezing_early_requires_a_reason_and_is_recorded_as_early():
    """A shortened round that looks like a normal one is indistinguishable afterwards from one
    that ran its full length."""
    r = _round()
    with pytest.raises(RoundError, match="before the announced deadline"):
        r.freeze(1500.0)
    frozen = r.freeze(1500.0, reason="upstream model revision withdrawn mid-round")
    assert frozen.early is True
    assert frozen.deadline_used == 1500.0
    assert "withdrawn" in frozen.reason


def test_the_freeze_seals_the_submission_set_by_content():
    """Without a content seal, 'the validator added a favourite's late upload' and 'the
    validator did not' produce identical records."""
    a = _round()
    _submitted(a, "miner-a")
    b = _round(round_id="r-1")
    _submitted(b, "miner-a")
    _submitted(b, "miner-b", at=1150.0)
    assert a.freeze(2000.0).seal_digest != b.freeze(2000.0).seal_digest
    assert a.freeze(2000.0).seal_digest == _round_with_same_submission().freeze(2000.0).seal_digest


def _round_with_same_submission():
    r = _round()
    _submitted(r, "miner-a")
    return r


def test_submissions_are_refused_once_the_round_is_frozen():
    r = _round()
    r.freeze(2000.0)
    with pytest.raises(RoundStateError, match="while the round is FROZEN"):
        _submitted(r, at=2001.0)


# --- grading completeness --------------------------------------------------------------------


def test_grading_refuses_while_a_standing_submission_is_unscored():
    """A rate over the wrong denominator, whose omission is indistinguishable from a miner who
    never submitted."""
    r = _round()
    _submitted(r, "miner-a")
    _submitted(r, "miner-b", at=1150.0)
    r.freeze(2000.0)
    r.record_verdict(r.token(), "miner-a", passed=True)
    with pytest.raises(RoundError, match="miner-b"):
        r.grade(2010.0)


def test_a_verdict_for_a_miner_who_never_submitted_is_refused():
    r = _round()
    _submitted(r, "miner-a")
    r.freeze(2000.0)
    with pytest.raises(RoundError, match="no standing submission"):
        r.record_verdict(r.token(), "miner-ghost", passed=True)


def test_a_second_verdict_for_one_miner_is_refused_rather_than_overwriting():
    """Silently overwriting means the last grader to run wins and two graders disagreeing
    leaves no trace."""
    r = _round()
    _submitted(r, "miner-a")
    r.freeze(2000.0)
    r.record_verdict(r.token(), "miner-a", passed=True)
    with pytest.raises(RoundError, match="already has a verdict"):
        r.record_verdict(r.token(), "miner-a", passed=False)


def test_a_verdict_is_bound_to_the_submission_that_was_standing_at_the_freeze():
    r = _round()
    _submitted(r, at=1100.0)
    standing = _submitted(r, at=1200.0, digest="sha256:" + "1" * 64)
    r.freeze(2000.0)
    verdict = r.record_verdict(r.token(), "miner-a", passed=True)
    assert verdict.submission_digest == standing.submission_digest


# --- 6. settling opens one commitment, and only one -------------------------------------------


def test_the_salt_is_not_revealed_before_the_round_settles():
    r = _round()
    _submitted(r)
    with pytest.raises(RoundStateError, match="while the round is OPEN"):
        r.reveal(MASTER)
    r.freeze(2000.0)
    with pytest.raises(RoundStateError, match="while the round is FROZEN"):
        r.reveal(MASTER)
    r.record_verdict(r.token(), "miner-a", passed=True)
    r.grade(2010.0)
    with pytest.raises(RoundStateError, match="while the round is GRADED"):
        r.reveal(MASTER)


def test_a_settled_round_reveals_a_salt_that_opens_its_own_commitment():
    """The audit itself: a validator that graded against a different check than the one it
    committed to fails here, and that is the only way anyone outside could find out."""
    r = _round()
    _submitted(r)
    _through_grading(r)
    r.settle(2200.0)
    reveal = r.reveal(MASTER)
    assert reveal.opens(HIDDEN_BODY) is True
    assert reveal.opens("pytest tests/test_something_else.py -q") is False


def test_revealing_one_task_leaves_every_other_commitment_sealed():
    """Under one shared salt the first honest audit would have made every unspent withheld
    check brute-forceable. HMAC is a PRF keyed on the master, so salt_i says nothing about
    salt_j."""
    spent = _round(challenge=_challenge("tc-log-rotation-order"))
    other_task = "lh-i18n-catalog-parity"
    _submitted(spent)
    _through_grading(spent)
    spent.settle(2200.0)
    reveal = spent.reveal(MASTER)

    assert reveal.salt == derive_task_salt(MASTER, "tc-log-rotation-order")
    assert reveal.salt != derive_task_salt(MASTER, other_task)
    # The revealed salt cannot open the still-sealed task's commitment.
    assert salted_digest(HIDDEN_BODY, reveal.salt) != _commitment(other_task)
    assert MASTER not in json.dumps(reveal.to_record())


def test_revealing_a_round_whose_task_committed_to_nothing_is_refused():
    """Handing back a salt anyway would let a reader believe an audit happened when no check
    was ever committed to."""
    r = _round(challenge=_challenge(commit=False))
    _submitted(r)
    _through_grading(r)
    r.settle(2200.0)
    with pytest.raises(RoundError, match="nothing to open"):
        r.reveal(MASTER)


def test_the_reveal_is_not_part_of_the_payload_a_miner_polls():
    """A salt in the view fetched every few seconds is one refactor away from being published
    by a round that has not settled."""
    r = _round()
    _submitted(r)
    _through_grading(r)
    r.settle(2200.0)
    view = r.public_view()
    assert view["reveal_available"] is True
    assert r.reveal(MASTER).salt not in json.dumps(view)


# --- the registry -----------------------------------------------------------------------------


def test_the_registry_refuses_a_reused_round_id():
    """Two sets of receipts quoting one round means a miner's proof of submission stops
    identifying anything."""
    reg = Registry()
    reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    with pytest.raises(RoundError, match="already registered"):
        reg.open_round(_challenge("t-b"), round_id="r-1", opened_at=1000.0, deadline=2000.0)


def test_the_registry_refuses_a_second_open_round_over_one_task():
    """Two windows over one task give a miner two 'last' submissions under two deadlines, and
    merging them has no principled answer."""
    reg = Registry()
    reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    with pytest.raises(RoundError, match="already has an open round"):
        reg.open_round(_challenge("t-a"), round_id="r-2", opened_at=1000.0, deadline=2000.0)


def test_a_task_may_open_a_new_round_once_the_previous_one_has_closed():
    reg = Registry()
    first = reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    first.freeze(2000.0)
    second = reg.open_round(_challenge("t-a"), round_id="r-2", opened_at=2100.0, deadline=3000.0)
    assert reg.open_for_task("t-a") is second


def test_an_unknown_round_id_is_refused_by_name():
    with pytest.raises(RoundError, match="no round 'r-9'"):
        Registry().get("r-9")


def test_the_snapshot_round_trips_through_a_json_file(tmp_path):
    reg = Registry()
    r = reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    _submitted(r)
    r.freeze(2000.0)
    path = reg.save(tmp_path / "rounds" / "snapshot.json")
    loaded = Registry.read_snapshot(path)
    assert loaded == reg.snapshot()
    assert loaded["rounds"][0]["round_id"] == "r-1"
    assert loaded["rounds"][0]["frozen"]["deadline_used"] == 2000.0


def test_a_snapshot_of_an_ungraded_round_carries_no_verdict(tmp_path):
    reg = Registry()
    r = reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    _submitted(r)
    r.freeze(2000.0)
    r.record_verdict(r.token(), "miner-a", passed=True)
    text = reg.save(tmp_path / "snapshot.json").read_text(encoding="utf-8")
    assert "verdicts" not in json.loads(text)["rounds"][0]
    assert HIDDEN_BODY not in text


def test_the_snapshot_does_not_hand_back_a_writable_round(tmp_path):
    """A validator that crashed mid-round must not get an OPEN round back from a JSON loader
    and reopen a window that miners watched close."""
    reg = Registry()
    reg.open_round(_challenge("t-a"), round_id="r-1", opened_at=1000.0, deadline=2000.0)
    loaded = Registry.read_snapshot(reg.save(tmp_path / "snapshot.json"))
    assert isinstance(loaded, dict)
    assert not any(isinstance(v, Round) for v in loaded["rounds"])

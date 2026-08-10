"""Accepting a miner's dataset by pull request, or refusing it with a reason."""

import json
from pathlib import Path

import pytest

from eval.rollout_track import (
    ACCEPT,
    REJECT,
    Submission,
    SubmissionError,
    added_lines,
    check_append_only,
    check_binding,
    check_exports,
    check_manifest_work,
    check_novelty,
    check_paths,
    check_scope,
    check_shape,
    gate,
)
from hermes.harness import digest_mapping
from hermes.seed import open_round

# A plausible published attestation. Its contents are never trusted -- the gate reads
# `eat_nonce` and `hwmodel` only from JWKS-verified tokens -- so the fixture only has to
# exist, and `_verified_attestation` below supplies the verification results.
ATTESTATION = {"passed": True, "token": "eyJ.stub.token", "tdx": {"quote_b64": "AAAA", "report_data": "00" * 64}}


@pytest.fixture(autouse=True)
def _verified_attestation(monkeypatch):
    """Stub the NVIDIA/Intel verification so the suite runs without network or hardware.

    This module used to be skipped entirely unless a receipt and key pair captured from a
    live Cathedral account happened to be sitting in a scratch directory, which meant the
    rollout gate's tests never ran in CI. Attestation verification is now against public
    roots with a pure-data interface, so it can be stubbed honestly and the gate's own
    logic is exercised everywhere. Tests that care about a specific failure override one
    of these.
    """
    import eval.verify as verify

    monkeypatch.setattr(
        verify,
        "check_gpu_signature",
        lambda att: {"verified": True, "claims": {"hwmodel": "RTX PRO 6000 Blackwell Server Edition"}},
    )
    monkeypatch.setattr(
        verify,
        "signed_attestation_claims",
        lambda att, gpu_sig=None: {"hwmodel": "RTX PRO 6000 Blackwell Server Edition"},
    )
    monkeypatch.setattr(verify, "check_claim_binding", lambda d, att, gpu_sig=None: True)
    monkeypatch.setattr(verify, "check_tdx_signature", lambda att: {"verified": True})
    monkeypatch.setattr(verify, "check_tdx_binding", lambda d, att: True)
    monkeypatch.setattr(verify, "check_tdx_measurement", lambda att, allowed=(): (None, "unpinned"))


TASKS = [f"task-{i:03d}" for i in range(20)]
MINERS = ["miner-alpha", "miner-bravo"]
ROUND = open_round("round-001", TASKS, MINERS, seed="a" * 64)
MINE = list(ROUND.assigned_to("miner-alpha"))
DIGEST_A = "sha256:" + "a" * 64


def _runs(task_id):
    """Two registered teachers with well-formed trajectory digests: a real tournament."""
    return [
        {
            "teacher_id": t,
            "trajectory_sha256": digest_mapping({"task": task_id, "teacher": t}),
            "passed": True,
            "disqualified": False,
        }
        for t in ("kimi-k3", "qwen3.8-max")
    ]


def _manifest(**overrides):
    body = {
        "schema_version": 1,
        "round_id": "round-001",
        "miner_id": "miner-alpha",
        "harness_digest": DIGEST_A,
        "teachers": ["kimi-k3", "qwen3.8-max"],
        "exports": {},
        "tasks": [{"task_id": t, "comparable": True, "runs": _runs(t), "winner": "kimi-k3"} for t in MINE],
    }
    body.update(overrides)
    return {**body, "manifest_digest": digest_mapping(body)}


def _publish(manifest, tmp_path):
    """Write the export files a submission claims, and return their digests."""
    from hermes.arena import export_digests

    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_id": t["task_id"], "trajectory_sha256": t["runs"][0]["trajectory_sha256"]} for t in manifest["tasks"]
    ]
    (tmp_path / "sft.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "dpo.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "router.jsonl").write_text("", encoding="utf-8")
    return export_digests(tmp_path), {"sft": len(rows), "dpo": 0, "router": 0}


def _record(manifest=None, **overrides):
    manifest = manifest or _manifest()
    record = {
        "schema_version": 1,
        "round_id": "round-001",
        "miner_id": "miner-alpha",
        "hf_url": "https://huggingface.co/datasets/spark/rollouts-round-001-alpha",
        # A URL names a repo; only a revision names its bytes.
        "hf_revision": "4f6c1b9a2d3e5f708192a3b4c5d6e7f809a1b2c3",
        "manifest_digest": manifest["manifest_digest"],
        "harness_digest": DIGEST_A,
        "receipt_id": "7aac98d2-f2d5-4b26-8f0a-b8b007052271",
        "task_ids": sorted(MINE),
        "export_digests": {"sft": "sha256:" + "b" * 64},
        "rows": {},
    }
    record.update(overrides)
    return record


def _gate(record=None, manifest=None, base_text="", export_dir=None, **kwargs):
    manifest = manifest or _manifest()
    record = record if record is not None else _record(manifest)
    return gate(
        base_text=base_text,
        head_text=base_text + json.dumps(record) + "\n",
        round_record=ROUND.to_record(),
        manifest=manifest,
        attestation=ATTESTATION,
        export_dir=export_dir if export_dir is not None else Path("."),
        **kwargs,
    )


def _accepted(tmp_path):
    """A submission that is honest in every respect: real manifest, real published rows."""
    manifest = _manifest()
    digests, rows = _publish(manifest, tmp_path / "exports")
    record = _record(manifest, export_digests=digests, rows=rows)
    return _gate(record=record, manifest=manifest, export_dir=tmp_path / "exports"), record


# --- the happy path -------------------------------------------------------------------


def test_a_well_formed_submission_is_accepted(tmp_path):
    result, _ = _accepted(tmp_path)
    assert result.verdict == ACCEPT, result.issues
    assert result.submission is not None and result.submission.miner_id == "miner-alpha"


def test_the_accepted_record_round_trips(tmp_path):
    submission = _accepted(tmp_path)[0].submission
    assert Submission.from_record(submission.to_record()) == submission


def test_the_result_is_json_safe(tmp_path):
    assert json.loads(json.dumps(_accepted(tmp_path)[0].to_record()))["accepted"] is True


# --- substance: doing nothing must not pay ---------------------------------------------


def test_a_manifest_that_admits_no_work_is_refused():
    """Every task incomparable, no runs, no winner -- perfectly consistent, and empty."""
    empty = _manifest(tasks=[{"task_id": t, "comparable": False, "runs": [], "winner": None} for t in MINE])
    result = _gate(record=_record(empty), manifest=empty)
    assert result.verdict == REJECT
    assert any("nothing in it is a tournament result" in i for i in result.issues)


def test_an_invented_teacher_is_refused():
    bad = _manifest(
        tasks=[
            {
                "task_id": t,
                "comparable": True,
                "runs": [{"teacher_id": "made-up", "trajectory_sha256": DIGEST_A, "passed": True}],
                "winner": "made-up",
            }
            for t in MINE
        ]
    )
    issues = check_manifest_work(bad)
    assert any("not in the registry" in i for i in issues)


def test_a_comparable_task_with_one_teacher_is_refused():
    solo = _manifest(
        tasks=[{"task_id": t, "comparable": True, "runs": _runs(t)[:1], "winner": "kimi-k3"} for t in MINE]
    )
    assert any("two is the minimum" in i for i in check_manifest_work(solo))


def test_a_winner_that_did_not_run_the_task_is_refused():
    odd = _manifest(
        tasks=[{"task_id": t, "comparable": True, "runs": _runs(t), "winner": "claude-fable-5"} for t in MINE]
    )
    assert any("winner that did not run it" in i for i in check_manifest_work(odd))


def test_a_malformed_trajectory_digest_is_refused():
    bad = _manifest(
        tasks=[
            {
                "task_id": t,
                "comparable": True,
                "runs": [{"teacher_id": "kimi-k3", "trajectory_sha256": "x"}, {"teacher_id": "qwen3.8-max"}],
                "winner": "kimi-k3",
            }
            for t in MINE
        ]
    )
    assert any("no usable trajectory digest" in i for i in check_manifest_work(bad))


# --- the published rows must exist ---------------------------------------------------------


def test_unfetched_exports_are_refused_rather_than_skipped():
    """A miner published nothing at all and was paid, before this existed."""
    issues = check_exports(_record(), _manifest(), None)
    assert issues and "nobody looked at" in issues[0]


def test_digests_that_do_not_match_the_published_files_are_refused(tmp_path):
    manifest = _manifest()
    _publish(manifest, tmp_path)
    issues = check_exports(_record(manifest, export_digests={"sft": "sha256:" + "0" * 64}), manifest, tmp_path)
    assert any("do not match the published files" in i for i in issues)


def test_an_inflated_row_count_is_refused(tmp_path):
    manifest = _manifest()
    digests, _ = _publish(manifest, tmp_path)
    issues = check_exports(
        _record(manifest, export_digests=digests, rows={"sft": 100000, "dpo": 0, "router": 0}), manifest, tmp_path
    )
    assert any("rows mismatch" in i for i in issues)


def test_rows_citing_trajectories_outside_the_manifest_are_refused(tmp_path):
    """Rows the sealed check never saw."""
    manifest = _manifest()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sft.jsonl").write_text(json.dumps({"trajectory_sha256": "sha256:" + "9" * 64}) + "\n")
    (tmp_path / "dpo.jsonl").write_text("")
    (tmp_path / "router.jsonl").write_text("")
    from hermes.arena import export_digests as recompute

    issues = check_exports(_record(manifest, export_digests=recompute(tmp_path), rows={}), manifest, tmp_path)
    assert any("absent from the attested manifest" in i for i in issues)


# --- scope: work you were not assigned --------------------------------------------------


def test_work_on_another_miners_tasks_is_rejected():
    """Duplicated effort is what the seeded assignment exists to prevent."""
    theirs = [t for t in TASKS if not ROUND.owns("miner-alpha", t)]
    manifest = _manifest(tasks=[{"task_id": theirs[0], "comparable": True, "runs": [], "winner": "kimi-k3"}])
    result = _gate(record=_record(manifest, task_ids=[theirs[0]]), manifest=manifest)
    assert result.verdict == REJECT
    assert any("were not assigned" in i for i in result.issues)


def test_an_unregistered_miner_is_rejected():
    result = _gate(record=_record(miner_id="miner-nobody"))
    assert any("not registered" in i for i in result.issues)


def test_a_submission_for_a_different_round_is_rejected():
    assert any("names round" in i for i in _gate(record=_record(round_id="round-999")).issues)


def test_scope_is_decided_by_recomputation_not_by_a_validator():
    """A rejection anyone can reproduce is arguable rather than arbitrary."""
    assert check_scope(_record(), ROUND.to_record()) == []


def test_a_round_whose_seed_does_not_match_its_commitment_is_unusable():
    tampered = ROUND.to_record()
    tampered["commitment"] = "sha256:" + "0" * 64
    assert any("unusable" in i for i in check_scope(_record(), tampered))


# --- binding: the receipt must be about these rows ----------------------------------------


def test_a_manifest_that_does_not_match_its_own_body_is_rejected():
    manifest = _manifest()
    manifest["manifest_digest"] = "sha256:" + "f" * 64
    assert any("internally inconsistent" in i for i in check_binding(_record(), manifest))


def test_citing_a_manifest_from_a_different_batch_is_rejected():
    """Otherwise good rows plus another batch's receipt both check out on their own."""
    other = _manifest(round_id="round-002")
    assert any("does not match the published manifest" in i for i in check_binding(_record(), other))


def test_a_harness_mismatch_between_submission_and_manifest_is_rejected():
    assert any(
        "harness_digest does not match" in i
        for i in check_binding(_record(harness_digest="sha256:" + "c" * 64), _manifest())
    )


def test_task_ids_must_match_the_manifest():
    result = _gate(record=_record(task_ids=sorted(MINE)[:1]))
    assert any("do not match the manifest" in i for i in result.issues)


# --- attestation ---------------------------------------------------------------------------


def test_the_binding_no_longer_rests_on_a_self_reported_identifier():
    """`receipt_id` used to be cross-checked against a signed receipt, which was the closest
    thing to binding available: the receipt could not name the manifest it checked, so the
    gate compared an id the submitter wrote to an id the platform wrote and hoped the rest
    followed. Nothing signs that id now, and nothing needs to -- `check_claim_binding`
    compares the export's own digest to the nonce inside an NVIDIA-signed token.

    The field survives as a replay key for `check_novelty`, which is defence in depth
    rather than the thing holding the binding up. So an arbitrary value is no longer an
    attestation failure, and this test exists to say that the weaker check was removed on
    purpose rather than lost."""
    result = _gate(record=_record(receipt_id="00000000-0000-0000-0000-000000000000"))
    assert not any("the receipt is" in i for i in result.issues)
    assert not any("attestation" in i.lower() for i in result.issues)


def test_an_unverifiable_gpu_token_is_rejected(monkeypatch):
    import eval.verify as verify

    monkeypatch.setattr(verify, "check_gpu_signature", lambda att: {"verified": False, "reason": "bad signature"})
    monkeypatch.setattr(verify, "signed_attestation_claims", lambda att, gpu_sig=None: None)
    assert any("does not verify against NVIDIA" in i for i in _gate().issues)


def test_an_attestation_bound_to_other_work_is_rejected(monkeypatch):
    """The failure Cathedral could not detect. A receipt proved *a* sealed run happened;
    nothing recomputable proved which manifest it covered, so one honest green run could be
    cited beside any number of unchecked batches. The signed nonce commits to these bytes."""
    import eval.verify as verify

    monkeypatch.setattr(verify, "check_claim_binding", lambda d, att, gpu_sig=None: False)
    assert any("does not commit to these exports" in i for i in _gate().issues)


def test_a_gpu_attested_run_with_no_tdx_quote_is_rejected(monkeypatch):
    """GPU CC proves the GPU. It says nothing about the VM the harness ran in."""
    import eval.verify as verify

    monkeypatch.setattr(verify, "check_tdx_signature", lambda att: None)
    assert any("the VM it ran in was not" in i for i in _gate().issues)


def test_a_run_on_an_unaccepted_gpu_is_rejected(monkeypatch):
    import eval.verify as verify

    monkeypatch.setattr(verify, "signed_attestation_claims", lambda att, gpu_sig=None: {"hwmodel": "GeForce RTX 4090"})
    assert any("is not a" in i for i in _gate().issues)


def test_a_submission_with_no_attestation_is_rejected():
    result = gate(
        base_text="",
        head_text=json.dumps(_record()) + "\n",
        round_record=ROUND.to_record(),
        manifest=_manifest(),
        attestation=None,
        export_dir=Path("."),
    )
    assert any("no attestation.json" in i for i in result.issues)


# --- novelty ---------------------------------------------------------------------------------


def test_a_second_submission_for_the_same_assignment_is_rejected():
    """The same work claimed twice; cheaper to catch here than in the corpus."""
    first = json.dumps(_record()) + "\n"
    result = _gate(base_text=first, record=_record(hf_url="https://huggingface.co/datasets/spark/again"))
    assert any("has already submitted" in i for i in result.issues)


def test_a_different_miner_on_the_same_round_is_fine():
    """Each miner runs its own attested job, so each cites its own receipt."""
    first = json.dumps(_record()) + "\n"
    other = _record(miner_id="miner-bravo", receipt_id="b79e7afc-9c78-4d35-9100-292734b2babd")
    assert check_novelty(other, first) == []


# --- the registry is history --------------------------------------------------------------------


def test_rewriting_an_earlier_line_is_rejected_even_when_it_improves_it():
    """A reviewer cannot tell an improvement from a rewrite at merge time."""
    base = json.dumps(_record()) + "\n"
    head = json.dumps(_record(rows={"sft": 999})) + "\n"
    assert any("append-only" in i for i in check_append_only(base, head))


def test_dropping_a_line_is_rejected():
    base = json.dumps(_record()) + "\n" + json.dumps(_record(miner_id="miner-bravo")) + "\n"
    assert check_append_only(base, json.dumps(_record()) + "\n")


def test_appending_preserves_history():
    base = json.dumps(_record()) + "\n"
    assert check_append_only(base, base + json.dumps(_record(miner_id="miner-bravo")) + "\n") == []


# --- data only ---------------------------------------------------------------------------------------


def test_a_pr_touching_code_is_rejected():
    """Auto-merge that can carry executable changes is remote code execution with a skippable review."""
    issues = check_paths(["datasets/rollouts.jsonl", "eval/rollout_track.py"])
    assert issues and "unexpected paths" in issues[0]


def test_a_data_only_pr_passes_the_path_check():
    assert check_paths(["datasets/rollouts.jsonl"]) == []


def test_paths_are_not_checked_when_the_caller_cannot_supply_them():
    assert check_paths(None) == []


# --- malformed submissions -------------------------------------------------------------------------------


def test_more_than_one_appended_line_is_rejected():
    """Partially merging a PR is not something the merge button can express."""
    head = json.dumps(_record()) + "\n" + json.dumps(_record(miner_id="miner-bravo")) + "\n"
    result = gate(
        base_text="",
        head_text=head,
        round_record=ROUND.to_record(),
        manifest=_manifest(),
        attestation=ATTESTATION,
        export_dir=Path("."),
    )
    assert any("exactly one appended line" in i for i in result.issues)


def test_a_missing_field_is_named():
    issues = check_shape({k: v for k, v in _record().items() if k != "hf_url"})
    assert any("missing required field: hf_url" in i for i in issues)


def test_a_non_huggingface_url_is_rejected():
    """A bespoke fetch is a bespoke trust decision."""
    assert any("huggingface.co" in i for i in check_shape(_record(hf_url="https://example.com/data")))


def test_a_malformed_digest_is_rejected():
    assert any("sha256: digest" in i for i in check_shape(_record(manifest_digest="deadbeef")))


def test_invalid_json_is_reported_rather_than_crashing_the_gate():
    with pytest.raises(SubmissionError, match="not valid JSON"):
        added_lines("", "{not json}\n")


def test_every_reason_in_a_stage_is_reported_not_just_the_first():
    """A miner who resubmits three times to learn three problems stops submitting."""
    result = _gate(record=_record(hf_url="https://example.com/x", manifest_digest="deadbeef"))
    assert len(result.issues) >= 2


def test_shape_problems_stop_the_semantic_checks_rather_than_guessing_past_them():
    """Scope and binding cannot be checked on a record whose fields are missing; running
    them anyway would report confident nonsense alongside the real problem."""
    result = _gate(record={k: v for k, v in _record().items() if k != "task_ids"})
    assert result.verdict == REJECT
    assert result.issues == ("missing required field: task_ids",)


def test_semantic_problems_are_all_collected_once_the_shape_is_sound():
    other = _manifest(round_id="round-002")
    result = _gate(record=_record(miner_id="miner-nobody"), manifest=other)
    assert len(result.issues) >= 2


def test_a_receipt_may_back_only_one_submission():
    """A receipt proves a sealed check ran; nothing recomputable proves which manifest it
    checked, so one green receipt must not be spread across many batches."""
    first = json.dumps(_record()) + "\n"
    reused = _record(miner_id="miner-bravo", round_id="round-002")
    issues = check_novelty(reused, first)
    assert any("may back one submission" in i for i in issues)


def test_a_fresh_receipt_from_the_same_miner_is_fine():
    first = json.dumps(_record()) + "\n"
    fresh = _record(round_id="round-002", receipt_id="b79e7afc-9c78-4d35-9100-292734b2babd")
    assert check_novelty(fresh, first) == []


def test_both_reuses_are_reported_together():
    first = json.dumps(_record()) + "\n"
    assert len(check_novelty(_record(), first)) == 2


# --- regressions from the second review ------------------------------------------------


def test_a_closed_round_owes_nobody_work():
    from hermes.seed import CLOSED

    closed = {**ROUND.to_record(), "state": CLOSED}
    assert any("closed" in i for i in check_scope(_record(), closed))


def test_a_committed_round_has_no_window_to_work_in():
    from hermes.seed import COMMITTED

    early = {**ROUND.to_record(), "state": COMMITTED}
    assert any("no window" in i for i in check_scope(_record(), early))


def test_a_future_schema_version_is_refused():
    """This gate implements one contract and should say so, not validate by other rules."""
    assert any("does not implement" in i for i in check_shape(_record(schema_version=2)))


def test_duplicate_task_ids_in_a_submission_are_refused():
    dupe = sorted(MINE)[:1] * 2
    assert any("cannot be claimed twice" in i for i in check_shape(_record(task_ids=dupe)))


def test_malformed_export_digest_values_are_refused():
    assert any("64 hex characters" in i for i in check_shape(_record(export_digests={"sft": "nope"})))


def test_a_control_character_in_the_url_is_refused():
    """A newline splits the URL for anything parsing line-wise, and the prefix stays innocent."""
    evil = "https://huggingface.co/datasets/a/b\nhttps://evil.example"
    assert any("control character" in i for i in check_shape(_record(hf_url=evil)))


def test_a_manifest_repeating_a_task_is_refused():
    repeated = _manifest(
        tasks=[{"task_id": MINE[0], "comparable": True, "runs": _runs(MINE[0]), "winner": "kimi-k3"}] * 2
    )
    assert any("repeats task ids" in i for i in check_binding(_record(repeated), repeated))


def test_export_digests_must_match_the_ones_inside_the_attested_manifest():
    """Digests outside the manifest are a claim nobody attested."""
    manifest = _manifest(exports={"sft": "sha256:" + "1" * 64})
    issues = check_binding(_record(manifest, export_digests={"sft": "sha256:" + "2" * 64}), manifest)
    assert any("differ from the ones inside the attested manifest" in i for i in issues)


def test_a_non_integer_row_count_raises_a_typed_error():
    with pytest.raises(SubmissionError, match="must be an integer"):
        Submission.from_record(_record(rows={"sft": "many"}))


def test_a_resubmitted_identical_line_is_not_silently_skipped():
    """Set-membership diffing dropped it, so a PR appending a duplicate looked empty."""
    base = json.dumps(_record()) + "\n"
    assert len(added_lines(base, base + json.dumps(_record()) + "\n")) == 1


# --- a URL names a repo; only a revision names its bytes ------------------------------------


def test_a_submission_without_a_revision_is_refused():
    """Without one the download resolves whatever the branch points at when it runs, so the
    row that was verified is not the row anyone can fetch back."""
    record = _record()
    del record["hf_revision"]
    assert any("hf_revision" in i for i in check_shape(record))


def test_a_branch_name_is_not_a_revision():
    """`main` is a label the publisher can advance after the row is accepted, which is the
    exact property a pin exists to exclude."""
    for movable in ("main", "master", "latest", "HEAD", "dev"):
        issues = check_shape(_record(hf_revision=movable))
        assert any("movable ref" in i for i in issues), movable


def test_an_abbreviated_commit_id_is_refused():
    """Abbreviations resolve by prefix search, and a prefix unique today can collide in a
    repo that keeps growing."""
    issues = check_shape(_record(hf_revision="4f6c1b9"))
    assert any("abbreviated" in i for i in issues)


def test_a_full_commit_sha_is_accepted():
    assert check_shape(_record()) == []


def test_a_tag_or_arbitrary_string_is_refused():
    for value in ("v1.0.0", "refs/heads/main", "not-a-sha", "4F6C1B9A2D3E5F708192A3B4C5D6E7F809A1B2C3"):
        assert check_shape(_record(hf_revision=value)), value

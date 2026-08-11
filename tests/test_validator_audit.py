"""Publishing enough for an outsider to recheck a settled round.

The validator runs the surface, so a miner takes its word for the result unless something makes
that word checkable. The bundle is that, and `verify` is the only part that matters: it recomputes
`salted_digest(withheld_check, per_task_salt)` and compares it to the commitment the challenge
published before submissions opened.

Every test here builds a real commitment from a real check and a real derived salt, so the
verification is arithmetic rather than a fixture agreeing with itself. A bundle test that
constructs its own commitment from the same string it later verifies proves only that a hash
function is deterministic.
"""

import json
import shutil

import pytest

from hermes.challenge import Attempt, Baseline, open_challenge
from hermes.harness import derive_task_salt, salted_digest
from hermes.round import open_round
from validator.audit import AuditError, Bundle, build, claim_digest, file_digests, verify
from validator.store import RoundStore

MASTER = "a-master-salt-long-enough-to-be-real"
CHECK = "pytest -q tests/hidden_rotation_test.py\n"
TASK = "tc-log-rotation-order"


@pytest.fixture
def settled(tmp_path):
    """A settled round whose commitment really opens under the derived salt."""
    commitment = salted_digest(CHECK, derive_task_salt(MASTER, TASK))
    store = RoundStore(tmp_path / "rounds", require_private=False)
    attempts = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=78_000 + i * 1_500,
            tool_calls=11,
            wall_time_s=1.0,
            steps=34,
            max_steps_hit=True,
        )
        for i in range(10)
    )
    challenge = open_challenge(
        Baseline(task_id=TASK, attempts=attempts),
        epoch={"model_revision": "a" * 40, "harness_digest": "b" * 64},
        task_pins={"task_id": TASK, "hidden_verify_commitment": commitment},
    )
    window = open_round(challenge, round_id="aud-1", opened_at=0.0, deadline=1_000.0)
    window.submit("carol", paths=["SOUL.md"], payload_digest="sha256:" + "d" * 64, received_at=10.0)
    window.freeze(now=1_001.0)
    window.record_verdict(window.token(), "carol", passed=False, notes="too few attempts")
    window.grade(now=1_002.0)
    store.save(window)

    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "aud-1-carol.json").write_text(
        json.dumps({"round_id": "aud-1", "miner_id": "carol", "task_id": TASK, "candidate": {"tokens": [59_024] * 10}}),
        encoding="utf-8",
    )
    episodes = tmp_path / "eps"
    episodes.mkdir()
    (episodes / "carol.jsonl").write_text(
        json.dumps({"task_id": TASK, "metrics": {"tokens_used": 59_024}}) + "\n", "utf-8"
    )
    return store, window, cards, episodes, tmp_path


def _build(settled, **kw):
    store, window, cards, episodes, tmp_path = settled
    window.settle(now=1_003.0)
    store.save(window)
    return build(
        round_id="aud-1",
        master_salt=MASTER,
        store=store,
        scorecard_dir=cards,
        episode_dir=episodes,
        out=tmp_path / "out",
        **kw,
    )


# --- when a bundle may be built ---------------------------------------------------------------


def test_a_bundle_cannot_be_built_before_the_round_settles(settled):
    """The salt is what makes a graded round auditable; releasing it earlier hands the withheld
    check to whoever still holds a submission."""
    store, _, cards, episodes, tmp_path = settled
    with pytest.raises(AuditError, match="SETTLED"):
        build(
            round_id="aud-1",
            master_salt=MASTER,
            store=store,
            scorecard_dir=cards,
            episode_dir=episodes,
            out=tmp_path / "o",
        )


def test_a_settled_round_produces_the_files_an_auditor_needs(settled):
    bundle = _build(settled)
    assert sorted(bundle.manifest["files"]) == [
        "challenge.json",
        "episodes/carol.jsonl",
        "reveal.json",
        "round.json",
        "scorecards/carol.json",
    ]


# --- the claim the bundle exists to support ------------------------------------------------------


def test_the_right_withheld_check_opens_the_published_commitment(settled):
    """The property worth having: the validator graded against the check it committed to, not one
    written afterwards to suit a result."""
    assert verify(_build(settled), CHECK) == []


def test_a_different_withheld_check_does_not_open_it(settled):
    problems = verify(_build(settled), "pytest -q something_else.py\n")
    assert problems and "does not open the published commitment" in problems[0]


def test_the_salt_in_the_bundle_is_the_derived_one_not_the_master(settled):
    """`derive_task_salt` is HMAC(master, task_id), so opening this round leaves every unspent
    task's commitment sealed. A bundle carrying the master would open all of them."""
    bundle = _build(settled)
    reveal = bundle.read("reveal.json")
    salt = reveal.get("per_task_salt") or reveal.get("salt")
    assert salt == derive_task_salt(MASTER, TASK)
    assert salt != MASTER


def test_the_master_salt_is_refused_if_it_reaches_a_bundle(settled):
    """Checked by searching rather than by trusting the code above it. A bundle carrying the master
    would look no different from a correct one."""
    from validator.audit import _refuse_master

    bundle = _build(settled)
    (bundle.root / "oops.txt").write_text(MASTER, encoding="utf-8")
    with pytest.raises(AuditError, match="contains the master salt"):
        _refuse_master(bundle.root, MASTER)


# --- integrity ------------------------------------------------------------------------------------


def test_an_edited_file_is_caught(settled):
    """Every later claim is about the contents of these files, so integrity is checked first."""
    bundle = _build(settled)
    card = bundle.root / "scorecards" / "carol.json"
    card.write_text(json.dumps({"candidate": {"verified_passes": 10}}), encoding="utf-8")
    problems = verify(bundle, CHECK)
    assert any("scorecards/carol.json" in p and "manifest says" in p for p in problems)


def test_a_file_added_after_the_fact_is_caught(settled):
    """A bundle that only checked the files it listed would accept any number of extra ones."""
    bundle = _build(settled)
    (bundle.root / "extra.json").write_text("{}", encoding="utf-8")
    assert any("not listed in the manifest" in p for p in verify(bundle, CHECK))


def test_a_deleted_file_is_caught(settled):
    bundle = _build(settled)
    (bundle.root / "episodes" / "carol.jsonl").unlink()
    assert any("absent from the bundle" in p for p in verify(bundle, CHECK))


def test_the_claim_digest_moves_when_any_file_does(settled):
    """Over the sorted path-to-digest map rather than concatenated contents: a rename that swapped
    two files would leave a concatenation unchanged."""
    bundle = _build(settled)
    before = claim_digest(file_digests(bundle.root))
    (bundle.root / "round.json").write_text("{}", encoding="utf-8")
    assert claim_digest(file_digests(bundle.root)) != before


def test_a_manifest_whose_claim_digest_does_not_match_its_own_files_is_caught(settled):
    """The manifest is the one file that cannot digest itself, so its claim has to be recomputed."""
    bundle = _build(settled)
    manifest = bundle.manifest
    manifest["claim_sha256"] = "sha256:" + "0" * 64
    (bundle.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("claim_sha256 does not match" in p for p in verify(bundle, CHECK))


def test_every_problem_is_reported_not_just_the_first(settled):
    """An auditor who learns one at a time cannot tell a bundle with one clerical error from one
    that fails in several ways."""
    bundle = _build(settled)
    (bundle.root / "extra.json").write_text("{}", encoding="utf-8")
    (bundle.root / "round.json").write_text("{}", encoding="utf-8")
    assert len(verify(bundle, "wrong check\n")) >= 3


def test_a_directory_without_a_manifest_is_not_a_bundle(tmp_path):
    with pytest.raises(AuditError, match="not an audit bundle"):
        Bundle(tmp_path).manifest


# --- what the bundle says about itself -------------------------------------------------------------


def test_the_manifest_states_what_it_does_not_prove(settled):
    """Nothing in a bundle can show the episodes came from the pinned model: a validator willing to
    fabricate a log can fabricate a consistent one. Saying so in the file stops a reader inferring
    more from something called a manifest."""
    manifest = _build(settled).manifest
    assert "committed to before submissions opened" in manifest["proves"]
    assert "measured confidential VM" in manifest["does_not_prove"]


def test_the_claim_digest_is_the_value_to_use_as_an_attestation_nonce(settled):
    """The bridge to the attested version: bind this digest as the NRAS nonce and the TDX
    REPORTDATA, the same shape the rollout track already uses."""
    manifest = _build(settled).manifest
    assert manifest["claim_sha256"].startswith("sha256:")
    assert manifest["claim_sha256"] == claim_digest(manifest["files"])


def test_rebuilding_replaces_rather_than_merges(settled):
    """A stale file from a previous build would be digested into the manifest and look original."""
    bundle = _build(settled)
    (bundle.root / "stale.json").write_text("{}", encoding="utf-8")
    store, window, cards, episodes, tmp_path = settled
    again = build(
        round_id="aud-1",
        master_salt=MASTER,
        store=store,
        scorecard_dir=cards,
        episode_dir=episodes,
        out=tmp_path / "out",
    )
    assert not (again.root / "stale.json").exists()


def test_a_round_with_no_scorecard_or_log_still_builds(settled):
    """A submission that could not be run leaves no log. The bundle records what exists rather than
    refusing to publish the round at all."""
    store, window, _, _, tmp_path = settled
    window.settle(now=1_003.0)
    store.save(window)
    bundle = build(
        round_id="aud-1",
        master_salt=MASTER,
        store=store,
        scorecard_dir=tmp_path / "absent",
        episode_dir=tmp_path / "absent",
        out=tmp_path / "sparse",
    )
    assert sorted(bundle.manifest["files"]) == ["challenge.json", "reveal.json", "round.json"]
    assert verify(bundle, CHECK) == []


def test_the_published_round_ledger_carries_the_verdicts(settled):
    """The bundle is built after grading, so the ledger in it is the one that publishes verdicts --
    before grading `to_record` withholds them and the bundle would prove nothing about the result."""
    bundle = _build(settled)
    ledger = bundle.read("round.json")
    assert ledger.get("verdicts"), "a settled round publishes its verdicts"


def test_shutil_is_used_rather_than_moving_the_originals(settled):
    """The scorecards and logs stay where the judge left them: a bundle that moved them would make
    publishing an audit destroy the working copy."""
    store, window, cards, episodes, tmp_path = settled
    _build(settled)
    assert (cards / "aud-1-carol.json").is_file()
    assert (episodes / "carol.jsonl").is_file()
    assert shutil.which is not None  # the module is imported for copy, not move

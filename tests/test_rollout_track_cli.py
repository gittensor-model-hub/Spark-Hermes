"""The rollout gate's production caller: turning a pull request into a decision."""

import json

import pytest

from eval.rollout_track import SubmissionError
from eval.rollout_track_cli import submitted_row


def _row(**overrides):
    row = {"schema_version": 1, "round_id": "r1", "miner_id": "m1"}
    row.update(overrides)
    return row


def test_the_single_added_row_is_the_submission():
    base = ""
    head = json.dumps(_row()) + "\n"
    assert submitted_row(base, head)["miner_id"] == "m1"


def test_a_pr_that_adds_nothing_is_refused():
    text = json.dumps(_row()) + "\n"
    with pytest.raises(SubmissionError, match="adds no row"):
        submitted_row(text, text)


def test_a_pr_adding_two_rows_is_refused():
    """Two rows is either two submissions sharing one review or a mistake, and both are
    better refused than half-processed."""
    head = json.dumps(_row()) + "\n" + json.dumps(_row(miner_id="m2")) + "\n"
    with pytest.raises(SubmissionError, match="adds 2 rows"):
        submitted_row("", head)


def test_a_non_object_row_is_refused():
    """Rejected by `added_lines` with the gate's own error -- which is why the CLI reuses
    that class rather than declaring a second one of the same name."""
    with pytest.raises(SubmissionError, match="is not a JSON object"):
        submitted_row("", '["not", "an", "object"]\n')


def test_an_appended_row_is_found_after_existing_ones():
    """Positional, not set membership: a resubmitted byte-identical line must not vanish."""
    base = json.dumps(_row()) + "\n"
    head = base + json.dumps(_row(miner_id="m2")) + "\n"
    assert submitted_row(base, head)["miner_id"] == "m2"


def test_a_missing_round_announcement_is_a_rejection_not_a_crash(tmp_path, monkeypatch):
    """A round read from the head is a round the submitter could have written, so it is read
    from the base -- and its absence there means the round was never opened."""
    from eval import rollout_track_cli

    monkeypatch.setattr(rollout_track_cli, "git_show", lambda ref, path: "")
    with pytest.raises(SubmissionError, match="never opened"):
        rollout_track_cli.load_round("r-nope", "origin/main")


def test_an_unfetchable_export_yields_no_directory(tmp_path, monkeypatch):
    """None is not a shrug: check_exports treats it as a refusal, which is what stopped a
    submission that published nothing at all from being accepted."""
    from eval import rollout_track_cli

    assert rollout_track_cli.fetch_exports({"hf_url": "https://huggingface.co/datasets/x/y"}, tmp_path) is None


# --- a gate that cannot verify a receipt must reject, not skip -------------------------------


def test_the_attestation_is_read_from_the_exports_it_covers(tmp_path):
    """Safe to read from the miner's own snapshot in a way a receipt was not. A receipt was
    a third party's assertion about a run, so a submitter-supplied copy was a document the
    submitter chose to hand over and it had to come from the issuing API. This is tokens
    signed by NVIDIA and Intel that commit to the bundle they sit beside: editing it breaks
    a signature, substituting another miner's breaks the binding to this claim digest."""
    from eval.rollout_track_cli import load_attestation

    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "attestation.json").write_text('{"passed": true, "token": "abc"}', encoding="utf-8")
    assert load_attestation(exports) == {"passed": True, "token": "abc"}


def test_a_missing_attestation_is_none_rather_than_an_exception(tmp_path):
    """None reaches check_attestation, which rejects with a reason the miner can act on.
    Raising here would lose the other eight checks, and a miner who learns one problem per
    resubmission stops resubmitting."""
    from eval.rollout_track_cli import load_attestation

    exports = tmp_path / "exports"
    exports.mkdir()
    assert load_attestation(exports) is None
    assert load_attestation(None) is None


def test_a_malformed_attestation_is_none_rather_than_a_traceback(tmp_path):
    from eval.rollout_track_cli import load_attestation

    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "attestation.json").write_text("{not json", encoding="utf-8")
    assert load_attestation(exports) is None
    (exports / "attestation.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert load_attestation(exports) is None


def test_the_gate_needs_no_out_of_band_secret():
    """The blocker this replaced. A Cathedral receipt needed its signing keys pinned in this
    repository, obtained out of band, and until they were the gate rejected every submission
    on attestation. NVIDIA's JWKS and Intel's PCS are public and well-known."""
    from pathlib import Path

    import eval.rollout_track_cli as cli

    assert not hasattr(cli, "load_keys_or_explain")
    assert not Path("proof/cathedral_trusted_keys.json").exists()
    workflow = Path(".github/workflows/rollout_track.yml").read_text(encoding="utf-8")
    assert "CATHEDRAL_API_KEY" not in workflow
    assert "--trusted-keys" not in workflow

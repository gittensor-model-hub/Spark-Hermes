"""Taking a private bundle from someone who wants to beat the rules.

This is the only code in the repository that accepts input from an adversary, so the refusals are
the subject and the happy path is one test. Everything is checked before anything is stored: a
validator that wrote first and validated second would need a delete path that itself has to be
right, and the first bug in it is a file on disk nobody meant to accept.

The upload is a JSON map of path to text rather than an archive. A tarball or zip brings path
traversal, symlinks resolving outside the extraction root, and decompression bombs; a map of
strings has none of them. Several tests below exist to show the first class is still refused
anyway, because "the format makes it impossible" is a claim that stops being true the day someone
adds archive support.
"""

import json

import pytest

from validator.intake import (
    DONE,
    EVALUATING,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    PENDING,
    Intake,
    IntakeError,
    Receipt,
    bundle_digest,
    receipt_for_digest,
    submission_id,
    validate,
)

GOOD = {
    "SOUL.md": "# Operating identity\nOne call per turn.\n",
    "skills/protocol-discipline/SKILL.md": "---\nname: p\ndescription: d\n---\n# P\n\n1. Close the tag.\n",
}


@pytest.fixture
def intake(tmp_path):
    return Intake(root=tmp_path / "store", receipts=tmp_path / "receipts.jsonl")


# --- the happy path -------------------------------------------------------------------------------


def test_a_prose_bundle_is_accepted_and_stored(intake, tmp_path):
    receipt = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    assert receipt.status == PENDING
    assert receipt.files == 2 and receipt.bytes == sum(len(v.encode()) for v in GOOD.values())

    stored = intake.bundle_dir(receipt)
    assert (stored / "SOUL.md").read_text(encoding="utf-8") == GOOD["SOUL.md"]
    assert (stored / "skills/protocol-discipline/SKILL.md").is_file()


def test_the_receipt_is_appended_as_public_jsonl(intake):
    intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    line = json.loads(intake.receipts.read_text(encoding="utf-8").splitlines()[0])
    assert line["status"] == PENDING
    assert set(line) == {
        "submission_id",
        "round_id",
        "miner_id",
        "bundle_sha256",
        "received_at",
        "files",
        "bytes",
        "status",
    }, "a receipt carries envelope facts and nothing that could hint at merit"


# --- paths ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["../../etc/passwd", "../SOUL.md", "a/../../b", "/etc/passwd", "C:/x", "skills\\p\\SKILL.md", "SOUL.md\x00"],
)
def test_a_dangerous_path_is_refused(path):
    """Refused by shape, before the contract sees it. The contract matches names against patterns;
    it is not a path-safety check, and relying on it means the day someone adds a permissive
    pattern the traversal arrives with it."""
    assert validate({path: "x"})


def test_a_path_with_surrounding_whitespace_is_refused():
    assert validate({" SOUL.md": "x"})


def test_paths_are_checked_before_size_so_an_enormous_traversal_is_cheap():
    """Order matters: a path that escapes the root must be refused before anything measures it."""
    problems = validate({"../escape.md": "x" * (MAX_TOTAL_BYTES + 1)})
    assert problems and "escapes the bundle root" in problems[0]


# --- size --------------------------------------------------------------------------------------------


def test_too_many_files_is_refused():
    assert validate({f"skills/s{i}/SKILL.md": "x" for i in range(MAX_FILES + 1)})


def test_an_oversized_file_is_named_so_a_miner_knows_which_to_cut():
    problems = validate({**GOOD, "skills/p/references/big.md": "x" * (MAX_FILE_BYTES + 1)})
    assert any("big.md" in p for p in problems)


def test_an_oversized_bundle_is_refused_even_when_each_file_fits():
    files = {f"skills/s{i}/references/n.md": "x" * (MAX_FILE_BYTES - 1) for i in range(8)}
    problems = validate(files)
    assert any("in total" in p for p in problems)


# --- the contract ------------------------------------------------------------------------------------


def test_an_executable_is_refused_with_the_contracts_own_reason():
    problems = validate({**GOOD, "run_agent.py": "import os"})
    assert any("run_agent.py" in p for p in problems)


def test_a_shell_script_under_an_allowed_directory_is_refused():
    assert validate({**GOOD, "skills/protocol-discipline/go.sh": "echo hi"})


def test_the_contract_is_called_not_reimplemented():
    """A second copy of the rules would agree today and diverge on the first change, and the copy
    downstream of the divergence is the one that decides what runs."""
    from hermes.miner_contract import load as load_contract
    from validator.intake import check_contract

    assert check_contract({"run_agent.py": "x"}) == [str(v) for v in load_contract().check(["run_agent.py"])]


# --- shape --------------------------------------------------------------------------------------------


def test_an_empty_bundle_is_refused():
    """An empty surface runs as the unmodified baseline while occupying a submission slot."""
    assert validate({})


def test_a_non_object_bundle_is_refused():
    assert validate(["SOUL.md"])
    assert validate("SOUL.md")


def test_non_string_content_is_refused():
    """JSON permits numbers and nested objects; a file is text."""
    assert validate({"SOUL.md": 123})
    assert validate({"SOUL.md": {"nested": "object"}})


def test_a_round_or_miner_id_cannot_escape_the_store(intake):
    """Both become directory components."""
    for bad in ("../x", "a/b", "", "."):
        with pytest.raises(IntakeError, match="single path segment"):
            intake.accept(round_id=bad, miner_id="carol", files=GOOD)
        with pytest.raises(IntakeError, match="single path segment"):
            intake.accept(round_id="r-1", miner_id=bad, files=GOOD)


def test_nothing_is_written_when_validation_fails(intake, tmp_path):
    """Checked, not assumed. A store-then-validate design needs a delete path that has to be right,
    and the first bug in it is a file nobody meant to accept."""
    with pytest.raises(IntakeError):
        intake.accept(round_id="r-1", miner_id="carol", files={"run_agent.py": "import os"})
    assert not (tmp_path / "store").exists()
    assert not intake.receipts.exists()


# --- the digest, which is what the pull request commits to -------------------------------------------


def test_the_digest_does_not_depend_on_upload_order():
    """The same surface must digest the same however it was serialised, or a miner's public
    commitment would not match their own bundle."""
    assert bundle_digest(GOOD) == bundle_digest(dict(reversed(list(GOOD.items()))))


def test_any_change_moves_the_digest():
    for mutated in (
        {**GOOD, "SOUL.md": GOOD["SOUL.md"] + " "},
        {**GOOD, "skills/p2/SKILL.md": "extra"},
        {"SOUL.md": GOOD["SOUL.md"]},
    ):
        assert bundle_digest(mutated) != bundle_digest(GOOD)


def test_a_re_upload_of_the_same_bundle_is_idempotent(intake):
    """A retry after a dropped connection must not create a second pending submission nobody
    meant."""
    first = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    second = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=101.0)
    assert first.submission_id == second.submission_id
    assert len(intake.read_receipts()) == 1


def test_a_different_miner_gets_a_different_id_for_the_same_bundle():
    """Two miners uploading identical prose are two submissions, not one."""
    digest = bundle_digest(GOOD)
    assert submission_id(round_id="r-1", miner_id="carol", digest=digest) != submission_id(
        round_id="r-1", miner_id="dave", digest=digest
    )


def test_the_pull_requests_digest_selects_which_bundle_is_evaluated(intake):
    """What makes the public commitment authoritative: a miner who uploads twice and commits to the
    first is evaluated on the first, not on whatever arrived most recently."""
    first = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    revised = {**GOOD, "SOUL.md": "# Operating identity\nSomething else.\n"}
    second = intake.accept(round_id="r-1", miner_id="carol", files=revised, now=200.0)
    assert first.bundle_sha256 != second.bundle_sha256

    chosen = receipt_for_digest(intake.read_receipts(), round_id="r-1", digest=first.bundle_sha256)
    assert chosen is not None and chosen.submission_id == first.submission_id


def test_an_unknown_digest_selects_nothing(intake):
    intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    assert receipt_for_digest(intake.read_receipts(), round_id="r-1", digest="sha256:" + "0" * 64) is None


# --- status -------------------------------------------------------------------------------------------


def test_status_moves_through_the_queue(intake):
    receipt = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    assert intake.set_status(receipt.submission_id, EVALUATING).status == EVALUATING
    assert intake.set_status(receipt.submission_id, DONE).status == DONE
    assert [r.status for r in intake.read_receipts()] == [DONE], "one line per submission, rewritten"


def test_an_unknown_status_is_refused(intake):
    receipt = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    with pytest.raises(IntakeError, match="is not one of"):
        intake.set_status(receipt.submission_id, "winner")


def test_an_unknown_submission_cannot_have_a_status_set(intake):
    with pytest.raises(IntakeError, match="no submission"):
        intake.set_status("nope", DONE)


def test_a_malformed_receipt_line_does_not_break_the_file(intake):
    """One bad line must not make the dashboard unreadable or stop a later status being recorded."""
    intake.receipts.parent.mkdir(parents=True, exist_ok=True)
    intake.receipts.write_text("{not json\n", encoding="utf-8")
    receipt = intake.accept(round_id="r-1", miner_id="carol", files=GOOD, now=100.0)
    assert [r.submission_id for r in intake.read_receipts()] == [receipt.submission_id]


def test_a_receipt_round_trips():
    receipt = Receipt(
        submission_id="abc",
        round_id="r-1",
        miner_id="carol",
        bundle_sha256="sha256:" + "1" * 64,
        received_at=100.0,
        files=2,
        bytes=78,
    )
    assert Receipt.from_record(receipt.to_record()) == receipt

"""Publishing the board where the validator cannot be called.

The board reads the validator's own endpoints, which works for whoever runs it and nowhere else. A
static host serves files from a different origin, so the page's relative fetches hit nothing. This
module writes what the page can read instead.

The tests are mostly about what must NOT end up in that file. It is written to a public directory by
a process that also holds the private store, which makes it the one path in the system where a
withheld body could reach the world without passing through the API's screen.
"""

import json
import re

import pytest

from validator.board import SCHEMA, BoardError, snapshot, write
from validator.store import RoundStore


@pytest.fixture
def world(tmp_path):
    """A settled round with one submission, in a store outside the repository."""
    from hermes.challenge import Attempt, Baseline, open_challenge
    from hermes.round import open_round
    from validator.intake import Intake

    attempts = tuple(
        Attempt(
            public_passed=i < 4,
            hidden_passed=True if i < 4 else None,
            tokens=60_000 + i * 500,
            tool_calls=11,
            wall_time_s=100.0,
            steps=34,
        )
        for i in range(10)
    )
    challenge = open_challenge(
        Baseline(task_id="t1", attempts=attempts),
        epoch={"model_revision": "a" * 40, "harness_digest": "b" * 64},
        task_pins={"task_id": "t1", "hidden_verify_commitment": "sha256:" + "c" * 64},
    )
    window = open_round(challenge, round_id="r-1", opened_at=100.0, deadline=1_000.0)
    window.submit("carol", paths=["SOUL.md"], payload_digest="sha256:" + "d" * 64, received_at=110.0)

    store = RoundStore(tmp_path / "rounds", require_private=False)
    store.save(window)

    intake = Intake(root=tmp_path / "store", receipts=tmp_path / "receipts.jsonl")
    intake.accept(round_id="r-1", miner_id="carol", files={"SOUL.md": "# be careful\n"}, now=110.0)
    return store, intake, window


# --- what gets published ---------------------------------------------------------------------------


def test_the_snapshot_carries_the_round_and_its_receipts(world):
    store, intake, _ = world
    payload = snapshot(store=store, intake=intake, now=500.0)
    assert payload["schema_version"] == SCHEMA
    assert payload["generated_at"] == 500.0
    assert payload["round"]["round_id"] == "r-1"
    assert payload["round"]["task_id"] == "t1"
    assert len(payload["submissions"]) == 1
    assert payload["submissions"][0]["miner_id"] == "carol"


def test_the_baseline_travels_with_it(world):
    """The bar is the point. A board showing submissions with nothing to judge them against is a
    list of names, which is what this snapshot exists to stop being true on a static host."""
    store, intake, _ = world
    baseline = snapshot(store=store, intake=intake)["round"]["challenge"]["baseline"]
    assert baseline["attempts"] == 10
    assert baseline["verified_passes"] == 4
    assert baseline["true_pass_rate_interval"][0] < baseline["pass_rate"] < baseline["true_pass_rate_interval"][1]


def test_only_the_named_round_s_receipts_are_included(world):
    """A receipt from another round would appear under this round's submissions, which is a wrong
    denominator dressed as a longer list."""
    store, intake, _ = world
    intake.accept(round_id="r-other", miner_id="dave", files={"SOUL.md": "x\n"}, now=120.0)
    payload = snapshot(store=store, intake=intake)
    assert [s["miner_id"] for s in payload["submissions"]] == ["carol"]


def test_the_newest_round_is_the_one_published(world, tmp_path):
    from hermes.challenge import Attempt, Baseline, open_challenge
    from hermes.round import open_round

    store, intake, _ = world
    attempts = tuple(
        Attempt(public_passed=False, hidden_passed=None, tokens=1, tool_calls=1, wall_time_s=1.0, steps=1)
        for _ in range(10)
    )
    later = open_round(
        open_challenge(
            Baseline(task_id="t2", attempts=attempts),
            epoch={"model_revision": "a" * 40, "harness_digest": "b" * 64},
            task_pins={"task_id": "t2", "hidden_verify_commitment": "sha256:" + "e" * 64},
        ),
        round_id="r-2",
        opened_at=9_000.0,
        deadline=10_000.0,
    )
    store.save(later)
    assert snapshot(store=store, intake=intake)["round"]["round_id"] == "r-2"


# --- what must never get published ------------------------------------------------------------------


def test_no_withheld_body_reaches_the_file(world):
    """The screen the API applies to its read paths, applied here too — and not as belt-and-braces.

    This file is written to a public directory by a process holding the private store, so it is the
    one path where a withheld check could be published without ever passing through the API.
    """
    from hermes.round import WITHHELD_KEYS

    store, intake, _ = world
    text = json.dumps(snapshot(store=store, intake=intake))
    for key in WITHHELD_KEYS:
        assert f'"{key}"' not in text, key


def test_a_withheld_body_in_the_round_is_refused_rather_than_written(world, monkeypatch):
    """A guard nobody has seen fail. If a future `public_view` leaked a body, this must raise rather
    than write the file — the failure has to happen before anything reaches a public directory."""
    from hermes.round import ScoreLeakError

    store, intake, window = world
    leaky = dict(window.public_view())
    leaky["hidden_verify"] = "pytest tests/test_hidden.py -q"
    monkeypatch.setattr(type(window), "public_view", lambda self: leaky)
    with pytest.raises(ScoreLeakError):
        snapshot(store=store, intake=intake)


def test_a_receipt_that_grew_a_field_is_refused_not_filtered(world, monkeypatch):
    """Filtering silently would publish the next unconsidered field by a version of this function
    that forgot to add it to the list. Refusing names it."""
    from validator.intake import Receipt

    store, intake, _ = world
    original = Receipt.to_record
    monkeypatch.setattr(Receipt, "to_record", lambda self: {**original(self), "hidden_passed": True})
    with pytest.raises(BoardError, match="unexpected field"):
        snapshot(store=store, intake=intake)


def test_an_empty_store_is_refused_rather_than_publishing_nothing(tmp_path):
    """An empty snapshot renders as a board with no round, which reads as a competition that is not
    running rather than as a publisher that had nothing to publish."""

    with pytest.raises(BoardError, match="nothing to publish"):
        snapshot(store=RoundStore(tmp_path / "empty", require_private=False))


# --- writing it -----------------------------------------------------------------------------------


def test_the_write_is_atomic(world, tmp_path):
    """A web server reads this file. A reader arriving mid-write would get truncated JSON, and the
    board would report the validator as unreachable — a different and wrong thing to tell them."""
    store, intake, _ = world
    out = tmp_path / "board" / "state.json"
    write(out, snapshot(store=store, intake=intake))
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == SCHEMA
    assert not list(out.parent.glob("*.tmp")), "the temporary file is renamed, not left behind"


def test_the_cli_keeps_the_store_privacy_check_on_by_default():
    """`--allow-public-store` exists and is off unless asked for.

    `store_is_private` returns True outside a git repository -- there is nothing to publish to --
    so the check only bites on a store the working tree tracks, which a test cannot conjure in a
    tmpdir. `tests/test_validator_store.py` covers the check itself; what belongs here is that this
    CLI does not quietly waive it.

    An earlier version of this test asserted `main(...) != 0 or True` and then asserted on a
    docstring. Both pass for any implementation, which makes them worse than no test.
    """
    import inspect

    from validator import board

    source = inspect.getsource(board.main)
    assert "require_private=not args.allow_public_store" in source
    assert '"--allow-public-store"' in source
    assert 'action="store_true"' in source, "a flag that defaults on would waive the check silently"


def test_the_cli_writes_and_reports(world, tmp_path, capsys):
    from validator.board import main

    store, _, _ = world
    out = tmp_path / "docs" / "board" / "state.json"
    assert main(["--store", str(store.root), "--allow-public-store", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and "r-1" in printed
    assert out.is_file()


def test_the_published_page_is_the_same_page_the_validator_serves():
    """One board, two places it can be read from. A hand-kept second copy would agree today and
    diverge on the first change, and the copy downstream of the divergence is the one a reader sees
    -- so publishing copies it rather than trusting anyone to keep them in step."""
    from pathlib import Path

    from validator.board import PAGE

    published = Path("docs/board/index.html")
    assert published.is_file(), "the Pages board has no page to serve"
    assert published.read_text(encoding="utf-8") == PAGE.read_text(encoding="utf-8"), (
        "docs/board/index.html has drifted from validator/dashboard.html; re-run python -m validator.board"
    )


def test_publishing_writes_the_page_beside_the_snapshot(world, tmp_path):
    from validator.board import main

    store, _, _ = world
    out = tmp_path / "pages" / "board" / "state.json"
    assert main(["--store", str(store.root), "--allow-public-store", "--out", str(out)]) == 0
    assert (out.parent / "index.html").is_file()
    page = (out.parent / "index.html").read_text(encoding="utf-8")
    fetched = re.search(r'fetch\("([^"]*state\.json)"', page).group(1)
    # Next to, which means no directory component. The first version of this test asserted
    # "board/state.json" -- a nested path -- while its own docstring said "the path it is published
    # next to", so it pinned the bug instead of catching it: the page is served FROM board/, the
    # relative path resolved to board/board/state.json, and the published board reported "no
    # validator, no snapshot" with the snapshot sitting beside it.
    assert "/" not in fetched, f"{fetched!r} is not a sibling of the page"
    assert (out.parent / fetched).is_file(), f"the page fetches {fetched!r}, which was not published"

"""The validator's HTTP surface, driven against a real round.

Every test here goes through the app rather than calling `hermes.round` directly, because the
property being defended is what a *miner* can read over the wire. A round object that withholds
verdicts correctly is no use if the server serialises them anyway, and that gap is exactly
where the first version of this module was broken: it re-screened `RoundWindow.public_view()` against
`PUBLIC_VIEW_FIELDS`, which describes the metadata block and not the view, so `challenge` read
as an unexpected field and every call would have 500'd.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the validator API is an optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from hermes.challenge import Attempt, Baseline, open_challenge  # noqa: E402
from hermes.round import open_round  # noqa: E402
from validator import api  # noqa: E402

EPOCH = {"model_revision": "a" * 40, "harness_digest": "b" * 64}


def _failing_attempt(**kw):
    base = dict(
        public_passed=False,
        hidden_passed=None,
        tokens=30_000,
        tool_calls=9,
        wall_time_s=40.0,
        steps=12,
    )
    base.update(kw)
    return Attempt(**base)


def _round(round_id="r-1", deadline=1_000.0):
    challenge = open_challenge(
        Baseline(task_id="tc-env-shadowed-config", attempts=tuple(_failing_attempt() for _ in range(10))),
        epoch=EPOCH,
        task_pins={"task_id": "tc-env-shadowed-config", "hidden_verify_commitment": "sha256:" + "c" * 64},
    )
    return open_round(challenge, round_id=round_id, opened_at=0.0, deadline=deadline)


@pytest.fixture
def client():
    api.ROUNDS.clear()
    api.REVEALS.clear()
    yield TestClient(api.app)
    api.ROUNDS.clear()
    api.REVEALS.clear()


# --- the endpoints actually work ---------------------------------------------------------------


def test_health_answers_without_a_round(client):
    assert client.get("/v1/health").status_code == 200


def test_no_round_is_a_404_not_an_empty_success(client):
    """An empty 200 reads as "there is no work" and as "the validator is broken" identically."""
    assert client.get("/v1/round/current").status_code == 404


def test_the_public_view_serialises(client):
    """The case that was broken. `public_view()` carries `challenge`, which is not in
    PUBLIC_VIEW_FIELDS -- that frozenset describes the metadata block."""
    api.ROUNDS["r-1"] = _round()
    response = client.get("/v1/round/current")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["round_id"] == "r-1"
    assert body["task_id"] == "tc-env-shadowed-config"
    assert "challenge" in body


def test_the_challenge_endpoint_serves_the_commitment_and_no_body(client):
    """A substring check does not express this: `hidden_verify_commitment` legitimately contains
    the string "hidden_verify", so asserting its absence fails on the correct payload. The
    property is that no KEY is named `hidden_verify` -- the commitment proves which check will be
    used, and the body is the thing that must not travel."""
    api.ROUNDS["r-1"] = _round()
    body = client.get("/v1/round/r-1/challenge").json()
    assert body["withheld"]["hidden_verify_commitment"].startswith("sha256:")
    assert body["withheld"]["body_included"] is False

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert "hidden_verify" not in set(keys(body))


def test_an_unknown_round_is_a_404(client):
    assert client.get("/v1/round/nope").status_code == 404


# --- no correctness information before the freeze ----------------------------------------------


def test_results_are_refused_while_the_round_is_open(client):
    """The property this module exists for. A verdict served during an open round lets a miner
    resubmit and bisect onto the withheld check one attempt at a time."""
    api.ROUNDS["r-1"] = _round()
    response = client.get("/v1/round/r-1/results")
    assert response.status_code == 409
    assert "verdicts are published after grading" in response.json()["detail"]


def test_the_open_round_view_carries_no_verdicts(client):
    api.ROUNDS["r-1"] = _round()
    body = client.get("/v1/round/r-1").json()
    assert "verdicts" not in body


def test_the_reveal_is_refused_before_the_round_settles(client):
    """Releasing the salt early lets a miner confirm a guess at the withheld check while the
    round is still live. Safe to release afterwards only because salts are per-task."""
    api.ROUNDS["r-1"] = _round()
    response = client.get("/v1/round/r-1/reveal")
    assert response.status_code == 409
    assert "settles" in response.json()["detail"]


def test_receipts_are_served_while_open_because_they_carry_no_correctness(client):
    """A miner must be able to tell a malformed upload from a rejected one, or every debugging
    cycle is a guess. Envelope facts are not scores."""
    api.ROUNDS["r-1"] = _round()
    response = client.get("/v1/round/r-1/receipts")
    assert response.status_code == 200
    assert response.json()["receipts"] == []


# --- the structural guard ----------------------------------------------------------------------


def test_every_route_screens_its_response():
    """Read from source rather than by exercising the endpoints, because the failure mode is an
    endpoint added later by someone who never read the module docstring -- and such a handler
    would pass every behavioural test that only covers the routes somebody remembered."""
    assert api.unscreened_handlers() == []


def test_the_guard_would_catch_an_unscreened_handler():
    """A guard nobody has seen fail is a guard nobody knows works."""

    def leaky_handler() -> dict[str, str]:
        return {"hidden_passed": "yes"}

    original = api.route_handlers
    api.route_handlers = lambda: [*original(), leaky_handler]
    try:
        assert api.unscreened_handlers() == ["leaky_handler"]
    finally:
        api.route_handlers = original


# --- the reveal success path, which the first version of this file never exercised -------------


def test_a_settled_round_serves_the_published_reveal(client):
    """The gap that let a real bug through. The refusal path was tested and the success path was
    not, so `RoundWindow.reveal(master_salt)` being called with no argument went unnoticed until
    pyright flagged it -- a runtime crash on the one endpoint an auditor depends on."""
    r = _round()
    api.ROUNDS["r-1"] = r
    r.freeze(now=1_001.0)
    r.grade(now=1_002.0)
    r.settle(now=1_003.0)
    # Published the way the private side would: by calling RoundWindow.reveal with the master, which
    # is the step this process cannot perform. A hand-built dict was the first version of this
    # test, and the guard rejected it -- because it named the field `salt`, which is in
    # WITHHELD_KEYS. The real record says `per_task_salt`, and that naming is load-bearing
    # rather than stylistic: `salt` would be refused by the same check that protects every
    # other endpoint.
    api.REVEALS["r-1"] = r.reveal("a-test-master-salt-long-enough").to_record()

    response = client.get("/v1/round/r-1/reveal")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["per_task_salt"]
    assert body["opens_only_this_task"] is True


def test_a_settled_round_with_no_published_reveal_is_a_409_not_a_crash(client):
    """The API cannot derive a salt: the master lives where grading happens, not here. So a
    settled round whose reveal has not been posted says so rather than 500-ing or, worse,
    reaching for a secret this process should never hold."""
    r = _round()
    api.ROUNDS["r-1"] = r
    r.freeze(now=1_001.0)
    r.grade(now=1_002.0)
    r.settle(now=1_003.0)
    response = client.get("/v1/round/r-1/reveal")
    assert response.status_code == 409
    assert "no opened commitment has been published" in response.json()["detail"]


def test_the_api_never_holds_the_master_salt():
    """Structural, not aspirational. `RoundWindow.reveal` takes the master and derives the per-task
    salt; calling it here would put the secret that seals every unspent task in the corpus
    inside the process answering untrusted requests."""
    import inspect

    source = inspect.getsource(api)
    assert ".reveal(" not in source
    assert "master_salt" not in source.replace("master salt", "").replace("the master", "")


# --- the upload path, the only write on this server ----------------------------------------------
#
# An earlier version of this module said "this server does not accept submissions". It does now:
# the bundle arrives privately here and the pull request carries only its digest. The read paths are
# unchanged, and the response to an upload has to stay silent about merit -- a miner can call this
# repeatedly and cheaply, so anything it leaked about correctness would be a free oracle on the
# withheld check.

GOOD_BUNDLE = {
    "SOUL.md": "# Operating identity\nOne call per turn.\n",
    "skills/p/SKILL.md": "---\nname: p\ndescription: d\n---\n# P\n\n1. Close the tag.\n",
}


@pytest.fixture
def intake_at(tmp_path, monkeypatch):
    """Point the intake at a temporary store so uploads do not touch the repository.

    This fixture said that and did not do it. The upload endpoint builds its own `Intake()`, whose
    path defaults were plain dataclass defaults -- captured into the generated `__init__` when the
    class was created, so reassigning the module globals afterwards changed nothing. Two more lines
    patched `Intake.root` and `Intake.receipts` as class attributes, which a dataclass instance
    never consults either.

    So every API test wrote real bundles into `var/submissions` and appended to the real
    `datasets/receipts.jsonl`, and all of them passed: they assert on responses, and nothing
    asserted on where the files landed. `Intake` resolves its defaults through `default_factory`
    now, which makes these two patches the whole mechanism -- and the class-attribute patches an
    AttributeError rather than a no-op, which is how this was finally noticed.
    """
    from validator import intake as intake_module

    monkeypatch.setattr(intake_module, "SUBMISSION_DIR", tmp_path / "store")
    monkeypatch.setattr(intake_module, "RECEIPTS", tmp_path / "receipts.jsonl")
    return tmp_path


def test_an_upload_does_not_touch_the_real_store(client, intake_at, tmp_path):
    """The assertion the fixture's docstring implied and nothing made.

    Checked by what is on disk afterwards rather than by trusting the patch: the previous version of
    this isolation was a no-op, every test still passed, and the evidence was a `carol` submission
    sitting in the operator's `var/submissions` from a test run months of commits later.
    """
    # The repository's real paths, written out rather than imported. Importing them here would read
    # the values the fixture just patched, so the test would compare the temporary store against
    # itself and pass no matter what -- which is the same shape of mistake as the fixture's own.
    DEFAULT_STORE = Path("var/submissions")
    DEFAULT_RECEIPTS = Path("datasets/receipts.jsonl")

    from validator import intake as intake_module

    assert intake_module.SUBMISSION_DIR != DEFAULT_STORE, "the fixture is not patching anything"

    real_before = DEFAULT_RECEIPTS.read_bytes() if DEFAULT_RECEIPTS.is_file() else None
    api.ROUNDS["r-iso"] = _round(deadline=1e12)
    assert (
        client.post("/v1/round/r-iso/submission", json={"miner_id": "carol", "files": GOOD_BUNDLE}).status_code == 200
    )

    # It landed in the temporary store...
    assert (tmp_path / "store" / "r-iso" / "carol").is_dir()
    assert (tmp_path / "receipts.jsonl").is_file()
    # ...and nowhere near the real one.
    assert not (DEFAULT_STORE / "r-iso").exists()
    real_after = DEFAULT_RECEIPTS.read_bytes() if DEFAULT_RECEIPTS.is_file() else None
    assert real_after == real_before, "the suite appended to the operator's receipts file"


def test_an_upload_returns_a_receipt_and_nothing_about_merit(client, intake_at):
    api.ROUNDS["r-1"] = _round(deadline=1e12)
    response = client.post("/v1/round/r-1/submission", json={"miner_id": "carol", "files": GOOD_BUNDLE})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["bundle_sha256"].startswith("sha256:")
    assert not {"passed", "verdict", "hidden_passed", "score"} & set(body)


def test_the_dashboard_lists_receipts_for_a_round(client, intake_at):
    api.ROUNDS["r-1"] = _round(deadline=1e12)
    client.post("/v1/round/r-1/submission", json={"miner_id": "carol", "files": GOOD_BUNDLE})
    listed = client.get("/v1/submissions", params={"round_id": "r-1"}).json()["submissions"]
    assert [r["miner_id"] for r in listed] == ["carol"]
    assert listed[0]["status"] == "pending"


def test_a_bundle_the_contract_refuses_is_a_400_with_the_reason(client, intake_at):
    """A miner who learns one problem per upload stops uploading, so every reason is returned."""
    api.ROUNDS["r-1"] = _round(deadline=1e12)
    response = client.post(
        "/v1/round/r-1/submission", json={"miner_id": "carol", "files": {"run_agent.py": "import os"}}
    )
    assert response.status_code == 400
    assert "run_agent.py" in response.json()["detail"]


def test_a_traversing_path_is_refused(client, intake_at):
    api.ROUNDS["r-1"] = _round(deadline=1e12)
    response = client.post("/v1/round/r-1/submission", json={"miner_id": "carol", "files": {"../../etc/x": "y"}})
    assert response.status_code == 400
    assert "escapes the bundle root" in response.json()["detail"]


def test_an_upload_after_the_freeze_is_refused(client, intake_at):
    """A bundle taken after the freeze would sit in the store looking like a submission with no
    window left to judge it."""
    window = _round(deadline=1_000.0)
    api.ROUNDS["r-1"] = window
    window.freeze(now=1_001.0)
    response = client.post("/v1/round/r-1/submission", json={"miner_id": "carol", "files": GOOD_BUNDLE})
    assert response.status_code == 409
    assert "uploads are accepted while it is open" in response.json()["detail"]


def test_an_upload_to_an_unknown_round_is_a_404(client, intake_at):
    assert client.post("/v1/round/nope/submission", json={"miner_id": "x", "files": GOOD_BUNDLE}).status_code == 404


def test_the_upload_handler_screens_its_response_like_every_other(client, intake_at):
    """The structural guard covers writes too, or the one endpoint that takes untrusted input would
    be the one exempt from the rule the module exists for."""
    assert api.unscreened_handlers() == []
    assert api.submit in api.route_handlers()
    assert api.submissions in api.route_handlers()


# --- the submissions board -----------------------------------------------------------------------
#
# The dashboard the design asks for: unique id, submitter, and where the upload is in the queue.
# Served from the validator's own origin, so the page reads `/v1/submissions` on this host and
# there is no cross-origin request to allow and no second place to configure a URL.


def test_the_board_is_served_from_the_validator_itself(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Submissions" in response.text


def test_the_board_loads_no_code_or_assets_from_another_origin(client):
    """A validator serves no third-party script, and neither does its board. Data may come from a
    configured validator; a stylesheet, font or script may not come from anywhere."""
    page = client.get("/").text
    for attribute in ("src=", "href=", "@import"):
        for absolute in ("http://", "https://", "//cdn", "//unpkg"):
            assert f'{attribute}"{absolute}' not in page and f"{attribute}'{absolute}" not in page


def test_the_validator_origin_never_comes_from_the_url(client):
    """The board may be told which validator to read, and only by a file committed beside it.

    A `?validator=` parameter would let anyone render another validator's numbers under this
    project's name — which is exactly what serving the board from the validator's own origin
    avoided, and the reason the configurable origin is a committed file rather than a query string.
    """
    page = client.get("/").text
    for taken_from_the_url in ("location.search", "URLSearchParams", "location.hash", "document.referrer"):
        assert taken_from_the_url not in page, taken_from_the_url
    assert 'fetch("config.json"' in page, "the origin is read from the committed config"


def test_relative_paths_are_used_when_no_validator_is_configured(client):
    """The validator serving its own board is the case with no config file at all, and it must keep
    working: `endpoint()` returns the path unchanged when the base is empty."""
    page = client.get("/").text
    assert 'endpoint("/v1/round/current")' in page
    assert 'endpoint("/v1/submissions")' in page
    assert "return base ? `${base}${path}` : path;" in page


def _executable(page: str) -> str:
    """The page with its comments and stylesheet removed.

    Screened on the executable part rather than on the file, and the distinction is the same one
    `tests/test_base_model.py` learned: a check that forbids *naming* a thing forbids explaining
    why it is absent, and the first version of this test failed on the comment that says there is
    no score column. What could leak here is a receipt field being read and rendered; a static
    comment reads no field and renders nothing, so it cannot carry live data however it is worded.
    The stylesheet goes for the same reason and cost a second failure: `font-weight` is not a
    verdict, and CSS cannot read a receipt.
    """
    import re

    stripped = re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)
    stripped = re.sub(r"<style>.*?</style>", "", stripped, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", stripped, flags=re.MULTILINE)


# Field names the API publishes on purpose that happen to contain screened vocabulary. Enumerated
# rather than pattern-matched, each with the reason it is safe, and
# `test_every_vocabulary_exemption_is_still_in_use` refuses a stale one -- an exemption that
# outlives its field silently covers the next thing to use that name.
PUBLISHED_FIELDS = {
    # A boolean stating the invariant: no correctness information is served before the freeze. Its
    # name contains "score" because it is about the absence of one.
    "no_score_before_freeze": "asserts the no-score-before-freeze policy; carries no score",
}


def _screenable(page: str) -> str:
    """The executable page with reads of the PUBLISHED baseline removed.

    `baseline.pass_rate` is the pinned model's own unaided score on the task. It ships inside the
    challenge packet, which is committed to a public repository on purpose -- it is the bar a
    submission is measured against, and a board that hid it would be a list of names with nothing
    to judge them by. A *submission's* pass rate is the thing that must never appear before the
    round grades.

    A vocabulary check cannot tell those apart, because they use the same word. The API has the
    same problem and solves it by provenance: `_screened_body` deliberately applies only the
    withheld-body refusal to `public_view()`, because a blanket verdict-word screen would reject
    the one payload whose entire purpose is to publish verdicts after grading. This does the same
    thing one level down -- reads rooted at `baseline.` are the published block, and everything
    else stays strict.

    Narrow on purpose. `test_the_screen_still_catches_a_submission_pass_rate` holds it there.
    """
    import re

    code = _executable(page)
    for field in PUBLISHED_FIELDS:
        code = code.replace(field, "")
    return re.sub(r"baseline\.[A-Za-z_]+", "", code)


def test_the_board_never_carries_withheld_vocabulary():
    """The page is exempt from `_screened` because it serves a file rather than a payload, so the
    file is screened instead. These names must never appear at any stage of any round: a board that
    published a withheld check, or the secret that opens one, would end the competition it reports
    on."""
    from hermes.round import WITHHELD_KEYS

    code = _screenable(api.DASHBOARD.read_text(encoding="utf-8"))
    for word in WITHHELD_KEYS:
        assert word not in code, f"the board reads or renders {word!r}"


def test_the_board_gates_verdicts_on_the_round_state():
    """Verdict vocabulary used to be refused outright here, because the board only listed receipts
    and had no business naming a verdict. It has a verdicts panel now, and refusing the word would
    refuse the feature.

    The property was never really about vocabulary. It is: **no correctness information before the
    round is graded** -- which is about WHEN, and a word check cannot express when. Two things
    enforce it instead, and this test pins the second:

      1. `hermes.round.public_view` withholds `verdicts` entirely until GRADED, so the board cannot
         render what it is never sent. That is the real guarantee and it is tested server-side.
      2. The page gates its own rendering on the state, so a validator that somehow served early
         verdicts still would not display them.

    Belt and braces, and the braces are here."""
    page = api.DASHBOARD.read_text(encoding="utf-8")
    assert 'graded", "settled"' in page or '"graded", "settled"' in page, (
        "the verdict panel must name the states in which verdicts may be shown"
    )
    # And it must say so when they cannot exist, rather than rendering an empty table that reads as
    # a round nobody submitted to.
    assert "No verdicts until the round is graded" in page


def test_every_vocabulary_exemption_is_still_in_use():
    """An exemption for a field the board no longer reads is an exemption for whatever takes that
    name next."""
    page = api.DASHBOARD.read_text(encoding="utf-8")
    for field, reason in PUBLISHED_FIELDS.items():
        assert field in page, f"{field} is exempted and unused"
        assert reason, field


def test_the_screen_still_catches_a_submission_pass_rate():
    """The exemption above is for the published baseline and nothing else. A read of a submission's
    own rate, or a bare one, must still be refused -- otherwise the narrowing has swallowed the
    property it was carved out of."""
    assert "pass_rate" not in _screenable("<td>${baseline.pass_rate}</td>")
    assert "pass_rate" in _screenable("<td>${submission.pass_rate}</td>")
    assert "pass_rate" in _screenable("<td>${pass_rate}</td>")
    assert "pass_rate" in _screenable("<td>${verdict.pass_rate}</td>")


def test_the_screen_would_catch_a_verdict_column():
    """A guard nobody has seen fail is a guard nobody knows works -- and this one had to be
    narrowed to the executable part, which is exactly where a narrowing can go one step too far."""
    leaky = "<td>${s.score}</td>\n<!-- a comment mentioning score is fine -->"
    assert "score" in _executable(leaky)
    assert "score" not in _executable("<!-- there is no score column -->")
    assert "weight" not in _executable("<style>th { font-weight: 650; }</style>")
    assert "weight" in _executable("<td>${s.weight}</td>"), "narrowed to the styles, not past them"


def test_an_empty_round_and_an_unreachable_validator_are_different_messages():
    """Both produce no rows, and only one of them means the board is lying. Structural rather than
    behavioural -- the branch is client-side -- but the failure it guards is a copy-paste that
    makes the two read the same, which this does catch."""
    page = api.DASHBOARD.read_text(encoding="utf-8")
    assert "No submissions yet" in page
    assert page.count("This is not an empty round.") == 2, "both failure paths say so"


def test_a_missing_asset_is_a_500_rather_than_an_empty_board(client, monkeypatch, tmp_path):
    """The one reading this page must never give. A 200 with no rows would render as a round
    nobody has submitted to."""
    monkeypatch.setattr(api, "DASHBOARD", tmp_path / "gone.html")
    assert client.get("/").status_code == 500


# --- the guard that protects it, which could not fire on the case it names ------------------------


def test_the_handler_list_is_read_off_the_app_not_maintained_by_hand():
    """It was a literal list, so the guard could not catch what its own docstring described: a
    handler added by someone who never read the module docstring is also a handler nobody adds to
    a list. Derived from `app.routes`, a new endpoint is covered when it is registered."""
    names = {h.__name__ for h in api.route_handlers()}
    assert {"health", "current_round", "submissions", "submit", "reveal", "dashboard"} <= names


def test_a_newly_registered_leaky_route_is_caught_without_being_listed():
    """The proof the previous test is about something. This registers a route the way a future
    contributor would and asserts the guard notices, with nothing added to any list."""

    @api.app.get("/v1/oops")
    def oops() -> dict[str, str]:
        return {"score": "0.91"}

    try:
        assert "oops" in api.unscreened_handlers()
    finally:
        api.app.router.routes = [r for r in api.app.router.routes if getattr(r, "endpoint", None) is not oops]
    assert api.unscreened_handlers() == []


def test_every_exemption_names_a_real_handler():
    """An exemption that outlives its handler silently covers the next handler to take the name."""
    names = {h.__name__ for h in api.route_handlers()}
    assert set(api.SERVES_NO_DATA) <= names
    assert all(reason for reason in api.SERVES_NO_DATA.values()), "each exemption states why"


# --- reading this validator from a board hosted somewhere else -------------------------------------


def test_no_cross_origin_reads_are_allowed_by_default(monkeypatch):
    """Off unless asked for. A validator that shipped permissive CORS would let a page anywhere
    drive reads from a visitor's browser, and nobody deploying one would have chosen that."""
    monkeypatch.delenv("SPARK_BOARD_ORIGINS", raising=False)
    import importlib

    from validator import api as module

    reloaded = importlib.reload(module)
    try:
        assert reloaded.READ_ORIGINS == ()
        names = [m.cls.__name__ for m in reloaded.app.user_middleware]
        assert "CORSMiddleware" not in names
    finally:
        importlib.reload(module)


def test_a_named_origin_gets_read_access_and_nothing_more(monkeypatch):
    """Reads only, and only the origins named. `allow_methods` is GET, so a submission cannot be
    posted cross-origin however the browser is coaxed -- the write path takes untrusted input and
    CORS is not an authentication mechanism."""
    monkeypatch.setenv("SPARK_BOARD_ORIGINS", "https://gittensor-model-hub.github.io")
    import importlib

    from validator import api as module

    reloaded = importlib.reload(module)
    try:
        assert reloaded.READ_ORIGINS == ("https://gittensor-model-hub.github.io",)
        cors = [m for m in reloaded.app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
        assert len(cors) == 1
        options = cors[0].kwargs
        assert options["allow_methods"] == ["GET"]
        assert options["allow_credentials"] is False
        assert "*" not in options["allow_origins"], "a wildcard would invite any page to drive reads"
    finally:
        monkeypatch.delenv("SPARK_BOARD_ORIGINS", raising=False)
        importlib.reload(module)

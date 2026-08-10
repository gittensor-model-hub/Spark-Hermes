"""The validator's HTTP surface, driven against a real round.

Every test here goes through the app rather than calling `hermes.round` directly, because the
property being defended is what a *miner* can read over the wire. A round object that withholds
verdicts correctly is no use if the server serialises them anyway, and that gap is exactly
where the first version of this module was broken: it re-screened `Round.public_view()` against
`PUBLIC_VIEW_FIELDS`, which describes the metadata block and not the view, so `challenge` read
as an unexpected field and every call would have 500'd.
"""

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
    not, so `Round.reveal(master_salt)` being called with no argument went unnoticed until
    pyright flagged it -- a runtime crash on the one endpoint an auditor depends on."""
    r = _round()
    api.ROUNDS["r-1"] = r
    r.freeze(now=1_001.0)
    r.grade(now=1_002.0)
    r.settle(now=1_003.0)
    # Published the way the private side would: by calling Round.reveal with the master, which
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
    """Structural, not aspirational. `Round.reveal` takes the master and derives the per-task
    salt; calling it here would put the secret that seals every unspent task in the corpus
    inside the process answering untrusted requests."""
    import inspect

    source = inspect.getsource(api)
    assert ".reveal(" not in source
    assert "master_salt" not in source.replace("master salt", "").replace("the master", "")

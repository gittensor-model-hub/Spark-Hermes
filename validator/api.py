"""The validator's HTTP surface: round state out, nothing that decides a score in.

    uv run uvicorn validator.api:app --host 127.0.0.1 --port 8080

Read-mostly on purpose, and the reason is the identity decision. Miners are identified by the
GitHub pull request they open, so GitHub is the submission transport and `rollout_track.yml`
is the intake -- reading a submission without ever checking it out, because that job holds
secrets. This server does not accept submissions. It publishes what a miner needs in order to
work and what an auditor needs in order to check the result afterwards, and that is all.

## The one property this file exists to protect

**No correctness information may leave here before the round freezes.** A submission arrives as
a public pull request, so anything this server says about it is visible to the miner *and to
every competitor at once*. A hidden pass/fail served during an open round turns the withheld
verifier into a check-your-guess oracle: resubmit, read the response, bisect onto the check.
`overfit_rate` is the only measurement that catches a strategy which learned the published
check rather than the job, and it depends entirely on the withheld half staying withheld.

So the guard is at the serialisation boundary rather than in each handler's good intentions.
Every response goes through `_screened`, which calls `hermes.round.screen_public_payload`:
allowlist first, then a refusal of any withheld body, then a refusal of any verdict-shaped key.
A handler that forgets is caught by `test_every_route_screens_its_response`, which reads this
module's own source -- because the failure mode is a new endpoint added six months from now by
someone who never read this docstring.

## Why 409 rather than 404 on a premature read

`/results` before GRADED and `/reveal` before SETTLED answer 409 Conflict, not 404. A 404 says
"no such thing" and invites a miner to keep polling for it to appear; a 409 says "this exists
and the round is not there yet", which is true and is not a hint. The state is already public
in `/round/current`, so nothing is concealed by saying so.
"""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import FastAPI, HTTPException

from hermes.round import (
    GRADED,
    PUBLIC_VIEW_FIELDS,
    RECEIPT_FIELDS,
    SETTLED,
    RoundWindow,
    ScoreLeakError,
    refuse_withheld_body,
    screen_public_payload,
)

# Rounds this process is serving, keyed by round_id. In memory because `hermes.round` is pure
# logic with no storage of its own, and because the store a real deployment wants -- a file, a
# database, a git-tracked JSON directory -- is a deployment decision rather than an API one.
# Swapping it is one assignment; baking a database in here would not be.
ROUNDS: dict[str, RoundWindow] = {}

# Opened commitments, keyed by round_id, published by whatever settles the round.
#
# The API does NOT hold the master salt, and this dict is why. `RoundWindow.reveal` takes the master
# and derives the per-task salt from it, so calling it here would put the secret that seals
# every unspent task in the corpus inside the process that answers untrusted requests. Pyright
# caught the first version doing exactly that -- it flagged the missing argument, and the honest
# fix was not to pass the secret in but to stop needing it.
#
# So the private side derives the reveal at settle time, where the master already lives, and
# publishes the record here. This process serves a value it could not have computed.
REVEALS: dict[str, dict[str, Any]] = {}


def _screened(payload: dict[str, Any], *, allowed: frozenset[str], where: str) -> dict[str, Any]:
    """Full screening: allowlist, no withheld body, no verdict-shaped key.

    For payloads this module assembles itself, where the field set is known and no verdict
    belongs in it at any point in the round.

    A `ScoreLeakError` becomes a 500, deliberately. It means this server was about to publish
    something it must not, and the caller is not at fault -- answering 4xx would blame the
    miner for a bug on this side and, worse, would look like a normal refusal in a log.
    """
    try:
        return screen_public_payload(payload, allowed=allowed, where=where)
    except ScoreLeakError as exc:
        raise HTTPException(status_code=500, detail=f"refused to publish: {exc}") from exc


def _screened_body(payload: dict[str, Any], *, where: str) -> dict[str, Any]:
    """Withheld-body refusal only, for payloads `hermes.round` has already screened.

    The distinction is not a weakening, and getting it wrong is how this file would have
    shipped broken. `RoundWindow.public_view` and `RoundWindow.to_record` build their own payloads and screen
    the parts that need it -- `public_view` refuses a withheld body in the challenge packet and
    withholds `verdicts` entirely until GRADED; `to_record` screens its ledger extras against
    `LEDGER_FIELDS` and each receipt against `RECEIPT_FIELDS`.

    Re-screening their whole output here would be wrong twice. `PUBLIC_VIEW_FIELDS` describes
    the metadata block, not the view, so `challenge` would be rejected as unexpected. And
    `verdicts` is itself in `VERDICT_WORDS` -- correctly, because it must never appear early --
    so a blanket verdict-word refusal would reject the one payload whose entire purpose is to
    publish verdicts after grading. Both would have turned every call into a 500 while looking
    like extra caution.

    What still applies unconditionally is the withheld body: no state of the round makes a
    `hidden_verify` string publishable from here, so that check is repeated at the boundary
    where the leak would become public.
    """
    try:
        return refuse_withheld_body(payload, where=where)
    except ScoreLeakError as exc:
        raise HTTPException(status_code=500, detail=f"refused to publish: {exc}") from exc


def _round_or_404(round_id: str) -> RoundWindow:
    found = ROUNDS.get(round_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no round {round_id!r} on this validator")
    return found


app = FastAPI(
    title="Spark-Hermes validator",
    summary="RoundWindow state and audit records. Submissions arrive as GitHub pull requests.",
)


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return _screened(
        {"schema_version": "spark-round-v1", "state": "ok", "round_id": "", "task_id": ""},
        allowed=PUBLIC_VIEW_FIELDS,
        where="GET /v1/health",
    )


@app.get("/v1/round/current")
def current_round() -> dict[str, Any]:
    """The newest round's public view. 404 while no round has been opened."""
    if not ROUNDS:
        raise HTTPException(status_code=404, detail="no round has been opened on this validator")
    newest = max(ROUNDS.values(), key=lambda r: r.opened_at)
    return _screened_body(newest.public_view(), where="GET /v1/round/current")


@app.get("/v1/round/{round_id}")
def round_view(round_id: str) -> dict[str, Any]:
    return _screened_body(_round_or_404(round_id).public_view(), where=f"GET /v1/round/{round_id}")


@app.get("/v1/round/{round_id}/challenge")
def challenge(round_id: str) -> dict[str, Any]:
    """The challenge packet: the public half, and the commitment to the withheld half.

    `Challenge.to_record` already strips anything outside `PUBLISHABLE_TASK_KEYS` and reports
    the names it dropped, so this endpoint does not re-derive that rule -- it serves the packet
    and lets `_screened` refuse it if the packet itself ever carries a body.
    """
    record = _round_or_404(round_id).challenge.to_record()
    # Not PUBLIC_VIEW_FIELDS: a challenge packet has its own shape. The withheld-body and
    # verdict-word refusals still apply, which is the part that matters here.
    return _screened_body(record, where=f"GET /v1/round/{round_id}/challenge")


@app.get("/v1/round/{round_id}/receipts")
def receipts(round_id: str) -> dict[str, Any]:
    """Envelope facts per miner: accepted, malformed, refused, late, and what replaced what.

    Served while the round is OPEN on purpose. A miner has to be able to tell a broken upload
    from a rejected one, or every debugging cycle is a guess -- and none of these fields say
    anything about correctness. That distinction is the whole reason `RECEIPT_FIELDS` and the
    verdict allowlist are separate frozensets.
    """
    found = _round_or_404(round_id)
    return {
        "round_id": found.round_id,
        "receipts": [
            _screened(r.to_record(), allowed=RECEIPT_FIELDS, where=f"receipt {r.miner}") for r in found.receipts
        ],
    }


@app.get("/v1/round/{round_id}/results")
def results(round_id: str) -> dict[str, Any]:
    """Verdicts. Refused until the round has been graded.

    The state check is here as well as inside `hermes.round`, and the duplication is wanted:
    this is the boundary where a leak becomes public, and a guard that lives only one layer
    down is one refactor away from not being on this path.
    """
    found = _round_or_404(round_id)
    if found.state not in (GRADED, SETTLED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"round {round_id} is {found.state!r}; verdicts are published after grading. "
                "Serving one now would let a miner resubmit against the withheld check and "
                "bisect onto it one attempt at a time."
            ),
        )
    return _screened_body(found.to_record(), where=f"GET /v1/round/{round_id}/results")


@app.get("/v1/round/{round_id}/reveal")
def reveal(round_id: str) -> dict[str, Any]:
    """The opened commitment: this task's derived salt, so anyone can audit the grading.

    Only once SETTLED. Publishing the salt is what turns the commitment into an audit -- an
    auditor holding the check body recomputes the digest and confirms the validator graded
    against the check it committed to before submissions opened, rather than one written
    afterwards.

    Safe only because salts are per-task. `hermes.harness.derive_task_salt` is
    `HMAC(master, task_id)`, so releasing this task's salt says nothing about any other. Under
    one shared salt this endpoint would unseal every unspent task in the corpus on its first
    successful call.
    """
    found = _round_or_404(round_id)
    if found.state != SETTLED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"round {round_id} is {found.state!r}; the salt is released when the round settles. "
                "Releasing it earlier would let a miner confirm a guess at the withheld check "
                "while the round is still live."
            ),
        )
    record = REVEALS.get(round_id)
    if record is None:
        # SETTLED but nothing published. Not a 404: the round exists and is in the right state,
        # so this is the private side not having posted the opened commitment yet.
        raise HTTPException(
            status_code=409,
            detail=(
                f"round {round_id} has settled but no opened commitment has been published. The "
                "salt is derived where the master secret lives, not here, so this process cannot "
                "produce one on demand."
            ),
        )
    return _screened_body(dict(record), where=f"GET /v1/round/{round_id}/reveal")


def route_handlers() -> list[Any]:
    """Every handler this module serves, for the test that checks each one screens."""
    return [
        health,
        current_round,
        round_view,
        challenge,
        receipts,
        results,
        reveal,
    ]


def unscreened_handlers() -> list[str]:
    """Handlers whose body never calls `_screened`. Should always be empty.

    Read from source rather than by calling them, because the point is to catch an endpoint
    that nobody wrote a test for. A handler added without screening is exactly the shape of the
    leak this module exists to prevent, and it would pass every behavioural test that only
    exercises the endpoints somebody remembered.
    """
    offenders: list[str] = []
    for handler in route_handlers():
        source = inspect.getsource(handler)
        if "_screened(" not in source and "_screened_body(" not in source:
            offenders.append(handler.__name__)
    return offenders


__all__ = ["REVEALS", "ROUNDS", "app", "route_handlers", "unscreened_handlers"]

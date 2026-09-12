"""The validator's HTTP surface: round state out, nothing that decides a score in.

    uv run uvicorn validator.api:app --host 127.0.0.1 --port 8080

Read-mostly, with exactly one write path: a miner uploads their surface bundle privately here and
opens a pull request carrying only its digest.

That split is the design. The bundle stays private, so the surface remains the miner's edge. The
digest is public, timestamped and attributable, so the validator cannot evaluate a different bundle
than the one committed and the miner cannot revise after the fact. Neither half works alone: a
private upload with no public commitment is unauditable, and a public surface is no longer an edge.

An earlier version of this file said "this server does not accept submissions", and the reasoning
then was that GitHub was the transport. It is now the commitment; the transport is here. Everything
about the read paths is unchanged, and the upload path is validated by `validator.intake` before a
byte reaches the filesystem.

## The one property this file exists to protect

**No correctness information may leave here before the round freezes.** A hidden pass/fail served
during an open round turns the withheld verifier into a check-your-guess oracle: resubmit, read the
response, bisect onto the check. The upload endpoint makes this sharper rather than softer -- a
miner can now submit repeatedly and cheaply, so the response to an upload must say only that it
arrived and is well-formed. It returns a receipt, and a receipt carries envelope facts and nothing
else.
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
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from hermes.round import (
    GRADED,
    OPEN,
    PUBLIC_VIEW_FIELDS,
    RECEIPT_FIELDS,
    SETTLED,
    RoundWindow,
    ScoreLeakError,
    refuse_withheld_body,
    screen_public_payload,
)

# The fields an intake receipt adds beyond `RECEIPT_FIELDS`. Named here rather than widening the
# round's own allowlist: `RECEIPT_FIELDS` describes what a *round* receipt may carry, and merging
# the two frozensets would let a round receipt start publishing intake fields and nobody would
# notice. The screen still refuses anything outside the union.
INTAKE_FIELDS = frozenset(
    {
        "submission_id",
        "round_id",
        "miner_id",
        "bundle_sha256",
        "received_at",
        "files",
        "bytes",
        "status",
        "origin",
        "mode",
        "namespace",
        "issuer",
    }
)

# Rounds this process is serving, keyed by round_id.
#
# Still a dict, and now filled from `validator.store` at startup by `load_from_store()`. It was
# filled by nothing, so a live server answered 404 to every round and a restart during grading
# lost the grading -- the endpoints were all correct and there was no path by which they could
# ever have had anything to serve.
#
# In-process on purpose. The store is the durable copy; this is a read cache in front of it, and
# keeping the cache explicit is what lets `reload()` exist as one line rather than as a
# refactor.
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


def load_from_store(store: Any = None) -> tuple[int, list[tuple[str, str]]]:
    """Fill `ROUNDS` from the durable store. Returns (loaded, failures).

    Failures are returned rather than raised, and that choice matters here more than in most
    loaders: this runs at startup, so raising would mean one unparseable snapshot takes the whole
    validator down and every *healthy* round with it. A round that cannot be read is reported and
    skipped, which is recoverable; an outage is not.

    Reveals are deliberately not loaded. The API cannot derive a per-task salt -- the master lives
    where grading happens -- so a reveal is posted here by whatever settles the round, and reading
    one off disk at startup would give this process a signed-looking value it never checked.
    """
    from validator.store import RoundStore

    store = store or RoundStore()
    loaded, failed = store.load_all()
    ROUNDS.update(loaded)
    return len(loaded), failed


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Fill `ROUNDS` from the store before the first request is served.

    A lifespan rather than `@app.on_event("startup")`, which Starlette deprecates. Loading lazily
    on first request would be worse than either: the first caller would pay for the load and, if
    it failed, would see a 404 that reads as "no round has been opened" -- the one message that
    must not be ambiguous here.
    """
    count, failed = load_from_store()
    print(f"validator.api: loaded {count} round(s) from the store")
    for round_id, why in failed:
        # Loud, because a round that silently failed to load is indistinguishable from a round
        # that was never opened, and the second is a normal state.
        print(f"validator.api: WARNING round {round_id!r} could not be loaded: {why}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Spark-Hermes validator",
    summary="RoundWindow state and audit records. Submissions arrive as GitHub pull requests.",
)


# Origins allowed to read this validator from a browser. Read-only endpoints only, named
# explicitly, and empty by default.
#
# A board hosted somewhere else -- GitHub Pages -- cannot read this API without them: a page served
# from another origin gets no response body without the server's consent. So this is what turns the
# published-snapshot board into a live one.
#
# `*` is deliberately not offered. The write path here takes untrusted input, and while CORS is not
# an authentication mechanism, a wildcard invites a page anywhere to drive it from a visitor's
# browser. And the allowlist covers reads only: `allow_methods` is GET, so a submission cannot be
# posted cross-origin at all.
READ_ORIGINS = tuple(o for o in os.environ.get("SPARK_BOARD_ORIGINS", "").split(",") if o.strip())

if READ_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in READ_ORIGINS],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
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


class SubmissionRequest(BaseModel):
    """One uploaded surface bundle.

    A JSON map of path to text rather than an uploaded archive. A tarball or zip brings path
    traversal, symlinks that resolve outside the extraction root, and decompression bombs; a map of
    strings has none of those and costs a few kilobytes of encoding on a payload that is prose.
    """

    miner_id: str
    files: dict[str, str]


@app.post("/v1/round/{round_id}/submission")
def submit(round_id: str, request: SubmissionRequest) -> dict[str, Any]:
    """Accept a private bundle. Returns a receipt and nothing about its quality.

    Refusals are 400 with every reason, because a miner who learns one problem per upload stops
    uploading. They are deliberately verbose about *shape* and silent about *merit*: this endpoint
    can be called repeatedly and cheaply, so anything it leaked about correctness would be a free
    oracle on the withheld check.

    The round must be OPEN. A bundle accepted after the freeze would sit in the store looking like a
    submission while the window that could have judged it has closed.
    """
    from validator.intake import Intake, IntakeError

    window = _round_or_404(round_id)
    if window.state != OPEN or time.time() > window.deadline:
        raise HTTPException(
            status_code=409,
            detail=(
                f"round {round_id} is {window.state!r}; uploads are accepted while it is open. A "
                "bundle taken after the freeze would sit in the store looking like a submission "
                "with no window left to judge it."
            ),
        )
    try:
        receipt = Intake().accept(round_id=round_id, miner_id=request.miner_id, files=request.files)
    except IntakeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _screened(receipt.to_record(), allowed=RECEIPT_FIELDS | INTAKE_FIELDS, where="POST submission")


@app.get("/v1/submissions")
def submissions(round_id: str = "") -> dict[str, Any]:
    """The public receipts, for the dashboard. Envelope facts only.

    Served during an open round for the same reason `/receipts` is: a miner has to be able to tell
    a rejected upload from a lost one, and where they are in the queue is not a score.
    """
    from validator.intake import Intake

    found = [r for r in Intake().read_receipts() if not round_id or r.round_id == round_id]
    return {
        "round_id": round_id,
        "submissions": [
            _screened(r.to_record(), allowed=RECEIPT_FIELDS | INTAKE_FIELDS, where=f"receipt {r.submission_id}")
            for r in found
        ],
    }


DASHBOARD = Path(__file__).with_name("dashboard.html")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """The submissions board, served by the validator that issued the receipts.

    Static, and served from this origin rather than from GitHub Pages: the page reads
    `/v1/submissions` on this host, so there is no cross-origin request to allow, no second place
    to configure a URL, and no way to aim the board at a validator whose receipts nobody
    published. It loads nothing from a third party for the reason a validator serves no
    third-party script.

    Exempt from `_screened` because it returns a file rather than a payload assembled from round
    state -- and named in `SERVES_NO_DATA` so the exemption is a listed decision rather than an
    omission. `test_the_dashboard_page_carries_no_withheld_vocabulary` screens the file itself.
    """
    try:
        return DASHBOARD.read_text(encoding="utf-8")
    except OSError as exc:
        # A packaging mistake, not a request problem. Answering 200 with an apology would leave a
        # board that renders as an empty round, which is the one reading this page must never give.
        raise HTTPException(status_code=500, detail=f"dashboard asset is missing: {exc}") from exc


# Handlers that serve no data assembled from round state, and why each is exempt from screening.
# A name here is a decision on the record; `test_every_exemption_names_a_real_handler` refuses a
# stale one, because an exemption that outlives its handler silently covers the next handler to
# take that name.
SERVES_NO_DATA = {"dashboard": "returns a static asset, screened as a file by the test suite"}


def route_handlers() -> list[Any]:
    """Every endpoint this app serves, read off the app rather than listed by hand.

    It was a hand-maintained list, which made the guard below unable to catch the case its own
    docstring named: a handler added later by someone who never read the module docstring is also
    a handler nobody adds to a list. Derived from `app.routes`, a new endpoint is covered the
    moment it is registered.
    """
    from fastapi.routing import APIRoute

    return [route.endpoint for route in app.routes if isinstance(route, APIRoute)]


def unscreened_handlers() -> list[str]:
    """Handlers whose body never calls `_screened`. Should always be empty.

    Read from source rather than by calling them, because the point is to catch an endpoint
    that nobody wrote a test for. A handler added without screening is exactly the shape of the
    leak this module exists to prevent, and it would pass every behavioural test that only
    exercises the endpoints somebody remembered.
    """
    offenders: list[str] = []
    for handler in route_handlers():
        if handler.__name__ in SERVES_NO_DATA:
            continue
        source = inspect.getsource(handler)
        if "_screened(" not in source and "_screened_body(" not in source:
            offenders.append(handler.__name__)
    return offenders


__all__ = [
    "DASHBOARD",
    "REVEALS",
    "ROUNDS",
    "SERVES_NO_DATA",
    "app",
    "load_from_store",
    "route_handlers",
    "unscreened_handlers",
]

"""The round window: publish a challenge, take submissions, freeze at the deadline.

## Two modules say "round", and they are different axes

`hermes.seed` also has a round, and for a while both classes were called `Round` with neither
referencing the other -- so `from hermes.round import Round` and `from hermes.seed import Round`
were indistinguishable at a call site, and only one of them was wired into the live submission
gate. The class here is now `RoundWindow`, which is the word this module's own prose already
used, and the two axes are:

    hermes.seed.Round   ASSIGNMENT: who works on what. States committed -> open -> closed,
                        describing when the seed is VISIBLE. Deterministic per-miner
                        assignment by rendezvous hashing from a commit-revealed seed, so a
                        miner submitting work it was not assigned is rejectable by anyone
                        holding the announcement. Used by `eval.rollout_track`.

    RoundWindow (here)  LIFECYCLE: intake, freeze, grading, settlement. States open -> frozen
                        -> graded -> settled, describing what the validator will ACCEPT and
                        PUBLISH. Enforces that no correctness information escapes before the
                        freeze.

They compose rather than compete: a real round has both an assignment and a window. This class
does not yet hold a `seed.Round`, so nothing here checks that a submitting miner was assigned
the task -- `eval.rollout_track.check_scope` does that separately. Wiring the two together is
worth doing and is not done.

`hermes.challenge` decides what is worth competing on and `hermes.acceptance` decides who
won. Between them sat the part nobody had written: the *window*. Open it, take uploads while
it is open, close it at an announced moment, and only then let a number exist. Without that
the two ends are a pair of libraries -- there is nothing to submit to, nothing to freeze, and
no record afterwards of who was in.

Logic only. No socket, no table, no chain, and no wall clock: every mutating call takes the
time as an argument. That is not purity for its own sake. The rules below are all about
*ordering* -- what arrived before the cutoff, what may be computed after it -- and ordering
rules that live inside a request handler can only be tested by starting a server and hoping
the sleep was long enough. Here a thousand rounds with adversarial timing run in a second,
and a transport can wrap this without moving a single rule into a route.

## The rule the whole module is arranged around: no score before the freeze

`submit` may report that an upload was unreadable, or that it named a file the miner contract
does not allow. It may not report anything about whether the strategy *works*.

A hidden pass/fail returned during an open round turns the withheld verifier into a
check-your-guess oracle. The miner submits, reads the verdict, edits, resubmits, and bisects
their way onto the withheld check -- and the cost is not merely inflated scores.
`overfit_rate` is defined as the gap between the published check and the withheld one, so an
oracle deletes the only measurement that can tell a general strategy from one fitted to the
grader. It does it silently, because every individual submission looks legitimate and every
individual answer was true.

The miner still needs one thing, which is why `submit` returns anything at all: a malformed
upload and a refused one must be distinguishable, or a miner with a broken tarball spends the
round debugging their strategy. So the outcome vocabulary is exactly `MALFORMED`, `REFUSED`,
`LATE`, `ACCEPTED` -- four facts about the envelope, none about the contents.

That rule is a property of the types here, not a note in a docstring:

  * `Receipt` -- the only thing `submit` returns -- has no field that could hold a verdict,
    and `_check_receipt_fields` raises at **import** if somebody adds one.
  * `Verdict` cannot be constructed without a `FreezeToken`, and the only code in the process
    that mints a token is `RoundWindow.freeze`. Nothing reachable from an open round can make one,
    so during an open round there is no verdict object in existence to leak.
  * Every serialisable payload the module can emit is screened. `screen_public_payload` is an
    allowlist for the same reason `PUBLISHABLE_TASK_KEYS` is one, with a denylist of
    verdict-shaped names behind it. The two nets fail differently on purpose: the allowlist
    catches a field nobody thought about, the denylist catches a field somebody thought about
    and named wrongly.

## The deadline admits submissions; the freeze only changes state

These were one thing in the obvious design and they must not be. `freeze` is called by
something -- a cron, a loop, an operator -- and it is called *late*, because everything is.
If admission meant "arrived before `freeze()` ran", a validator whose timer fired forty
seconds after the announced deadline would silently admit forty seconds of extra submissions,
and which miners got in would depend on scheduling jitter nobody can audit.

So `received_at > deadline` is `LATE` even while the round is still `OPEN`, and `FrozenAt`
records the admission cutoff actually applied (`deadline_used`) separately from the moment the
call happened (`frozen_at`). A reader can see that the round ran long without being able to
confuse the two numbers.

## Two payloads, both safe to publish

`public_view` is what a miner may see; `to_record` is the validator's ledger. Neither carries
a verdict before the round is `GRADED`, including the ledger, and that is deliberate. Making
only the miner-facing payload safe leaves the ledger safe-by-operator-discipline -- and this
module exists because operator discipline is what fails. A grader that needs raw verdicts
mid-grading reads `RoundWindow.verdicts`, which returns in-process objects and never a dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

from hermes.challenge import Challenge
from hermes.harness import derive_task_salt, digest_mapping, salted_digest

SCHEMA_VERSION = "spark-round-v1"

# --- the lifecycle ---------------------------------------------------------------------------

OPEN = "open"
FROZEN = "frozen"
GRADED = "graded"
SETTLED = "settled"

LIFECYCLE: tuple[str, ...] = (OPEN, FROZEN, GRADED, SETTLED)

# Written as a table rather than "index + 1" over LIFECYCLE. An index comparison silently
# accepts whatever a future state gets inserted next to, and the mistake it would wave through
# is the expensive one: SETTLED reachable from FROZEN skips grading, so a round reveals its
# per-task salt while no verdict has been recorded -- the withheld check opened for a round it
# never judged.
_NEXT: dict[str, tuple[str, ...]] = {
    OPEN: (FROZEN,),
    FROZEN: (GRADED,),
    GRADED: (SETTLED,),
    SETTLED: (),
}

# --- what may be said about an upload --------------------------------------------------------

# The submission stands. Says nothing about whether it works.
ACCEPTED = "accepted"
# Not a submission: a field is missing or the wrong type. The miner's tooling is broken.
MALFORMED = "malformed"
# A submission, asking for something the miner contract does not allow.
REFUSED = "refused"
# Arrived after the admission cutoff. Recorded rather than dropped, so "my upload went
# through" and "I have no record of it" cannot both be true.
LATE = "late"

SUBMISSION_OUTCOMES: tuple[str, ...] = (ACCEPTED, MALFORMED, REFUSED, LATE)


class RoundError(ValueError):
    """A round cannot do what was asked of it."""


class RoundStateError(RoundError):
    """The round is in the wrong state for this operation. Names the state it is in."""


class ScoreLeakError(RoundError):
    """A payload would have carried correctness information out of an ungraded round.

    Not a caller error -- a defect in this module or in something that extended it. Raised
    rather than logged because the failure is invisible in production: the leaked field is
    valid JSON that a miner's client reads happily, and nothing downstream complains.
    """


# --- the screens -----------------------------------------------------------------------------

# Keys that may appear in a receipt. An allowlist rather than a denylist for the reason
# `hermes.challenge.PUBLISHABLE_TASK_KEYS` gives: a denylist keeps working right up until
# somebody adds a second field of the kind it was filtering.
RECEIPT_FIELDS = frozenset(
    {
        "round_id",
        "miner",
        "outcome",
        "revision",
        "replaced_previous",
        "replaced_digest",
        "submission_digest",
        "standing_digest",
        "problems",
        "received_at",
    }
)

# Keys the round's own metadata may publish. `challenge` and `verdicts` are assembled outside
# this screen and documented at the assembly site -- see `RoundWindow.public_view`.
PUBLIC_VIEW_FIELDS = frozenset(
    {
        "schema_version",
        "round_id",
        "task_id",
        "state",
        "opened_at",
        "deadline",
        "submissions",
        "replacements",
        "withheld_check_committed",
        "no_score_before_freeze",
        "frozen",
        "reveal_available",
    }
)

# What the validator's ledger adds on top of the public view, screened on its own because the
# ledger is the payload most likely to grow a field: it is where somebody debugging a round
# reaches for "just add the grader output next to the submission".
LEDGER_FIELDS = frozenset(
    {
        "standing_submissions",
        "attempts",
        "verdicts_recorded",
        "verdicts_withheld_until",
        "graded_at",
        "settled_at",
    }
)

# One entry of `standing_submissions`. Envelope facts about an upload, the same rule as a receipt.
SUBMISSION_RECORD_FIELDS = frozenset(
    {
        "miner",
        "revision",
        "submission_digest",
        "payload_digest",
        "paths",
        "received_at",
        "replaced_digest",
        "replacements",
    }
)

# Second net, behind the allowlists. Its whole job is to catch the realistic mistake: a
# maintainer adding a field, updating the allowlist to match, and not noticing what the field
# carries. Exact names, not substrings -- substring matching flagged the challenge packet's own
# `acceptance_thresholds_included`, and a screen that cries wolf is a screen somebody disables.
VERDICT_WORDS = frozenset(
    {
        "passed",
        "failed",
        "pass_rate",
        "score",
        "scores",
        "verdict",
        "verdicts",
        "grade",
        "graded_as",
        "reward",
        "weight",
        "weights",
        "correct",
        "hidden_passed",
        "public_passed",
        "hidden_result",
        "token_reduction",
        "decision",
        "rank",
        "placement",
        "accepted_by_gate",
    }
)

# Names under which the withheld half could travel. Screened separately from the verdict
# words, and over the challenge packet too, because publishing the round is the moment the
# body would actually escape and this module is the thing doing the publishing.
WITHHELD_KEYS = frozenset(
    {
        "hidden_verify",
        "hidden_verify_body",
        "hidden_verify_script",
        "hidden_tests",
        "hidden_test_body",
        "master_salt",
        "salt",
        "task_salt",
        "hidden_verify_salt",
    }
)


def _keys(payload: Any) -> list[str]:
    """Every key name anywhere in a nested payload."""
    found: list[str] = []
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                found.append(str(key))
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return found


def refuse_withheld_body(payload: dict[str, Any], *, where: str) -> dict[str, Any]:
    """Return `payload` unchanged, or refuse it for carrying the withheld half.

    Applied to the challenge packet even though `hermes.challenge` already strips it. That is
    not redundancy for its own sake: the packet's allowlist protects the `task` sub-dict, and
    this module publishes the *whole* record. A future field added one level up -- a debug
    dump, an "original task" echo -- would sail past an allowlist that only ever looked
    inside `task`.
    """
    for key in _keys(payload):
        if key.lower() in WITHHELD_KEYS:
            raise ScoreLeakError(
                f"{where}: the payload carries {key!r}, which is the withheld half. Publishing it "
                "hands every miner the answer key, and overfit_rate stops measuring anything "
                "because there is no longer a check the miner has not seen."
            )
    return payload


def screen_public_payload(payload: dict[str, Any], *, allowed: frozenset[str], where: str) -> dict[str, Any]:
    """Return `payload` unchanged, or refuse it. Allowlist first, verdict words behind it."""
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ScoreLeakError(
            f"{where}: {unexpected} are not in the published field set. Anything unrecognised is "
            "refused rather than assumed harmless -- a field nobody has considered is exactly how "
            "correctness information reaches a miner before the freeze."
        )
    refuse_withheld_body(payload, where=where)
    for key in _keys(payload):
        if key.lower() in VERDICT_WORDS:
            raise ScoreLeakError(
                f"{where}: {key!r} is verdict-shaped. A round that reports correctness before it "
                "freezes turns the withheld verifier into a check-your-guess oracle, and a miner "
                "can bisect their way onto the withheld check one resubmission at a time."
            )
    return payload


# --- submissions -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Submission:
    """One miner's standing upload. Paths and a digest -- never the bytes.

    A round is a lifecycle, not a file store. Holding the archive here would make every test
    of the state machine an I/O test, and the transport that eventually stores blobs has no
    business owning the deadline rules.
    """

    miner: str
    paths: tuple[str, ...]
    payload_digest: str
    received_at: float
    revision: int = 1
    replaced_digest: str = ""

    @property
    def digest(self) -> str:
        """Content address of the submission as the round saw it."""
        return digest_mapping(
            {
                "miner": self.miner,
                "paths": list(self.paths),
                "payload_digest": self.payload_digest,
                "revision": self.revision,
            }
        )


@dataclass(frozen=True)
class Receipt:
    """What a miner gets back from `submit`, and the whole of what they get back.

    Every field here is a fact about the envelope: did it parse, did it name allowed files,
    did it arrive in time, which of their uploads is now standing. There is deliberately no
    field a correctness result could be put into, and `_check_receipt_fields` fails the import
    if one appears. A `Decision` from `hermes.acceptance` cannot be smuggled through here
    because there is nowhere to put it.

    `revision` and `standing_digest` describe the submission that *counts*, which is not
    always the one just uploaded -- see `RoundWindow.submit` on why a malformed upload must not
    displace a good one. A `revision` of 0 with an empty `standing_digest` means nothing of
    this miner's stands: the upload was rejected and there was no earlier one to fall back to.
    """

    round_id: str
    miner: str
    outcome: str
    revision: int
    replaced_previous: bool
    replaced_digest: str
    submission_digest: str
    standing_digest: str
    problems: tuple[str, ...]
    received_at: float

    def __post_init__(self) -> None:
        if self.outcome not in SUBMISSION_OUTCOMES:
            raise RoundError(
                f"{self.outcome!r} is not a submission outcome. The vocabulary is fixed at "
                f"{list(SUBMISSION_OUTCOMES)} because it is the complete list of things a miner may "
                "learn from uploading, and a new member is how a correctness hint gets a name."
            )

    @property
    def accepted(self) -> bool:
        """Whether the upload now stands. Not whether it is any good."""
        return self.outcome == ACCEPTED

    def to_record(self) -> dict[str, Any]:
        record = {
            "round_id": self.round_id,
            "miner": self.miner,
            "outcome": self.outcome,
            "revision": self.revision,
            "replaced_previous": self.replaced_previous,
            "replaced_digest": self.replaced_digest,
            "submission_digest": self.submission_digest,
            "standing_digest": self.standing_digest,
            "problems": list(self.problems),
            "received_at": self.received_at,
        }
        return screen_public_payload(record, allowed=RECEIPT_FIELDS, where=f"receipt {self.miner}")


def _check_receipt_fields() -> None:
    """Fail the import if `Receipt` grew a field the allowlist has not seen.

    The tripwire, and the reason the allowlist is worth having at all. A maintainer adding
    `hidden_passed` to `Receipt` gets an ImportError on the next test run instead of a subnet
    that has been answering "is my strategy correct?" for a week.
    """
    declared = {f.name for f in fields(Receipt)}
    if declared != set(RECEIPT_FIELDS):
        added = sorted(declared - RECEIPT_FIELDS)
        removed = sorted(RECEIPT_FIELDS - declared)
        raise ScoreLeakError(
            f"Receipt's fields and RECEIPT_FIELDS disagree (added {added}, removed {removed}). A "
            "receipt is the only thing an open round tells a miner, so its field set is reviewed "
            "deliberately rather than inherited from whatever the dataclass happens to hold."
        )


_check_receipt_fields()


@lru_cache(maxsize=1)
def default_contract() -> Any:
    """The repo's miner contract, loaded once.

    Cached because `submit` is the hot path a transport would call and re-reading the JSON per
    upload makes the contract file's mtime part of the round's behaviour.
    """
    from hermes import miner_contract

    return miner_contract.load()


# --- the freeze ------------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeToken:
    """Proof that a round actually froze, and the only key that unlocks `Verdict`.

    A capability rather than a data record. `Verdict.__init__` demands one and `RoundWindow.freeze`
    is the only function that mints one, so there is no path from an open round to a verdict
    object -- not a discouraged path, an absent one. `RoundWindow.record_verdict` compares tokens by
    identity, so a token reconstructed with the right field values is still not the token this
    round minted.
    """

    round_id: str
    frozen_at: float
    deadline_used: float
    seal_digest: str


@dataclass(frozen=True)
class FrozenAt:
    """What the freeze did, including the parts that differed from the announcement.

    `deadline_used` is the admission cutoff actually applied; `frozen_at` is when the call
    happened. Separate fields because they differ in every real deployment -- the timer fires
    late -- and collapsing them would make "was my 12:00:01 upload in?" unanswerable from the
    record.
    """

    frozen_at: float
    deadline_used: float
    scheduled_deadline: float
    early: bool
    reason: str
    sealed_submissions: int
    seal_digest: str

    @property
    def overran_s(self) -> float:
        """How long after the announced deadline the freeze actually ran. Reported, not policed."""
        return round(max(0.0, self.frozen_at - self.scheduled_deadline), 6)

    def to_record(self) -> dict[str, Any]:
        return {
            "frozen_at": self.frozen_at,
            "deadline_used": self.deadline_used,
            "scheduled_deadline": self.scheduled_deadline,
            "early": self.early,
            "reason": self.reason,
            "overran_s": self.overran_s,
            "sealed_submissions": self.sealed_submissions,
            "seal_digest": self.seal_digest,
        }


@dataclass(frozen=True)
class Verdict:
    """One miner's result. Cannot exist before the freeze, by construction.

    The `token` argument is the whole point of the class. Correctness for a submission is
    computable the instant the submission lands -- the validator holds the withheld check --
    so nothing about the *data* stops a grader being called from inside a request handler. What
    stops it is that the result has nowhere to live: constructing this requires a `FreezeToken`
    and only `RoundWindow.freeze` mints one.

    `passed` is the withheld verdict. `decision` is left to `hermes.acceptance`, which owns the
    bar; a round records what happened and does not re-derive who won.
    """

    token: FreezeToken
    miner: str
    submission_digest: str
    passed: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.token, FreezeToken):
            raise RoundStateError(
                "a verdict needs the FreezeToken its round minted. Correctness is computable the "
                "moment a submission lands, so the only thing keeping it out of an open round is "
                "that there is no token to build this with until the window has closed."
            )
        if not self.miner:
            raise RoundError("a verdict about no miner scores nobody and inflates the denominator")

    def to_record(self) -> dict[str, Any]:
        """Verdict-shaped by design. Reachable only through a GRADED round -- see `RoundWindow.to_record`."""
        return {
            "miner": self.miner,
            "submission_digest": self.submission_digest,
            "passed": self.passed,
            "notes": self.notes,
        }


# --- the reveal ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reveal:
    """A settled round's opened commitment: the per-task salt, and what it proves.

    ## Why this is safe, and why a shared salt would have made it unsafe

    The commitment in the challenge packet is `salted_digest(hidden_verify, salt)`. Publishing
    it during the round proves the withheld check was fixed before anybody submitted, without
    saying what it is. Releasing the salt afterwards is what turns that into an audit: anyone
    holding the check body can recompute the digest and confirm the validator graded against
    the check it committed to, rather than one written after seeing the submissions. That is
    the entire reason a settled state exists.

    `hermes.harness.derive_task_salt` makes the release safe by giving every task its own
    salt, derived as `HMAC(master, task_id)`. HMAC is a PRF keyed on the master, so `salt_i`
    tells an attacker nothing about `salt_j`.

    Under one salt shared across the corpus the first reveal would be a catastrophe, and a
    quiet one. Revealing the salt for this spent task would reveal *the* salt, so every
    commitment still sealed on every unspent task becomes brute-forceable -- for exactly the
    reason `salted_digest` refuses a bare digest. A withheld check is a short shell command
    from a small space, usually a near neighbour of the `verify` published beside it, so a few
    hundred guesses recover it. Every sealed commitment turns back into the check-your-guess
    oracle this module's docstring is about, and it happens at the moment of the first
    successful audit: doing the honest thing would have broken every future round.

    So settling one round opens one task and leaves every other commitment sealed.
    """

    round_id: str
    task_id: str
    salt: str
    commitment: str
    settled_at: float

    def opens(self, hidden_verify_body: str) -> bool:
        """Whether this body under this salt reproduces the published commitment.

        The audit itself. A validator that graded against a different check than the one it
        committed to fails here, which is the only way anyone outside the validator could ever
        find that out.
        """
        if not self.commitment:
            return False
        return salted_digest(hidden_verify_body, self.salt) == self.commitment

    def to_record(self) -> dict[str, Any]:
        """Deliberately not part of `public_view`.

        A salt in the payload a miner polls every few seconds is one refactor away from being
        published by a round that has not settled. Handing it back only from an explicit
        `RoundWindow.reveal(master)` call keeps the one payload that legitimately carries a secret
        out of the one that is fetched constantly.
        """
        return {
            "round_id": self.round_id,
            "task_id": self.task_id,
            "per_task_salt": self.salt,
            "hidden_verify_commitment": self.commitment,
            "settled_at": self.settled_at,
            "opens_only_this_task": True,
            "why": (
                "the salt is HMAC(master, task_id), so opening this task leaves every other "
                "commitment sealed; under one shared salt this reveal would have made every "
                "unspent withheld check brute-forceable"
            ),
        }


# --- the round -------------------------------------------------------------------------------


@dataclass
class RoundWindow:
    """One competition window over one challenge.

    Construct through `open_round`, which refuses the arguments that produce a window nobody
    can submit to.
    """

    challenge: Challenge
    round_id: str
    opened_at: float
    deadline: float
    contract: Any = None
    state: str = field(default=OPEN, init=False)
    _clock: float = field(default=0.0, init=False)
    _submissions: dict[str, Submission] = field(default_factory=dict, init=False)
    _replacements: dict[str, int] = field(default_factory=dict, init=False)
    _receipts: list[Receipt] = field(default_factory=list, init=False)
    _verdicts: dict[str, Verdict] = field(default_factory=dict, init=False)
    _frozen: FrozenAt | None = field(default=None, init=False)
    _token: FreezeToken | None = field(default=None, init=False)
    _graded_at: float | None = field(default=None, init=False)
    _settled_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._clock = self.opened_at

    # --- identity -----------------------------------------------------------------------

    @property
    def task_id(self) -> str:
        return self.challenge.task_id

    @property
    def withheld_check_committed(self) -> bool:
        """Whether this round can measure overfit at all.

        Recorded rather than required. A task with no withheld check is a legitimate task, but
        a round over one cannot produce an `overfit_rate` and nothing downstream should read
        one from it. Refusing the round outright would be the wrong refusal -- it would block
        an honest task -- so the fact is published instead of assumed.
        """
        return bool(self.challenge.to_record()["withheld"]["hidden_verify_commitment"])

    # --- clock --------------------------------------------------------------------------

    def _advance(self, now: float, *, what: str) -> None:
        """Refuse a clock that runs backwards.

        Two transport workers with skewed clocks is not hypothetical, and the failure is
        exactly the one "only the last submission counts" is supposed to prevent: a
        replacement stamped a second earlier than the upload it replaces arrives second, wins
        the `last` comparison, and the miner is graded on the version they withdrew. Refused
        rather than sorted-by-timestamp, because a sort makes the outcome depend on which
        worker's clock was wrong.

        A time that is not a number is refused here by name rather than left to raise a
        comparison TypeError from inside the ordering check. Every ordering rule in this module
        rests on this value, so "the round cannot order this event" is the useful complaint; a
        `'<' not supported between str and float` sends the reader to the wrong layer.
        """
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise RoundError(
                f"{self.round_id}: {what} carries a time of {now!r}, which the round cannot order. "
                "Admission, replacement and the freeze are all decided by comparing these, so an "
                "unorderable timestamp is refused rather than coerced into one that sorts wrongly."
            )
        if now < self._clock:
            raise RoundError(
                f"{self.round_id}: {what} at {now} is earlier than {self._clock}, which this round has "
                "already seen. A clock that runs backwards makes 'the last submission counts' depend on "
                "which worker's clock was wrong, and the miner is graded on the version they withdrew."
            )
        self._clock = now

    # --- transitions --------------------------------------------------------------------

    def _require(self, expected: str, *, what: str) -> None:
        if self.state != expected:
            raise RoundStateError(
                f"{self.round_id}: cannot {what} while the round is {self.state.upper()}; "
                f"that needs {expected.upper()}. The lifecycle is "
                f"{' -> '.join(s.upper() for s in LIFECYCLE)}."
            )

    def _transition(self, target: str) -> None:
        if target not in _NEXT[self.state]:
            allowed = _NEXT[self.state]
            allowed_text = " or ".join(s.upper() for s in allowed) if allowed else "nothing (it is final)"
            raise RoundStateError(
                f"{self.round_id}: {self.state.upper()} -> {target.upper()} is not a legal transition; "
                f"from {self.state.upper()} the round may only move to {allowed_text}."
            )
        self.state = target

    # --- step 5: submissions ------------------------------------------------------------

    def submit(
        self,
        miner: str,
        *,
        paths: Any,
        payload_digest: str,
        received_at: float,
    ) -> Receipt:
        """Take one upload. Returns envelope facts only -- never a correctness result.

        Four outcomes, and the boundary between them is the whole contract with a miner:

        `MALFORMED` the upload is not a submission (missing miner, no paths, no digest).
        `REFUSED`   it is a submission and names files the contract does not allow.
        `LATE`      it arrived after the admission cutoff.
        `ACCEPTED`  it now stands.

        A miner can tell a broken tarball from a rejected one, which they must be able to do
        or a packaging bug costs them the round while they rewrite a strategy that was fine.
        They cannot tell an accepted-and-correct submission from an accepted-and-wrong one,
        because nothing in this method can compute that and `Receipt` has nowhere to put it.

        ## A bad upload must not displace a good one

        Refused and malformed uploads leave the standing submission exactly as it was. The
        obvious implementation -- store first, validate after -- loses the round for a miner
        whose v1 was fine and whose v2 archive was truncated: `last one counts` would count
        nothing. The receipt reports `standing_digest` so they can see which upload is the one
        that will be graded, rather than inferring it from a status code.
        """
        self._require(OPEN, what="accept a submission")
        self._advance(received_at, what="a submission")

        standing = self._submissions.get(miner)
        problems: list[str] = []

        # Schema: is this a submission at all? Checked before the contract so a miner with a
        # broken uploader is told that, rather than being handed a list of path violations
        # derived from a field that never arrived.
        if not isinstance(miner, str) or not miner.strip():
            problems.append("no miner id: a submission nobody owns cannot be scored or replaced")
        path_list: list[str] = []
        if isinstance(paths, (list, tuple)):
            bad = [p for p in paths if not isinstance(p, str) or not p.strip()]
            if bad:
                problems.append(f"{len(bad)} path entries are not non-empty strings")
            else:
                path_list = [str(p) for p in paths]
        else:
            problems.append(f"paths must be a list of strings, got {type(paths).__name__}")
        if not path_list and not problems:
            problems.append("a submission with no files changes nothing about how the model behaves")
        if not isinstance(payload_digest, str) or not payload_digest.startswith("sha256:"):
            problems.append(
                "payload_digest must be a 'sha256:...' content address; without one the round cannot "
                "show later that it graded the bytes the miner uploaded"
            )

        if problems:
            return self._record(
                miner=miner if isinstance(miner, str) else "",
                outcome=MALFORMED,
                problems=problems,
                received_at=received_at,
                standing=standing,
                submission_digest="",
            )

        # Late before contract. A miner who missed the cutoff needs to know that first: fixing
        # a path in a submission that could never be admitted is wasted work.
        cutoff = self._frozen.deadline_used if self._frozen else self.deadline
        if received_at > cutoff:
            return self._record(
                miner=miner,
                outcome=LATE,
                problems=[
                    f"arrived at {received_at}, after the {cutoff} cutoff. Admission is decided by the "
                    "announced deadline rather than by when the validator got round to freezing, so a "
                    "validator running late does not quietly widen the window."
                ],
                received_at=received_at,
                standing=standing,
                submission_digest="",
            )

        violations = (self.contract or default_contract()).check(path_list)
        if violations:
            return self._record(
                miner=miner,
                outcome=REFUSED,
                problems=[str(v) for v in violations],
                received_at=received_at,
                standing=standing,
                submission_digest="",
            )

        revision = (standing.revision + 1) if standing else 1
        submission = Submission(
            miner=miner,
            paths=tuple(path_list),
            payload_digest=payload_digest,
            received_at=received_at,
            revision=revision,
            replaced_digest=standing.digest if standing else "",
        )
        self._submissions[miner] = submission
        if standing:
            self._replacements[miner] = self._replacements.get(miner, 0) + 1
        return self._record(
            miner=miner,
            outcome=ACCEPTED,
            problems=[],
            received_at=received_at,
            standing=submission,
            submission_digest=submission.digest,
        )

    def _record(
        self,
        *,
        miner: str,
        outcome: str,
        problems: list[str],
        received_at: float,
        standing: Submission | None,
        submission_digest: str,
    ) -> Receipt:
        """Log every attempt, accepted or not, and hand back the receipt.

        Rejected attempts are kept because the alternative is unresolvable: a miner says the
        upload went through, the validator has no record of it, and there is no way to tell a
        dropped submission from an imagined one.
        """
        receipt = Receipt(
            round_id=self.round_id,
            miner=miner,
            outcome=outcome,
            revision=standing.revision if standing else 0,
            replaced_previous=bool(standing and standing.replaced_digest) and outcome == ACCEPTED,
            replaced_digest=standing.replaced_digest if (standing and outcome == ACCEPTED) else "",
            submission_digest=submission_digest,
            standing_digest=standing.digest if standing else "",
            problems=tuple(problems),
            received_at=received_at,
        )
        self._receipts.append(receipt)
        return receipt

    @property
    def submissions(self) -> dict[str, Submission]:
        """The standing submission per miner -- the last accepted one, and only that one."""
        return dict(self._submissions)

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        """Every upload attempt in order, including the refused ones."""
        return tuple(self._receipts)

    def replacements(self, miner: str) -> int:
        """How many times this miner replaced their submission. Zero for a first upload."""
        return self._replacements.get(miner, 0)

    def receipt_for(self, miner: str) -> Receipt | None:
        """This miner's most recent receipt. Their own envelope facts, nobody else's."""
        for receipt in reversed(self._receipts):
            if receipt.miner == miner:
                return receipt
        return None

    # --- step 6: freeze -----------------------------------------------------------------

    def freeze(self, now: float, *, reason: str = "") -> FrozenAt:
        """Close the window. Idempotent from FROZEN, refused from GRADED onward.

        A second call from `FROZEN` returns the first `FrozenAt` unchanged rather than raising,
        because the caller is a retry: transports retry, and a freeze that raised on the
        second attempt would have an operator reaching for the manual path at the one moment
        the round is most sensitive. What must never happen is the retry *moving* anything --
        a second freeze that recomputed `deadline_used` from a later `now` would widen the
        admission window on a retry and admit uploads the first freeze had excluded. So the
        stored record is returned untouched.

        From `GRADED` or `SETTLED` it refuses. That is not a retry, it is a request to reopen a
        round whose verdicts are already recorded.

        Freezing before the announced deadline shortens a window miners were told the length
        of, so it requires a `reason` and is recorded as `early`. Freezing after it is normal
        and is recorded as an overrun, with `deadline_used` still the announced deadline.
        """
        if self.state == FROZEN:
            assert self._frozen is not None
            return self._frozen
        self._require(OPEN, what="freeze")
        self._advance(now, what="a freeze")

        early = now < self.deadline
        if early and not reason.strip():
            raise RoundError(
                f"{self.round_id}: freezing at {now} is before the announced deadline {self.deadline} and "
                "shortens a window whose length miners were told. Refused without a reason to record: a "
                "shortened round that looks like a normal one is indistinguishable afterwards from one "
                "that ran its full length."
            )
        deadline_used = now if early else self.deadline

        # Sealed by content, so a submission inserted after the freeze is detectable rather
        # than deniable. Without this, "the validator added a favourite's late upload" and "the
        # validator did not" produce identical records.
        sealed = {m: s.digest for m, s in sorted(self._submissions.items())}
        seal_digest = digest_mapping({"round_id": self.round_id, "submissions": sealed})

        self._frozen = FrozenAt(
            frozen_at=now,
            deadline_used=deadline_used,
            scheduled_deadline=self.deadline,
            early=early,
            reason=reason.strip(),
            sealed_submissions=len(sealed),
            seal_digest=seal_digest,
        )
        self._token = FreezeToken(
            round_id=self.round_id,
            frozen_at=now,
            deadline_used=deadline_used,
            seal_digest=seal_digest,
        )
        self._transition(FROZEN)
        return self._frozen

    @property
    def frozen_at(self) -> FrozenAt | None:
        return self._frozen

    def token(self) -> FreezeToken:
        """The freeze token, for a grader. Refused before the freeze because none exists."""
        if self._token is None:
            raise RoundStateError(
                f"{self.round_id}: no freeze token while the round is {self.state.upper()}. The token is "
                "minted by freeze() and is the only way to construct a Verdict, which is what keeps a "
                "correctness result from existing during an open round."
            )
        return self._token

    # --- grading ------------------------------------------------------------------------

    def record_verdict(
        self,
        token: FreezeToken,
        miner: str,
        *,
        passed: bool,
        notes: str = "",
    ) -> Verdict:
        """Record one miner's withheld-check result. FROZEN only, one per miner.

        The token is compared by identity, not equality. A token rebuilt with the right field
        values is not the token this round minted, so a caller cannot manufacture the
        capability from the round's public record.
        """
        self._require(FROZEN, what="record a verdict")
        if token is not self._token:
            raise RoundStateError(
                f"{self.round_id}: that is not this round's freeze token. Compared by identity rather "
                "than by value, because the token's field values are all in the round's public record "
                "and a value comparison would let anyone holding that record mint the capability."
            )
        if miner not in self._submissions:
            raise RoundError(
                f"{self.round_id}: {miner!r} has no standing submission, so there is nothing to grade. A "
                "verdict for a miner who never submitted inflates the denominator of every rate the "
                "round reports."
            )
        if miner in self._verdicts:
            raise RoundError(
                f"{self.round_id}: {miner!r} already has a verdict. A regrade must be explicit -- "
                "silently overwriting means the last grader to run wins and two graders disagreeing "
                "leaves no trace."
            )
        verdict = Verdict(
            token=token,
            miner=miner,
            submission_digest=self._submissions[miner].digest,
            passed=passed,
            notes=notes,
        )
        self._verdicts[miner] = verdict
        return verdict

    @property
    def verdicts(self) -> dict[str, Verdict]:
        """Recorded verdicts as objects. In-process only; never a serialisable payload.

        Refused before the freeze. Belt and braces rather than the actual defence: during an
        open round this dict is necessarily empty, because the only function that fills it
        needs a token that only `freeze` mints.
        """
        if self.state == OPEN:
            raise RoundStateError(
                f"{self.round_id}: verdicts are not readable while the round is OPEN. Nothing has been "
                "graded, and a caller expecting an empty dict here is a caller about to publish one."
            )
        return dict(self._verdicts)

    def grade(self, now: float) -> None:
        """FROZEN -> GRADED. Refuses while any standing submission is unscored.

        A round graded with a miner missing publishes a leaderboard whose denominator is wrong
        and whose omission looks exactly like a miner who did not submit.
        """
        self._require(FROZEN, what="grade")
        self._advance(now, what="grading")
        missing = sorted(set(self._submissions) - set(self._verdicts))
        if missing:
            raise RoundError(
                f"{self.round_id}: {len(missing)} standing submissions have no verdict ({missing[:5]}). A "
                "round graded with a miner missing reports a rate over the wrong denominator, and the "
                "omission is indistinguishable from a miner who never submitted."
            )
        self._graded_at = now
        self._transition(GRADED)

    def settle(self, now: float) -> None:
        """GRADED -> SETTLED. The point at which the commitment may be opened."""
        self._require(GRADED, what="settle")
        self._advance(now, what="settling")
        self._settled_at = now
        self._transition(SETTLED)

    # --- the reveal ---------------------------------------------------------------------

    def reveal(self, master_salt: str) -> Reveal:
        """Open this round's commitment. SETTLED only.

        Gated on SETTLED rather than on FROZEN because the salt is what makes a *graded*
        round auditable, and a salt released before grading hands the withheld check to
        whoever is still holding a submission -- during the one interval when the validator
        might still accept a resubmission after an operational retry. See `Reveal` for why
        opening one task leaves every other commitment sealed, and what a shared salt would
        have done instead.
        """
        self._require(SETTLED, what="reveal the per-task salt")
        commitment = self.challenge.to_record()["withheld"]["hidden_verify_commitment"]
        if not commitment:
            raise RoundError(
                f"{self.round_id}: this challenge published no withheld-check commitment, so there is "
                "nothing to open. Handing back a salt anyway would let a reader believe an audit "
                "happened when no check was ever committed to."
            )
        assert self._settled_at is not None
        return Reveal(
            round_id=self.round_id,
            task_id=self.task_id,
            salt=derive_task_salt(master_salt, self.task_id),
            commitment=commitment,
            settled_at=self._settled_at,
        )

    # --- payloads -----------------------------------------------------------------------

    def _metadata(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "round_id": self.round_id,
            "task_id": self.task_id,
            "state": self.state,
            "opened_at": self.opened_at,
            "deadline": self.deadline,
            "submissions": len(self._submissions),
            "replacements": sum(self._replacements.values()),
            "withheld_check_committed": self.withheld_check_committed,
            # Stated in the payload the way `hermes.challenge` states
            # `acceptance_thresholds_included`: a reader should not have to infer a guarantee
            # from the absence of a field.
            "no_score_before_freeze": True,
            "frozen": self._frozen.to_record() if self._frozen else None,
            "reveal_available": self.state == SETTLED and self.withheld_check_committed,
        }
        return screen_public_payload(record, allowed=PUBLIC_VIEW_FIELDS, where=f"round {self.round_id}")

    def public_view(self) -> dict[str, Any]:
        """Everything a miner may see, at the current state.

        Assembled in three parts, and the seams are where a leak would hide, so they are named
        here rather than left to a reader to spot.

        `challenge` is `Challenge.to_record()` verbatim -- the packet already carries only the
        public half plus `hidden_verify_commitment`, and reimplementing that split here would
        give the repo two places to keep the withheld list correct. It goes through
        `refuse_withheld_body` rather than the field allowlist, because its own keys
        legitimately include the baseline's `pass_rate`: that is the frozen model's published
        performance, which is what *defines* the challenge, not a miner's score.

        `verdicts` appears only from GRADED, and is assembled outside the screen for the
        obvious reason that it is verdict-shaped.

        Everything else is round metadata and goes through both nets.
        """
        view: dict[str, Any] = {
            "challenge": refuse_withheld_body(
                self.challenge.to_record(), where=f"round {self.round_id} challenge packet"
            ),
            **self._metadata(),
        }
        if self.state in (GRADED, SETTLED):
            view["verdicts"] = [v.to_record() for v in sorted(self._verdicts.values(), key=lambda v: v.miner)]
        return view

    def to_record(self) -> dict[str, Any]:
        """The validator's ledger: the public view plus who submitted what.

        Withholds verdicts before GRADED exactly as `public_view` does. Making only the
        miner-facing payload safe would leave this one safe by operator discipline, and a
        ledger is precisely the file that gets copied into a status page by somebody who did
        not read this docstring.
        """
        extra: dict[str, Any] = {
            "standing_submissions": [
                screen_public_payload(
                    {
                        "miner": s.miner,
                        "revision": s.revision,
                        "submission_digest": s.digest,
                        "payload_digest": s.payload_digest,
                        "paths": list(s.paths),
                        "received_at": s.received_at,
                        "replaced_digest": s.replaced_digest,
                        "replacements": self._replacements.get(s.miner, 0),
                    },
                    allowed=SUBMISSION_RECORD_FIELDS,
                    where=f"ledger entry {s.miner}",
                )
                for s in sorted(self._submissions.values(), key=lambda s: s.miner)
            ],
            # Each receipt screens itself against RECEIPT_FIELDS on the way out.
            "attempts": [r.to_record() for r in self._receipts],
            "graded_at": self._graded_at,
            "settled_at": self._settled_at,
        }
        if self.state not in (GRADED, SETTLED):
            # A count is not a score. Published so a reader can tell "grading has not started"
            # from "grading found nothing", which otherwise look identical from outside.
            extra["verdicts_recorded"] = len(self._verdicts)
            extra["verdicts_withheld_until"] = GRADED
        screen_public_payload(extra, allowed=LEDGER_FIELDS, where=f"ledger {self.round_id}")
        return {**self.public_view(), **extra}


def open_round(
    challenge: Challenge,
    *,
    round_id: str,
    opened_at: float,
    deadline: float,
    contract: Any = None,
) -> RoundWindow:
    """Publish a round over one challenge. Refuses a window nobody can submit to.

    Mirrors `hermes.challenge.open_challenge` deliberately: the refusals live at the moment of
    publication, because a round with a deadline in the past is discovered by the miners who
    could not submit to it, and by then the epoch is spent.
    """
    if not round_id.strip():
        raise RoundError("a round needs an id; submissions are addressed to it and receipts quote it")
    if deadline <= opened_at:
        raise RoundError(
            f"{round_id}: deadline {deadline} is not after the open time {opened_at}, so every submission "
            "would be LATE. A round nobody can enter still consumes an epoch of the challenge."
        )
    return RoundWindow(
        challenge=challenge,
        round_id=round_id.strip(),
        opened_at=opened_at,
        deadline=deadline,
        contract=contract,
    )


# --- the registry ----------------------------------------------------------------------------


@dataclass
class Registry:
    """The rounds this validator knows about: in memory, with a JSON snapshot for audit.

    Thin on purpose. It exists to enforce the two invariants that span rounds and cannot be
    checked from inside one, and to write a record a stranger can read.
    """

    rounds: dict[str, RoundWindow] = field(default_factory=dict)

    def add(self, round_: RoundWindow) -> RoundWindow:
        """Register a round. Refuses a duplicate id and a second open round on one task.

        The task rule is the one that matters. Two open rounds over the same challenge means
        two windows, two sets of submissions and two deadlines over one task -- and when they
        are merged there is no principled answer to which submission was a miner's last, which
        is the single fact the whole submission path is built to establish.
        """
        if round_.round_id in self.rounds:
            raise RoundError(
                f"{round_.round_id} is already registered. Reusing a round id makes two sets of receipts "
                "quote the same round, and a miner's proof of submission stops identifying anything."
            )
        clash = self.open_for_task(round_.task_id)
        if clash is not None and round_.state == OPEN:
            raise RoundError(
                f"{round_.task_id} already has an open round ({clash.round_id}). Two open windows over one "
                "task give a miner two 'last' submissions under two deadlines, and merging them has no "
                "principled answer -- which is the one fact the submission path exists to pin down."
            )
        self.rounds[round_.round_id] = round_
        return round_

    def open_round(
        self,
        challenge: Challenge,
        *,
        round_id: str,
        opened_at: float,
        deadline: float,
        contract: Any = None,
    ) -> RoundWindow:
        """`open_round` plus registration, which is the ordinary path."""
        return self.add(
            open_round(
                challenge,
                round_id=round_id,
                opened_at=opened_at,
                deadline=deadline,
                contract=contract,
            )
        )

    def get(self, round_id: str) -> RoundWindow:
        try:
            return self.rounds[round_id]
        except KeyError:
            raise RoundError(f"no round {round_id!r} is registered") from None

    def open_for_task(self, task_id: str) -> RoundWindow | None:
        """The open round over this task, if there is one."""
        return next((r for r in self.rounds.values() if r.task_id == task_id and r.state == OPEN), None)

    def snapshot(self) -> dict[str, Any]:
        """Every round as a record. Safe to publish at any state -- see `RoundWindow.to_record`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "rounds": [self.rounds[k].to_record() for k in sorted(self.rounds)],
        }

    def save(self, path: Any) -> Path:
        """Write the snapshot as JSON.

        Sorted keys and a trailing newline, so two validators with the same rounds produce
        byte-identical files and a diff between two snapshots is a diff in the rounds rather
        than in dict ordering.
        """
        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @staticmethod
    def read_snapshot(path: Any) -> dict[str, Any]:
        """Read a snapshot back for inspection. Deliberately does not rehydrate live rounds.

        A `RoundWindow` read from disk would be a state machine whose history is a claim in a file,
        and the first use of it is the dangerous one: a validator that crashed mid-round
        reloads, gets a writable `OPEN` round back, and reopens a window that had already
        closed -- admitting submissions from miners who watched the freeze happen. Resuming a
        round is a real requirement and it needs a decision about what the restarted validator
        is allowed to believe; it is not something a JSON loader should grant by accident.
        """
        return json.loads(Path(str(path)).read_text(encoding="utf-8"))


__all__ = [
    "ACCEPTED",
    "FROZEN",
    "GRADED",
    "LATE",
    "LEDGER_FIELDS",
    "LIFECYCLE",
    "MALFORMED",
    "OPEN",
    "PUBLIC_VIEW_FIELDS",
    "RECEIPT_FIELDS",
    "REFUSED",
    "SCHEMA_VERSION",
    "SETTLED",
    "SUBMISSION_OUTCOMES",
    "SUBMISSION_RECORD_FIELDS",
    "VERDICT_WORDS",
    "WITHHELD_KEYS",
    "FreezeToken",
    "FrozenAt",
    "Receipt",
    "Registry",
    "Reveal",
    "RoundWindow",
    "RoundError",
    "RoundStateError",
    "ScoreLeakError",
    "Submission",
    "Verdict",
    "default_contract",
    "open_round",
    "refuse_withheld_body",
    "screen_public_payload",
]

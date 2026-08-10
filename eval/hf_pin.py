"""One place that decides whether a Hugging Face reference names fixed bytes.

Four call sites already download from the Hub -- `eval/verify.py`, `eval/mix_registry.py`,
`eval/dataset_verify.py`, `eval/training_track_gate.py` -- and not one of them passed
`revision=`. Every one resolved `main` at download time.

Be precise about what that did and did not break. Two of the four pass a `claimed_sha256`
and compare the bytes they got, so a miner could not swap content undetected: the digest
catches it. What breaks is *retrieval*. Re-fetch a merged row a month later and you get a
digest mismatch rather than the artifact, because `main` moved. So of the questions an
accepted row is supposed to answer, the one that had no answer was:

    Can the artifacts be retrieved unchanged?

A revision fixes that, and it is strictly additive: the digest check still runs and still
catches a mismatch. The revision means there is something to check against a year from now.

**A branch name is not a pin.** `main`, `master`, `latest`, `dev`, a tag -- all of them are
labels a publisher can move after the row was accepted, which is exactly the property being
excluded. Only a full 40-character commit SHA is accepted. Not a short SHA either: Hub
short SHAs are ambiguous by construction and the resolution rule can change under a repo
that grows.
"""

from __future__ import annotations

import re
from typing import Any

# A full Hub commit id. Deliberately not `[0-9a-f]{7,40}` -- an abbreviation is a prefix
# search, and a prefix that is unique today can collide tomorrow in a repo that keeps
# growing. Forty or nothing.
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Names that look like a pin to a reader and are not. Listed explicitly so the refusal can
# say which one was used, rather than "does not match a regex".
MOVABLE = frozenset({"main", "master", "latest", "head", "dev", "develop", "stable", "default"})


class HFPinError(ValueError):
    """A Hugging Face reference does not name fixed bytes."""


def check_revision(revision: Any, *, field: str = "hf_revision") -> list[str]:
    """Problems with a claimed revision, worst first. Empty means it pins bytes."""
    if revision is None or not str(revision).strip():
        return [
            f"{field} is missing; without it the download resolves whatever the branch points at "
            "when it runs, so the accepted artifact cannot be retrieved unchanged later"
        ]
    value = str(revision).strip()
    if value.lower() in MOVABLE:
        return [
            f"{field}={value!r} is a movable ref, not a revision. The publisher can advance it "
            "after this row is accepted, which is the exact property a pin exists to exclude"
        ]
    if not COMMIT_SHA.match(value):
        # A short sha gets its own sentence: it is the near-miss most likely to be offered in
        # good faith, and "does not match" would not explain why 40 characters are required.
        if re.match(r"^[0-9a-f]{7,39}$", value):
            return [
                f"{field}={value!r} is an abbreviated commit id. Abbreviations resolve by prefix "
                "search, and a prefix that is unique today can collide in a repo that keeps "
                "growing, so the full 40 characters are required"
            ]
        return [f"{field}={value!r} is not a 40-character commit sha"]
    return []


def require_revision(revision: Any, *, field: str = "hf_revision") -> str:
    """The revision, or a refusal. For call sites that cannot carry on without one."""
    problems = check_revision(revision, field=field)
    if problems:
        raise HFPinError(problems[0])
    return str(revision).strip()


def pinned_download(snapshot_download: Any, *, revision: Any, field: str = "hf_revision", **kwargs: Any) -> Any:
    """`snapshot_download` with the revision checked first.

    A thin wrapper rather than a convention, because a convention is what produced four call
    sites that each forgot the same argument. Passing the downloader in keeps this module
    free of a `huggingface_hub` import, which matters: it is imported lazily at every call
    site precisely so the package stays optional.
    """
    return snapshot_download(revision=require_revision(revision, field=field), **kwargs)


def download_with_optional_pin(snapshot_download: Any, *, revision: Any, **kwargs: Any) -> Any:
    """`snapshot_download` with a revision when there is one, and without when there is not.

    For the paths that cannot require a pin yet -- the training track and the dataset
    registry both have live rows that predate the field. Branching rather than passing
    `revision=None` is deliberate twice over: `None` is equivalent to omitting it for the
    real client but not for every wrapper standing in for it, and a `**{...}` spread defeats
    the type checker, which then reads every keyword as possibly a `str`.
    """
    if revision:
        return snapshot_download(revision=require_revision(revision), **kwargs)
    return snapshot_download(**kwargs)

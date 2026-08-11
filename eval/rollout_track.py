"""The rollout track: accepting a miner's dataset by pull request, or refusing it with a reason.

A miner runs a seeded round, generates rollouts against the pinned teachers, publishes the
exports to Hugging Face, and opens a pull request appending one line to
`datasets/rollouts.jsonl`. This decides whether that line may merge.

The shape is deliberately the training track's, because that flow is live and its security
spine is the part worth copying: the workflow checks out the *trusted base*, fetches the PR
head as a git object it never executes, and the submission is data-only so auto-merge can
never carry code. Everything below assumes that arrangement and checks the payload.

Five things a submission has to survive, and each catches a different way of being wrong:

**Scope.** The round announcement says which tasks this miner was assigned. Work on someone
else's tasks is not fraud, it is duplicated effort -- the thing seeding exists to prevent --
and it is rejected by recomputation rather than by consulting a validator, so a rejection is
arguable rather than arbitrary.

**Binding.** The manifest digest a submission claims must be the digest its own rows
produce. Without this a miner could publish good rows, cite a receipt from a different
batch, and have both check out individually.

**Attestation.** The run must have happened on an Intel TDX CC node with an NVIDIA CC GPU
-- a Targon RTX PRO 6000 Blackwell -- and the signed evidence must commit to *these*
exports. Verified against NVIDIA's NRAS JWKS and Intel's DCAP/PCS: public roots the
validator fetches, so a miner-published `attestation.json` is safe to read and nothing
trusts its own `passed` flag or `claims` sidecar.

This replaced a Cathedral receipt, and the reason is the one `check_novelty` states below:
a receipt proved that *a* sealed check ran, but nothing a reviewer could recompute proved
*which* manifest it checked, because the platform's input digest was unsigned and its
preimage unpublished. So binding had to be reconstructed sideways -- local recomputation
plus one-receipt-one-submission. Here the export's `claim_sha256` is the nonce inside the
NRAS-signed token and the REPORTDATA inside the Intel-signed quote, so "which export" is
recomputed from bytes rather than assumed. One-per-receipt stays as defence in depth; it is
no longer the thing holding the binding up.

It also removed the operational blocker: a Cathedral receipt needed its signing keys pinned
here, obtained out of band, and until they were the gate rejected every submission on
attestation. NVIDIA's JWKS and Intel's PCS are public. There is no key to obtain.

**Substance.** A manifest that digests cleanly can still describe nothing: every task
marked incomparable, no runs, no winner. That is not a lie a miner must be caught in, it is
a manifest admitting in its own fields that no work happened -- and it was accepted until
`check_manifest_work` existed. Teachers must be ones the registry declares, and the
published rows must exist and match their digests, which means fetching them.

**Novelty.** Rows whose trajectories already exist in the registry earn nothing. Paying per
submitted row rather than per *surviving* row is what makes resubmission profitable, and a
corpus that grows without teaching anything new is the failure the whole seeding design is
built against.

**Append-only.** The registry is history. A submission that rewrites an earlier line is
rejected even when the rewrite is an improvement, because a reviewer cannot tell those
apart at merge time and the ledger's value is that nobody can.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.hf_pin import check_revision
from hermes.harness import digest_mapping
from hermes.seed import CLOSED, COMMITTED, Round, SeedError

ROLLOUT_REGISTRY = Path("datasets/rollouts.jsonl")
ROUNDS_DIR = Path("datasets/rounds")

SCHEMA_VERSION = 1

REQUIRED_FIELDS = (
    "schema_version",
    "round_id",
    "miner_id",
    "hf_url",
    "hf_revision",
    "manifest_digest",
    "harness_digest",
    "receipt_id",
    "task_ids",
    "export_digests",
)

# A rollout submission may only touch data. Auto-merge that can carry executable changes is
# not auto-merge, it is remote code execution with a review step somebody will skip.
ALLOWED_PATHS = (ROLLOUT_REGISTRY.as_posix(),)

# Measurements of guest images we approve, as hex MRTD values.
#
# Empty on purpose rather than absent: the expected MRTD for a reproducible guest image has to be
# obtained by building that image, not by reading it off a submission, and an allowlist populated
# from anything a submitter supplies is not an allowlist. While it is empty the measured-VM check
# reports a caveat; the moment it has an entry, an unmeasured guest is refused.
APPROVED_GUEST_MEASUREMENTS: tuple[str, ...] = ()

REJECT = "rollout:REJECT"
ACCEPT = "rollout:ACCEPT"


class SubmissionError(ValueError):
    """A submission is malformed."""


@dataclass(frozen=True)
class Submission:
    """One miner's claim on one round."""

    round_id: str
    miner_id: str
    hf_url: str
    hf_revision: str
    manifest_digest: str
    harness_digest: str
    receipt_id: str
    task_ids: tuple[str, ...]
    export_digests: dict[str, str]
    rows: dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "hf_url": self.hf_url,
            "hf_revision": self.hf_revision,
            "manifest_digest": self.manifest_digest,
            "harness_digest": self.harness_digest,
            "receipt_id": self.receipt_id,
            "task_ids": sorted(self.task_ids),
            "export_digests": dict(sorted(self.export_digests.items())),
            "rows": dict(sorted(self.rows.items())),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Submission:
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise SubmissionError(f"submission is missing {', '.join(missing)}")
        return cls(
            round_id=str(record["round_id"]),
            miner_id=str(record["miner_id"]),
            hf_url=str(record["hf_url"]),
            hf_revision=str(record.get("hf_revision") or ""),
            manifest_digest=str(record["manifest_digest"]),
            harness_digest=str(record["harness_digest"]),
            receipt_id=str(record["receipt_id"]),
            task_ids=tuple(str(t) for t in record["task_ids"]),
            export_digests={str(k): str(v) for k, v in (record["export_digests"] or {}).items()},
            rows=_int_map(record.get("rows")),
            schema_version=_as_int(record.get("schema_version", SCHEMA_VERSION), "schema_version"),
        )


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(f"{name} must be an integer, got {value!r}") from exc


def _int_map(value: Any) -> dict[str, int]:
    return {str(k): _as_int(v, f"rows[{k!r}]") for k, v in (value or {}).items()}


def check_shape(record: dict[str, Any]) -> list[str]:
    """Structural problems with a submitted line."""
    issues: list[str] = []
    if not isinstance(record, dict):
        return ["submission must be a JSON object"]
    for name in REQUIRED_FIELDS:
        if not record.get(name):
            issues.append(f"missing required field: {name}")
    if issues:
        return issues
    version = record["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        # A future version may mean anything; this gate implements one contract and should
        # say so rather than validate a document by rules it was not written under.
        issues.append(f"schema_version must be {SCHEMA_VERSION}; this gate does not implement {version!r}")
    for name in ("manifest_digest", "harness_digest"):
        value = str(record[name])
        if not value.startswith("sha256:") or len(value) != 71:
            issues.append(f"{name} must be a sha256: digest with 64 hex characters")
    # A URL names a repo; only a revision names its bytes. Without one the verified
    # snapshot is whatever the branch pointed at during the gate run, and re-fetching the
    # merged row later returns something else -- or nothing that matches.
    issues.extend(check_revision(record.get("hf_revision")))
    url = str(record["hf_url"])
    if any(ch < " " or ch == "\x7f" for ch in url):
        # A newline in a URL splits it for anything that parses line-wise downstream, and
        # the visible prefix stays innocent.
        issues.append("hf_url contains a control character")
    if not url.startswith("https://huggingface.co/"):
        # A submission that points anywhere else cannot be fetched by the same code path
        # every other submission uses, and a bespoke fetch is a bespoke trust decision.
        issues.append("hf_url must be a https://huggingface.co/ URL")
    if not isinstance(record.get("task_ids"), list) or not record["task_ids"]:
        issues.append("task_ids must be a non-empty list")
    elif len(set(record["task_ids"])) != len(record["task_ids"]):
        repeated = sorted({t for t in record["task_ids"] if record["task_ids"].count(t) > 1})
        issues.append(f"duplicate task id in task_ids {repeated}; one assignment cannot be claimed twice")
    if not isinstance(record.get("export_digests"), dict) or "sft" not in (record.get("export_digests") or {}):
        issues.append("export_digests must name at least the sft export")
    else:
        malformed = sorted(
            k for k, v in record["export_digests"].items() if not str(v).startswith("sha256:") or len(str(v)) != 71
        )
        if malformed:
            issues.append(f"export_digests values must be sha256: digests with 64 hex characters: {malformed}")
    for key in ("rows", "schema_version"):
        value = record.get(key)
        if key == "rows" and value is not None and not isinstance(value, dict):
            issues.append("rows must be an object of counts")
    return issues


def check_scope(record: dict[str, Any], round_record: dict[str, Any]) -> list[str]:
    """Whether this miner was assigned these tasks, by recomputation."""
    try:
        round_ = Round.from_record(round_record)
    except SeedError as exc:
        return [f"round announcement is unusable: {exc}"]
    if round_.state == CLOSED:
        return [f"round {round_.round_id} is closed; it no longer owes anyone work"]
    if round_.state == COMMITTED:
        # The seed is out but the round has not opened, so nobody was owed this work yet.
        return [f"round {round_.round_id} is committed but not open; there was no window to work in"]
    if record.get("round_id") != round_.round_id:
        return [f"submission names round {record.get('round_id')!r}, announcement is {round_.round_id!r}"]
    miner = str(record.get("miner_id") or "")
    if miner not in round_.miner_ids:
        return [f"{miner!r} is not registered in round {round_.round_id}"]
    out_of_scope = [t for t in record.get("task_ids") or () if t not in round_.task_ids or not round_.owns(miner, t)]
    if out_of_scope:
        return [
            f"tasks {sorted(out_of_scope)[:5]} were not assigned to {miner!r}; work on another miner's "
            "tasks is duplicated effort, which is what the seeded assignment exists to prevent"
        ]
    return []


def check_binding(record: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Whether the manifest a submission cites is the one its own rows produce."""
    issues: list[str] = []
    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    recomputed = digest_mapping(body)
    if manifest.get("manifest_digest") != recomputed:
        issues.append(
            f"manifest is internally inconsistent: it claims {manifest.get('manifest_digest')} but its "
            f"own body digests to {recomputed}"
        )
    if record.get("manifest_digest") != manifest.get("manifest_digest"):
        # Without this a miner can publish good rows, cite a receipt from a different
        # batch, and have both check out on their own.
        issues.append("submission's manifest_digest does not match the published manifest")
    if record.get("harness_digest") != manifest.get("harness_digest"):
        issues.append("submission's harness_digest does not match the manifest's")
    if record.get("round_id") != manifest.get("round_id") or record.get("miner_id") != manifest.get("miner_id"):
        issues.append("submission and manifest disagree about the round or the miner")
    if manifest.get("exports") and record.get("export_digests") != manifest.get("exports"):
        # The manifest is what the sealed check saw. Digests outside it are a claim nobody
        # attested, so the two must agree or the attestation covers different files.
        issues.append("submission's export_digests differ from the ones inside the attested manifest")
    manifest_ids = [t.get("task_id") for t in manifest.get("tasks") or ()]
    if len(set(manifest_ids)) != len(manifest_ids):
        repeated = sorted({i for i in manifest_ids if manifest_ids.count(i) > 1})
        issues.append(f"manifest repeats task ids {repeated}; one assignment cannot carry two outcomes")
    submitted = set(record.get("task_ids") or ())
    in_manifest = set(manifest_ids)
    if submitted != in_manifest:
        issues.append(
            f"task_ids do not match the manifest: submitted {sorted(submitted)[:5]}, manifest has {sorted(in_manifest)[:5]}"
        )
    return issues


def check_manifest_work(manifest: dict[str, Any]) -> list[str]:
    """Whether the manifest describes work, rather than merely digesting cleanly.

    `check_binding` proves the manifest is internally consistent and that the submission
    cites it. Both documents come from the miner, so on their own that is a tautology: a
    manifest whose every task says `comparable: false, runs: [], winner: null` is perfectly
    consistent and admits, in its own fields, that nothing happened. It was accepted.

    So the content is checked against what a tournament actually is. Teachers must be ones
    the registry declares -- an invented teacher id is not a model that ran -- every
    comparable task needs at least two of them with well-formed trajectory digests, and the
    winner must be among the teachers that ran it.
    """
    from hermes.teachers import REGISTRY

    issues: list[str] = []
    comparable = 0
    for task in manifest.get("tasks") or ():
        runs = task.get("runs") or []
        ran = {r.get("teacher_id") for r in runs}
        if not task.get("comparable"):
            continue
        comparable += 1
        task_id = task.get("task_id")
        unknown = sorted(t for t in ran if t not in REGISTRY)
        if unknown:
            issues.append(f"task {task_id!r} names teachers not in the registry: {unknown}")
        if len(ran) < 2:
            issues.append(f"task {task_id!r} is marked comparable with {len(ran)} teacher(s); two is the minimum")
        malformed = [
            r.get("teacher_id")
            for r in runs
            if not str(r.get("trajectory_sha256", "")).startswith("sha256:")
            or len(str(r.get("trajectory_sha256", ""))) != 71
        ]
        if malformed:
            issues.append(f"task {task_id!r} has runs with no usable trajectory digest: {sorted(malformed)}")
        if task.get("winner") is not None and task.get("winner") not in ran:
            issues.append(f"task {task_id!r} names a winner that did not run it")
    if not comparable:
        issues.append(
            "no task in the manifest had two teachers produce a candidate; it contains no comparisons, "
            "so nothing in it is a tournament result"
        )
    return issues


def check_exports(record: dict[str, Any], manifest: dict[str, Any], export_dir: Path | None) -> list[str]:
    """Whether the published rows exist and are the ones claimed.

    **Refuses when the exports were not fetched.** Skipping would mean accepting a
    submission whose published rows nobody read, and a miner can publish nothing at all --
    a nonexistent repo, invented digests, and a hundred thousand claimed rows were accepted
    before this existed. A check that cannot run must refuse, not wave through.

    Row counts are compared, not trusted, and every published trajectory must appear in the
    manifest: rows citing trajectories the attested manifest never mentions are rows the
    sealed check never saw.
    """
    if export_dir is None:
        return [
            "exports were not fetched, so the published rows were never read; a submission whose "
            "content nobody looked at cannot be accepted"
        ]
    from hermes.arena import export_digests as recompute

    issues: list[str] = []
    recomputed = recompute(export_dir)
    claimed = record.get("export_digests") or {}
    if recomputed != claimed:
        issues.append(f"export_digests do not match the published files: recomputed {recomputed}, claimed {claimed}")

    actual: dict[str, int] = {}
    for name in ("sft", "dpo", "router"):
        path = export_dir / f"{name}.jsonl"
        if path.is_file():
            actual[name] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    stated = {k: int(v) for k, v in (record.get("rows") or {}).items()}
    if stated and stated != actual:
        issues.append(f"rows mismatch: claimed {stated}, the files contain {actual}")

    cited = {
        r.get("trajectory_sha256")
        for t in manifest.get("tasks") or ()
        for r in t.get("runs") or ()
        if r.get("trajectory_sha256")
    }
    sft = export_dir / "sft.jsonl"
    if sft.is_file():
        published = set()
        for line in sft.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    published.add(json.loads(line).get("trajectory_sha256"))
                except json.JSONDecodeError:
                    issues.append("sft.jsonl contains a line that is not JSON")
                    break
        stray = sorted(x for x in published - cited if x)
        if stray:
            issues.append(f"published rows cite trajectories absent from the attested manifest: {stray[:3]}")
    return issues


def check_attestation(
    record: dict[str, Any], attestation: dict[str, Any] | None, export_dir: Path | None
) -> tuple[list[str], list[str]]:
    """Whether the run was attested on a CC node, bound to *this* export.

    Verified against NVIDIA's NRAS JWKS and Intel's DCAP/PCS -- public roots, fetched by
    the validator, not supplied by the miner. That is what makes a miner-published
    attestation safe to read: nothing here trusts the JSON's own `passed` flag or its
    `claims` sidecar, both of which the submitter can write. The two facts that decide the
    outcome, `eat_nonce` and `hwmodel`, are taken only from JWKS-verified tokens.

    **This replaces a Cathedral receipt, and the reason is written a few lines below in
    `check_novelty`:** a receipt proved that *a* sealed check ran and passed, but "nothing
    a reviewer can recompute proves *which* manifest it checked, because the input digest
    the platform records is unsigned and its preimage is unpublished." That is the whole
    weakness. Here the export's own `claim_sha256` is the nonce inside the NRAS-signed
    token and the REPORTDATA inside the Intel-signed TDX quote, so "which export" is not a
    claim to be trusted -- it is recomputed from the bytes and compared.

    It also removes the operational blocker. A Cathedral receipt needed its signing keys
    pinned in this repository, obtained out of band, and until they were the gate rejected
    every submission on attestation. NVIDIA's JWKS and Intel's PCS are public and
    well-known; there is no key to obtain.
    """
    if attestation is None:
        return (
            [
                "no attestation.json published beside the exports; a rollout must be generated on an "
                "Intel TDX CC node with an NVIDIA CC GPU and attested, or its provenance is a claim"
            ],
            [],
        )
    if export_dir is None:
        return (["exports were not fetched, so no claim digest exists to bind the attestation to"], [])

    from eval.verify import (
        check_claim_binding,
        check_gpu_signature,
        check_tdx_binding,
        check_tdx_measurement,
        check_tdx_signature,
        signed_attestation_claims,
    )

    issues: list[str] = []
    # Caveats are not issues. A caveat says the submission is acceptable *and* that something it
    # would be natural to assume has not been established; folding the two together would either
    # reject every submission or hide the gap.
    caveats: list[str] = []

    gpu_sig = check_gpu_signature(attestation)
    if not gpu_sig or not gpu_sig.get("verified"):
        issues.append(
            f"GPU attestation token does not verify against NVIDIA's JWKS: {(gpu_sig or {}).get('reason', 'no token')}"
        )

    # Binding, not merely presence. An attestation that verifies but commits to a different
    # bundle is one honestly-earned green run cited beside any number of unchecked batches --
    # the exact reuse `check_novelty` had to forbid by hand when the receipt could not bind.
    if check_claim_binding(export_dir, attestation, gpu_sig=gpu_sig) is not True:
        issues.append("the signed GPU nonce does not commit to these exports; the attestation is for other work")

    tdx_sig = check_tdx_signature(attestation)
    if tdx_sig is None:
        issues.append("no Intel TDX quote; the GPU was attested but the VM it ran in was not")
    elif not tdx_sig.get("verified"):
        issues.append(f"TDX quote does not verify: {tdx_sig.get('reason', 'unknown')}")
    elif check_tdx_binding(export_dir, attestation) is not True:
        issues.append("the TDX quote's REPORTDATA does not commit to these exports")

    # The measured-VM leg. `check_tdx_measurement` returns None when nothing is pinned, and the
    # first version of this tested `if measured is False` -- which cannot fire on None, so with no
    # allowlist it was a check incapable of failing that read as one that passed.
    #
    # Enforced when an allowlist exists, reported when it does not, and the two are different
    # states rather than one silence. `APPROVED_GUEST_MEASUREMENTS` is empty today: pinning an MRTD
    # requires building a reproducible guest image and reading the measurement off *that*, not off
    # a submission. Until it is populated a quote proves a genuine confidential VM ran and
    # committed to this bundle, and the submission carries that caveat instead of implying more.
    measured, reason = check_tdx_measurement(attestation, APPROVED_GUEST_MEASUREMENTS)
    if measured is False:
        issues.append(f"TDX measured-VM check failed: {reason}")
    elif measured is None:
        if APPROVED_GUEST_MEASUREMENTS:
            # We asked for the check and could not get an answer. With an allowlist configured,
            # "unknown" is a refusal: the whole point of pinning is that an unmeasured guest stops
            # being acceptable.
            issues.append(
                "the TDX quote carries no guest measurement to compare against the approved list, "
                f"so it cannot be shown to have booted an approved image: {reason}"
            )
        else:
            caveats.append(
                "no approved guest measurement is pinned, so this quote proves a genuine "
                "confidential VM ran and committed to these exports, not that it ran an image we "
                "approved"
            )

    claims = signed_attestation_claims(attestation, gpu_sig=gpu_sig)
    if claims is None:
        issues.append("no JWKS-verified claims, so the GPU model cannot be corroborated")
    else:
        from eval.training_gpus import accepted_training_gpu_label, claimed_hwmodels, is_accepted_training_gpu

        models = claimed_hwmodels(claims)
        if not any(is_accepted_training_gpu(m) for m in models):
            issues.append(f"attested GPU {models or ['unknown']} is not a {accepted_training_gpu_label()}")

    return issues, caveats


def check_novelty(record: dict[str, Any], registry_text: str) -> list[str]:
    """Whether this assignment, or this receipt, has already been claimed.

    Two distinct reuses, and the second one is the load-bearing check.

    One submission per (round, miner). Rows are deduplicated downstream by
    `hermesbench.identity`, but a second submission for the same assignment is not a
    near-duplicate to be scored -- it is the same work claimed twice, and catching it here
    is cheaper than catching it in the corpus.

    **One submission per receipt.** A receipt proves that *a* sealed check ran and passed;
    nothing a reviewer can recompute proves *which* manifest it checked, because the input
    digest the platform records is unsigned and its preimage is unpublished. So without
    this, one honestly-earned green receipt could be cited beside any number of unchecked
    batches. Single-use is what stops a receipt being spread, and it costs nothing --
    receipts are per-run by construction.
    """
    round_key = (record.get("round_id"), record.get("miner_id"))
    receipt_id = record.get("receipt_id")
    issues: list[str] = []
    for line in registry_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            existing = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (existing.get("round_id"), existing.get("miner_id")) == round_key:
            issues.append(f"{round_key[1]!r} has already submitted for round {round_key[0]!r}")
        if receipt_id and existing.get("receipt_id") == receipt_id:
            issues.append(
                f"receipt {receipt_id!r} was already cited by {existing.get('miner_id')!r} for round "
                f"{existing.get('round_id')!r}; a receipt attests one run and may back one submission"
            )
    return issues


def check_paths(changed_paths: list[str] | None) -> list[str]:
    """Rollout PRs are data-only."""
    if changed_paths is None:
        return []
    unexpected = sorted({p for p in changed_paths if p not in ALLOWED_PATHS})
    if unexpected:
        return [f"rollout-track PRs may only change {list(ALLOWED_PATHS)}; unexpected paths: {unexpected!r}"]
    return []


def check_append_only(base_text: str, head_text: str) -> list[str]:
    """The registry is history; a rewrite is rejected even when it is an improvement."""
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
    head_lines = [line.strip() for line in head_text.splitlines() if line.strip()]
    if head_lines[: len(base_lines)] != base_lines:
        return [
            f"{ROLLOUT_REGISTRY.as_posix()} is append-only; rebase onto the latest base and preserve "
            "every existing line in order"
        ]
    return []


def added_lines(base_text: str, head_text: str) -> list[dict[str, Any]]:
    """The JSON objects this PR appends."""
    # Positional, not set membership. A miner resubmitting a line byte-identical to one
    # already in the base would have it silently skipped, and a PR that appends nothing
    # would look like a PR that appends nothing wrong.
    base = [line.strip() for line in base_text.splitlines() if line.strip()]
    head = [line.strip() for line in head_text.splitlines() if line.strip()]
    added = []
    for number, line in enumerate(head[len(base) :], start=len(base) + 1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SubmissionError(f"line {number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise SubmissionError(f"line {number} is not a JSON object")
        added.append(record)
    return added


@dataclass(frozen=True)
class GateResult:
    """The gate's verdict and every reason behind it."""

    verdict: str
    issues: tuple[str, ...] = ()
    submission: Submission | None = None
    # Things the gate could not establish that do not refuse the submission.
    #
    # Separate from `issues` because folding them together forces a choice between rejecting every
    # submission and hiding the gap. An accepted submission with an unmeasured guest is genuinely
    # acceptable under today's rules *and* proves less than a reader would assume; a verdict with
    # no room for that has to lie in one direction or the other.
    caveats: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPT

    def to_record(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "accepted": self.accepted,
            "issues": list(self.issues),
            # In the record as well as on the result. A report copied into a pull request comment
            # without them reads as an unqualified pass.
            "caveats": list(self.caveats),
            "submission": self.submission.to_record() if self.submission else None,
        }


def gate(
    *,
    base_text: str,
    head_text: str,
    round_record: dict[str, Any],
    manifest: dict[str, Any],
    attestation: dict[str, Any] | None = None,
    changed_paths: list[str] | None = None,
    export_dir: Path | None = None,
) -> GateResult:
    """Decide one rollout submission.

    Every check runs and every reason is reported, rather than stopping at the first
    failure. A miner who has to resubmit three times to learn three problems will stop
    submitting, and the reasons are cheap to collect.
    """
    issues: list[str] = []
    issues.extend(check_paths(changed_paths))
    issues.extend(check_append_only(base_text, head_text))

    try:
        added = added_lines(base_text, head_text)
    except SubmissionError as exc:
        return GateResult(verdict=REJECT, issues=(*issues, str(exc)))

    if len(added) != 1:
        # One submission per PR: a batch of lines cannot be accepted or rejected as a unit,
        # and partially merging a PR is not something the merge button can express.
        return GateResult(verdict=REJECT, issues=(*issues, f"expected exactly one appended line, found {len(added)}"))

    record = added[0]
    issues.extend(check_shape(record))
    if issues:
        return GateResult(verdict=REJECT, issues=tuple(issues))

    issues.extend(check_scope(record, round_record))
    issues.extend(check_binding(record, manifest))
    issues.extend(check_manifest_work(manifest))
    issues.extend(check_exports(record, manifest, export_dir))
    attestation_issues, caveats = check_attestation(record, attestation, export_dir)
    issues.extend(attestation_issues)
    issues.extend(check_novelty(record, base_text))

    if issues:
        return GateResult(verdict=REJECT, issues=tuple(issues), caveats=tuple(caveats))
    return GateResult(verdict=ACCEPT, submission=Submission.from_record(record), caveats=tuple(caveats))

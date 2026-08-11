"""Everything an outsider needs to recheck a settled round, and a check that they can.

    python -m validator.audit build  --round r-001 --master-salt-env SPARK_MASTER_SALT
    python -m validator.audit verify --bundle var/audits/r-001 --withheld-check ./hidden.sh

The validator runs the surface, so a miner takes its word for the result unless something makes
that word checkable. This is that: a directory published after a round settles, holding the
challenge it was run against, the episode logs the verdicts came from, the scorecards, the round's
own ledger, and the opened commitment.

## What it proves, and what it does not

`verify` recomputes `salted_digest(withheld_check, per_task_salt)` and compares it to the
commitment the challenge published *before submissions opened*. That is the property worth having:
the validator graded against the check it committed to, not one written afterwards to suit a
result. The salt is per-task -- `derive_task_salt` is `HMAC(master, task_id)` -- so opening this
round leaves every unspent task's commitment sealed.

It does **not** prove the episodes came from the pinned model. Nothing in a bundle can: a validator
willing to fabricate a log can fabricate a consistent one. Binding that requires running the
evaluation inside a measured confidential VM and using this bundle's digest as the attestation
nonce, which is the same shape `proof.bundle` uses for the rollout track. Stating the limit here
rather than letting a reader infer more from a file called `manifest.json`.

## The master salt never enters a bundle

`RoundWindow.reveal` takes the master and returns the derived per-task salt; only the derived one
is written. A bundle carrying the master would open every commitment in the corpus, which is the
exact failure `derive_task_salt` exists to prevent -- and it would do it silently, because the
bundle would look no different.

`build` refuses if the master appears anywhere in the assembled files, checked by searching rather
than by trusting the code above it not to have put it there.

## Refused before the round settles

The salt is what makes a graded round auditable, and releasing it earlier hands the withheld check
to whoever still holds a submission. `RoundWindow.reveal` enforces that; this refuses first so the
error names the bundle rather than the round.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIT_DIR = Path("var/audits")
MANIFEST = "manifest.json"


class AuditError(RuntimeError):
    """A bundle cannot be built or does not check out."""


@dataclass(frozen=True)
class Bundle:
    root: Path

    @property
    def manifest(self) -> dict[str, Any]:
        path = self.root / MANIFEST
        if not path.is_file():
            raise AuditError(f"{path} is missing; this is not an audit bundle")
        return json.loads(path.read_text(encoding="utf-8"))

    def read(self, relative: str) -> dict[str, Any]:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_digests(root: Path) -> dict[str, str]:
    """Every file in the bundle except the manifest, which cannot digest itself."""
    return {
        p.relative_to(root).as_posix(): digest_bytes(p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }


def claim_digest(digests: dict[str, str]) -> str:
    """One digest over the whole bundle.

    Over the sorted path-to-digest map rather than over concatenated contents: a file added,
    removed or edited all move the result, and a concatenation in directory order would miss a
    rename that swapped two files.

    This is the value to use as an attestation nonce when the evaluation runs inside a measured VM
    -- the same binding `proof.bundle.claim_sha256` provides for the rollout track.
    """
    return digest_bytes(json.dumps(digests, sort_keys=True).encode("utf-8"))


def build(
    *,
    round_id: str,
    master_salt: str,
    store: Any = None,
    scorecard_dir: Path | None = None,
    episode_dir: Path | None = None,
    out: Path | None = None,
) -> Bundle:
    """Assemble the bundle for one settled round."""
    from hermes.round import SETTLED
    from validator.store import RoundStore

    store = store or RoundStore()
    window = store.load(round_id)
    if window.state != SETTLED:
        raise AuditError(
            f"round {round_id} is {window.state!r}; a bundle is built once it has SETTLED. The salt "
            "is what makes a graded round auditable, and releasing it earlier hands the withheld "
            "check to whoever still holds a submission."
        )

    root = (out or AUDIT_DIR) / round_id
    if root.exists():
        shutil.rmtree(root)
    (root / "episodes").mkdir(parents=True)
    (root / "scorecards").mkdir(parents=True)

    reveal = window.reveal(master_salt)
    _write(root / "round.json", window.to_record())
    _write(root / "challenge.json", window.challenge.to_record())
    _write(root / "reveal.json", reveal.to_record())

    scorecards = scorecard_dir or Path("var/scorecards")
    episodes = episode_dir or Path("var/judge") / round_id
    for miner in sorted(window.submissions):
        card = scorecards / f"{round_id}-{miner}.json"
        if card.is_file():
            shutil.copy(card, root / "scorecards" / f"{miner}.json")
        log = episodes / f"{miner}.jsonl"
        if log.is_file():
            shutil.copy(log, root / "episodes" / f"{miner}.jsonl")

    _refuse_master(root, master_salt)
    digests = file_digests(root)
    _write(
        root / MANIFEST,
        {
            "round_id": round_id,
            "task_id": window.task_id,
            "settled_at": window.to_record().get("settled_at"),
            "files": digests,
            "claim_sha256": claim_digest(digests),
            # Said in the bundle so a reader does not infer more from it than it carries.
            "proves": "the validator graded against the check it committed to before submissions opened",
            "does_not_prove": (
                "that the episodes came from the pinned model; binding that needs the evaluation to "
                "run inside a measured confidential VM with claim_sha256 as the attestation nonce"
            ),
        },
    )
    return Bundle(root)


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refuse_master(root: Path, master_salt: str) -> None:
    """Search the assembled files for the master secret.

    Checked rather than assumed. `reveal` returns the derived salt and nothing here writes the
    master, but a bundle carrying it would open every commitment in the corpus and would look no
    different from a correct one -- so the cheap search is worth more than the confidence.
    """
    if not master_salt:
        return
    needle = master_salt.encode("utf-8")
    for path in sorted(root.rglob("*")):
        if path.is_file() and needle in path.read_bytes():
            raise AuditError(
                f"{path.relative_to(root).as_posix()} contains the master salt. A bundle carrying it "
                "opens every unspent commitment in the corpus, which is the failure derive_task_salt "
                "exists to prevent."
            )


def verify(bundle: Bundle, withheld_check: str) -> list[str]:
    """Recheck a bundle against the withheld check an auditor holds. Empty means it checks out.

    Every problem, not the first: an auditor who learns one at a time cannot tell a bundle with a
    single clerical error from one that fails in several ways at once.
    """
    from hermes.harness import salted_digest

    problems: list[str] = []
    manifest = bundle.manifest

    # Integrity first. Every later claim is about the contents of these files.
    actual = file_digests(bundle.root)
    claimed = {str(k): str(v) for k, v in (manifest.get("files") or {}).items()}
    for name in sorted(set(claimed) | set(actual)):
        if name not in actual:
            problems.append(f"{name}: listed in the manifest and absent from the bundle")
        elif name not in claimed:
            problems.append(f"{name}: present in the bundle and not listed in the manifest")
        elif actual[name] != claimed[name]:
            problems.append(f"{name}: digests {actual[name][:23]}..., manifest says {claimed[name][:23]}...")
    if claim_digest(claimed) != str(manifest.get("claim_sha256") or ""):
        problems.append("claim_sha256 does not match the file digests it is computed from")

    reveal = bundle.read("reveal.json")
    challenge = bundle.read("challenge.json")
    published = str((challenge.get("withheld") or {}).get("hidden_verify_commitment") or "")
    salt = str(reveal.get("per_task_salt") or reveal.get("salt") or "")
    if not published:
        problems.append("the challenge published no withheld-check commitment; there is nothing to open")
    elif not salt:
        problems.append("the reveal carries no per-task salt, so the commitment cannot be opened")
    else:
        # The claim the bundle exists to support.
        recomputed = salted_digest(withheld_check, salt)
        if recomputed != published:
            problems.append(
                f"the withheld check does not open the published commitment: it digests to "
                f"{recomputed[:23]}... under this salt and the challenge committed to {published[:23]}.... "
                "Either this is not the check that graded the round, or the round was graded against "
                "something other than what it committed to."
            )

    if str(reveal.get("round_id") or "") != str(manifest.get("round_id") or ""):
        problems.append("the reveal is for a different round than the manifest names")
    if not reveal.get("opens_only_this_task", True):
        problems.append("the reveal claims to open more than this task, which a per-task salt cannot")
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("--round", dest="round_id", default="")
    parser.add_argument("--master-salt-env", default="SPARK_MASTER_SALT", help="env var holding the master salt")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--scorecards", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None, help="verify only")
    parser.add_argument("--withheld-check", type=Path, default=None, help="verify only")
    args = parser.parse_args(argv)

    from validator.store import RoundStore

    try:
        if args.action == "build":
            master = os.environ.get(args.master_salt_env, "")
            if not master:
                print(
                    f"validator.audit: {args.master_salt_env} is not set. The master salt is read from "
                    "the environment and never from a flag, so it does not land in a shell history or "
                    "a process listing.",
                    file=sys.stderr,
                )
                return 2
            bundle = build(
                round_id=args.round_id,
                master_salt=master,
                store=RoundStore(args.store),
                scorecard_dir=args.scorecards,
                episode_dir=args.episodes,
                out=args.out,
            )
            manifest = bundle.manifest
            print(f"built {bundle.root}")
            print(f"  {len(manifest['files'])} file(s)")
            print(f"  claim_sha256 {manifest['claim_sha256']}")
            print(f"  proves: {manifest['proves']}")
            print(f"  does not prove: {manifest['does_not_prove']}")
            return 0

        if args.bundle is None or args.withheld_check is None:
            print("validator.audit: verify needs --bundle and --withheld-check", file=sys.stderr)
            return 2
        problems = verify(Bundle(args.bundle), args.withheld_check.read_text(encoding="utf-8"))
    except (AuditError, OSError) as exc:
        print(f"validator.audit: {exc}", file=sys.stderr)
        return 2

    if problems:
        print(f"{args.bundle}: FAILS")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{args.bundle}: checks out")
    print("  the withheld check opens the commitment the challenge published before submissions opened,")
    print("  and every file digests to what the manifest claims")
    return 0


__all__ = [
    "AUDIT_DIR",
    "MANIFEST",
    "AuditError",
    "Bundle",
    "build",
    "claim_digest",
    "digest_bytes",
    "file_digests",
    "main",
    "verify",
]


if __name__ == "__main__":
    raise SystemExit(main())

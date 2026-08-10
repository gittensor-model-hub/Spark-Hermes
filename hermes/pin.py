"""Building a `HarnessPin` from the world, rather than declaring one by hand.

`HarnessPin` names every digest that can change what an agent observes -- commit, system
prompt, tool schemas, container image, dependency lock -- and `is_pinned` states what a
real pin needs. Nothing constructed one. It appears in the package only in `__all__`, and
the sole place it is instantiated is a test, which means `harness_digest` has always been
computed from values a caller typed.

That is the difference between a pin and a label. A label can be copied from the last run
after the harness changed underneath it; a pin cannot, because it is read from the thing
it describes.

Everything here reads from disk or from git and refuses when it cannot. The refusals
matter more than the reads: a pin assembled with blanks in it is worse than no pin,
because `is_pinned` would still be satisfied by three non-empty strings and the fair-fight
check would wave through two harnesses that were never the same.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from hermes.harness import HarnessError, digest_mapping, digest_text
from hermes.router.manifest import HarnessPin


def git_commit(root: Path | None = None) -> str:
    """The working tree's commit, refusing a dirty tree.

    A dirty tree cannot be pinned: the commit names one state and the files on disk are a
    different one, so anyone re-running from that commit gets a different harness while
    the digest claims otherwise. This is the check that makes the pin mean something on a
    developer machine rather than only in CI.
    """
    cwd = str(root) if root else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        # Not stripped: porcelain prefixes every line with two status columns and a space,
        # so stripping the blob eats the first line's leading space and the `line[3:]` slice
        # below then cuts a character off that one filename.
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise HarnessError(f"cannot read the git commit: {exc}") from exc
    if dirty.strip():
        changed = [line[3:] for line in dirty.splitlines()[:5]]
        raise HarnessError(
            f"working tree is dirty ({', '.join(changed)}...); the commit names one harness and the "
            "files on disk are another, so the pin would describe a state nobody can reproduce"
        )
    return commit


def digest_file(path: Path) -> str:
    if not path.is_file():
        raise HarnessError(f"{path} does not exist; it cannot be pinned")
    return digest_text(path.read_text(encoding="utf-8"))


def digest_tool_schemas(schemas: dict[str, dict[str, Any]]) -> str:
    """Digest the advertised tool set.

    Over the schemas, not the names. Two harnesses offering `terminal` with different
    parameter sets are different harnesses, and a name-only digest would call them equal --
    which is exactly the drift a pin exists to catch.
    """
    if not schemas:
        raise HarnessError("no tool schemas to pin; a harness that advertises nothing measures nothing")
    return digest_mapping({name: schemas[name] for name in sorted(schemas)})


def build_pin(
    *,
    system_prompt: str,
    tool_schemas: dict[str, dict[str, Any]],
    root: Path | None = None,
    lock_file: Path | None = None,
    container_image_digest: str = "",
    release: str = "",
    tag: str = "",
    conformance_verified: bool = False,
) -> HarnessPin:
    """Assemble a pin by reading the harness, not by describing it.

    `container_image_digest` is accepted rather than discovered, because there is no image
    in this repo yet and inventing a value for the field would be the same failure the
    module exists to prevent. Left empty it simply does not contribute -- the pin is then
    honest that the container is unpinned, which is a weaker claim rather than a false one.
    """
    if not system_prompt.strip():
        raise HarnessError("no system prompt to pin; it is the largest single thing that changes agent behaviour")
    lock = lock_file if lock_file is not None else ((root or Path(".")) / "uv.lock")
    return HarnessPin(
        release=release,
        tag=tag,
        commit=git_commit(root),
        system_prompt_digest=digest_text(system_prompt),
        tool_schema_digest=digest_tool_schemas(tool_schemas),
        dependency_lock_digest=digest_file(lock) if lock.is_file() else "",
        container_image_digest=container_image_digest,
        conformance_verified=conformance_verified,
    )


def load_tool_schemas(path: Path) -> dict[str, dict[str, Any]]:
    """Read a committed tool-schema file.

    Committed rather than assembled at runtime, so the schemas the model was shown are in
    the history alongside the results they produced. A schema set that lives only in the
    process that ran cannot be checked afterwards, and the digest would be pinning
    something nobody can look at.
    """
    if not path.is_file():
        raise HarnessError(f"{path} does not exist; the tool schemas must be committed to be pinnable")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise HarnessError(f"{path} is not a non-empty object of tool schemas")
    for name, schema in data.items():
        if not isinstance(schema, dict) or "parameters" not in schema:
            raise HarnessError(f"{path}: tool {name!r} has no parameters; the signature is the point")
    return data

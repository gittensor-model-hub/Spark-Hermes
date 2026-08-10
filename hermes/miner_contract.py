"""What a miner is allowed to submit, and why each refusal exists.

The agent is upstream and unmodified. A miner competes by changing how the *same* frozen
model approaches a task -- the behavioural policy in `SOUL.md`, the strategy in a skill, the
playbooks a skill pulls in -- and by nothing else. Three Markdown paths.

**Deny by default.** A submission may touch only what the contract names. The alternative
fails in the dangerous direction: upstream adds a file, nobody updates a denylist, and it is
silently miner-editable. Anything unrecognised is refused with the reason that it is
unrecognised, which is a fixable complaint rather than a hole.

**Extension is checked as well as path.** `skills/x/references/helper.py` matches the
references glob and is not a playbook. A contract that only matched directories would admit
executable content through a path that reads as documentation.

## What this cannot do, stated because someone will otherwise assume it

Markdown is an instruction to a model that holds a terminal. Nothing here prevents a
`SKILL.md` that says "write a helper script that greps, tests and summarises in one pass,
then run it" -- which produces exactly the one-call-hides-thirty-operations effect that the
`scripts/` denial exists to prevent. The contract governs what is **submitted**, not what is
**executed**.

The right conclusion is not a stricter allowlist -- it is that tool-call count is the softest
of the efficiency dimensions and must never carry a submission alone. Tokens and wall time
are not fooled the same way: the hidden operations still cost seconds, and the helper's
output still enters context as tokens.

## Config is not policed here

`config.yaml` is denied, but the real mechanism is upstream's managed scope: a root-owned
`/etc/hermes/config.yaml` whose keys win over the user's config, their `.env`, and the shell
environment. Pinned keys become *immutable* rather than merely forbidden, so there is nothing
to check. See `hermes.profile`, which also records why that only holds when the validator
owns the machine.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

CONTRACT_PATH = PurePosixPath(__file__).parent / "miner_contract.json"


class ContractError(ValueError):
    """The contract itself is missing or malformed."""


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True)
class MinerContract:
    allowed: tuple[tuple[str, str], ...]
    denied: tuple[tuple[str, str], ...]
    extensions: tuple[str, ...]
    raw: dict[str, Any]

    def check(self, paths: list[str]) -> list[Violation]:
        """Every reason this submission is refused. Empty means acceptable.

        All of them, not the first: a miner who learns one problem per resubmission stops
        resubmitting, and the reasons cost nothing to collect.
        """
        violations: list[Violation] = []
        for raw in paths:
            path = PurePosixPath(raw.strip().lstrip("./")).as_posix()
            if not path or path in (".", ".."):
                continue
            if ".." in PurePosixPath(path).parts:
                # A submission is a set of files, not a set of instructions about where to
                # put them. `../../run_agent.py` inside an archive is how an allowlist that
                # only pattern-matches gets walked past.
                violations.append(Violation(path, "path escapes the submission root"))
                continue

            explicit = next((why for pattern, why in self.denied if fnmatch.fnmatch(path, pattern)), None)
            if explicit is not None:
                violations.append(Violation(path, explicit))
                continue

            if not any(fnmatch.fnmatch(path, pattern) for pattern, _ in self.allowed):
                violations.append(
                    Violation(
                        path,
                        "not in the contract. A submission may contain only "
                        + ", ".join(pattern for pattern, _ in self.allowed)
                        + " -- anything else is refused rather than assumed harmless, because the "
                        "alternative is that a file nobody has considered becomes editable by default",
                    )
                )
                continue

            if self.extensions and not any(path.endswith(ext) for ext in self.extensions):
                violations.append(
                    Violation(
                        path,
                        f"matches an allowed path but is not {' or '.join(self.extensions)}; a strategy is "
                        "prose, and executable content arriving through a documentation path is the "
                        "scripts/ refusal wearing a different filename",
                    )
                )
        return violations

    def reason_for(self, pattern: str) -> str:
        for candidate, why in (*self.allowed, *self.denied):
            if candidate == pattern:
                return why
        raise KeyError(pattern)


def load(path: Any = None) -> MinerContract:
    import pathlib

    source = pathlib.Path(str(path or CONTRACT_PATH))
    if not source.is_file():
        raise ContractError(f"no miner contract at {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    allowed = tuple((str(e["path"]), str(e["why"])) for e in record.get("allowed") or ())
    denied = tuple((str(e["path"]), str(e["why"])) for e in record.get("denied") or ())
    if not allowed:
        # A contract with nothing allowed refuses every submission, and would do it while
        # looking like a working gate.
        raise ContractError(f"{source}: the contract allows nothing, so no submission can pass")
    return MinerContract(
        allowed=allowed,
        denied=denied,
        extensions=tuple(str(e) for e in record.get("extensions") or ()),
        raw=record,
    )


__all__ = ["CONTRACT_PATH", "ContractError", "MinerContract", "Violation", "load"]

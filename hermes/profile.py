"""Assemble a run profile: pinned agent, pinned model, pinned config, miner Markdown.

A rollout is only a comparison if everything except the miner's strategy is identical. The
agent is an upstream commit, the model is a revision, the tool set is whatever that agent
ships -- and the config has to be pinned too, because `agent.reasoning_effort` alone moves
token count directly. A miner who turns it down wins the efficiency axis without discovering
anything, and trades success rate to do it, which is what "correctness before efficiency"
exists to forbid.

## Config is pinned by upstream's managed scope, not by our validation

Hermes reads a root-owned managed layer -- `/etc/hermes/config.yaml` by default, relocatable
with `HERMES_MANAGED_DIR` -- whose keys win over the user's `config.yaml`, their `.env`, and
the shell environment, per key. Pinned keys become **immutable**, so there is nothing for a
gate to check. That is strictly better than an allowlist a miner submits against: an
allowlist can be walked past, a filesystem permission cannot.

## And the limit that decides the execution model

Upstream states the enforcement plainly: *"That filesystem permission is the enforcement
mechanism"*, and then the consequence: *"A user who can set `HERMES_MANAGED_DIR` can repoint
managed scope at a directory they control, defeating it."*

So managed scope enforces against a standard user **on someone else's machine**. On attested
miner hardware the miner is root: they can edit the managed file or export
`HERMES_MANAGED_DIR` at anything. Config pinning is therefore:

    validator-executed   enforced. Set HERMES_MANAGED_DIR in the service unit or image.
    miner-executed       NOT enforced by configuration. It has to be enforced by
                         measurement -- the resolved config must be inside the attested
                         workload root, or the pin is an assertion.

`doctor_probe` exists for the second case: upstream reports the *resolved* managed directory,
so a redirect is observable rather than silent. Capturing it makes the pin checkable instead
of assumed. It is not a substitute for measurement; it is what measurement should cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Every key a miner must not be able to move, and the reason each one matters. Pinned in the
# managed layer rather than listed in a denylist, so they are immutable rather than forbidden.
#
# `agent.reasoning_effort` is here deliberately. It is the one knob that changes a scored
# metric without changing behaviour quality: low effort spends fewer tokens and loses success
# rate. That is a dial, not a discovery, and leaving it open means the cheapest possible move
# tops the efficiency leaderboard. If the right setting is an open question it is a maintainer
# sweep once per epoch, not a competition.
PINNED_CONFIG_KEYS: tuple[str, ...] = (
    "model",
    "provider",
    "agent.reasoning_effort",
    "terminal.backend",
    "terminal.timeout",
    "toolsets",
    "tool_output",
    "compression",
    "context.engine",
)

MANAGED_DIR_ENV = "HERMES_MANAGED_DIR"


class ProfileError(ValueError):
    """A profile cannot be assembled from what was supplied."""


@dataclass(frozen=True)
class RunProfile:
    """One reproducible arrangement of agent, model, config and strategy."""

    agent_repository: str
    agent_commit: str
    model_repository: str
    model_revision: str
    managed_config: dict[str, Any]
    miner_files: tuple[str, ...] = ()
    managed_dir: str = "/etc/hermes"
    validator_executed: bool = True

    @property
    def config_pin_enforced(self) -> bool:
        """Whether the managed layer actually holds against the party being measured.

        False on miner hardware, and reported rather than assumed. A profile that claims a
        pinned config on a machine the miner is root on is claiming a property upstream
        explicitly says it does not have.
        """
        return self.validator_executed

    def to_record(self) -> dict[str, Any]:
        return {
            "agent": {"repository": self.agent_repository, "commit": self.agent_commit},
            "model": {"repository": self.model_repository, "revision": self.model_revision},
            "managed_dir": self.managed_dir,
            "pinned_config_keys": sorted(self.managed_config),
            "miner_files": sorted(self.miner_files),
            "validator_executed": self.validator_executed,
            # Stated on every record. The difference between these two is the difference
            # between a pin and a request.
            "config_pin_enforced": self.config_pin_enforced,
            "config_pin_mechanism": (
                "root-owned managed scope; filesystem permission is the enforcement"
                if self.config_pin_enforced
                else "NONE -- the executing party can repoint HERMES_MANAGED_DIR, so the pin must be "
                "covered by attested measurement of the resolved config"
            ),
        }


def managed_config(values: dict[str, Any]) -> dict[str, Any]:
    """The managed layer, refusing anything that leaves a required key unpinned.

    Checked at assembly rather than at run time. A profile discovered to be missing a pin
    after the run is a run that has to be thrown away, and the comparison it was part of
    with it.
    """
    missing = [k for k in PINNED_CONFIG_KEYS if k not in values]
    if missing:
        raise ProfileError(
            "these keys must be pinned in the managed layer and are not: "
            + ", ".join(missing)
            + ". An unpinned key is one a miner can move, and `agent.reasoning_effort` alone "
            "changes token count without changing behaviour quality"
        )
        # Unknown extras are allowed: pinning more than the minimum is a tightening.
    return dict(values)


def assemble(
    *,
    agent_repository: str,
    agent_commit: str,
    model_repository: str,
    model_revision: str,
    config: dict[str, Any],
    miner_dir: Path | None = None,
    managed_dir: str = "/etc/hermes",
    validator_executed: bool = True,
) -> RunProfile:
    """Build a profile, refusing a miner submission the contract does not accept."""
    from eval.hf_pin import check_revision
    from hermes.miner_contract import load as load_contract

    if len(agent_commit) != 40 or any(c not in "0123456789abcdef" for c in agent_commit.lower()):
        # The same rule the model pin follows. A branch name resolves to whatever the
        # repository holds when someone runs it, so two runs could agree on every other
        # digest and still have run different agents.
        raise ProfileError(f"agent commit {agent_commit!r} is not a 40-character sha; a branch is not a pin")
    issues = check_revision(model_revision, field="model revision")
    if issues:
        raise ProfileError("; ".join(issues))

    files: tuple[str, ...] = ()
    if miner_dir is not None:
        if not miner_dir.is_dir():
            raise ProfileError(f"miner submission {miner_dir} is not a directory")
        files = tuple(sorted(p.relative_to(miner_dir).as_posix() for p in miner_dir.rglob("*") if p.is_file()))
        violations = load_contract().check(list(files))
        if violations:
            raise ProfileError("miner submission refused:\n  " + "\n  ".join(str(v) for v in violations))

    return RunProfile(
        agent_repository=agent_repository,
        agent_commit=agent_commit,
        model_repository=model_repository,
        model_revision=model_revision,
        managed_config=managed_config(config),
        miner_files=files,
        managed_dir=managed_dir,
        validator_executed=validator_executed,
    )


def write_managed_layer(profile: RunProfile, root: Path) -> Path:
    """Write the managed config to `root`, for a validator that owns the machine.

    Returns the file written. Does not chown or chmod: making it root-owned is the
    deployment's job, and doing it here would suggest this function is what makes the pin
    hold when the filesystem permission is.
    """
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(profile.managed_config, sort_keys=True), encoding="utf-8")
    return path


def doctor_probe(managed_dir_env: str | None) -> dict[str, Any]:
    """What to record so a repointed managed directory is visible in the run.

    Upstream's `hermes doctor` reports the *resolved* managed directory. Capturing it turns
    "the config was pinned" from an assertion into something a reviewer can check against the
    profile -- and on miner hardware it is the only observable there is, because the mechanism
    itself does not hold there.
    """
    return {
        "probe": "hermes doctor",
        "reads": "the resolved managed directory",
        "why": "a redirect of HERMES_MANAGED_DIR defeats the pin, and upstream makes it visible",
        f"{MANAGED_DIR_ENV}_observed": managed_dir_env or "",
    }


def compose_system_prompt(base: str, miner_dir: Path) -> str:
    """Fold a validated miner submission into the harness system prompt.

    `assemble` proves a submission is *allowed*; this is what makes it *take effect*. Until
    both existed in one path the miner-editable surface was unreachable: the contract, the
    pinned config keys and the profile were all built and tested, while the runner read its
    system prompt from a single file with no way in. A miner could submit a perfect SKILL.md
    and it would change nothing about a run.

    ## Order, and why

    `SOUL.md` goes at the head, where the contract says it belongs -- "the profile's
    behavioural identity, loaded at the head of the system prompt". The harness prompt follows
    it, so the rules the benchmark's own measurements depend on -- report only numbers you
    observed, recover rather than work around -- cannot be displaced by a submission. A miner
    sets the identity; the harness keeps the invariants.

    ## `references/` is refused rather than ignored

    The contract admits `skills/*/references/*.md` so a miner can "optimise *when* extra
    knowledge is worth its tokens". That property only exists with lazy loading: Hermes pulls a
    reference in when the agent reaches for it, and pays for it then. This runner has no such
    affordance, and `LocalToolExecutor` confines `file_read` to the workspace, so the agent
    could not reach a strategy directory even if it were told where one was.

    That leaves two bad options and one honest one. Eager loading destroys the exact trade-off
    references exist for, since every task would pay for every reference whether it used one or
    not. Silent ignoring is worse: a miner would tune references and then be scored on a
    strategy that never contained them, with nothing to show the difference. So they are refused
    with a reason, and stay refused until lazy loading exists.

    ## What this does to token accounting, stated because it runs against the miner

    `SKILL.md` is loaded eagerly, which production Hermes would not do -- it reads a skill's
    full text only when the agent reaches for it. A long SKILL.md therefore costs tokens here on
    every task from the first turn, so the token metric under this runner is stricter than
    production would be. Recorded rather than hidden: a miner optimising against this number
    deserves to know which direction the difference runs.
    """
    soul = miner_dir / "SOUL.md"
    skills = sorted(miner_dir.glob("skills/*/SKILL.md"))
    references = sorted(miner_dir.glob("skills/*/references/*.md"))

    if references:
        shown = ", ".join(p.relative_to(miner_dir).as_posix() for p in references[:3])
        raise ProfileError(
            f"submission carries {len(references)} reference file(s) ({shown}"
            f"{', ...' if len(references) > 3 else ''}), and this runner has no lazy-loading "
            "affordance for them. Loading them eagerly would destroy the token trade-off they "
            "exist for, since every task would pay for every reference whether it used one or "
            "not; ignoring them would score you on a strategy that never ran. Fold what you "
            "need into SKILL.md, or wait for reference loading."
        )

    parts: list[str] = []
    if soul.is_file():
        text = soul.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    parts.append(base.strip())
    for skill in skills:
        text = skill.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"# Strategy: {skill.parent.name}\n\n{text}")
    return "\n\n".join(parts) + "\n"


__all__ = [
    "MANAGED_DIR_ENV",
    "PINNED_CONFIG_KEYS",
    "ProfileError",
    "RunProfile",
    "assemble",
    "compose_system_prompt",
    "doctor_probe",
    "managed_config",
    "write_managed_layer",
]

"""Exact active agent/model handoff for competition baseline and admitted execution.

Epochs bind the installed evaluator separately from the active agent artifact. Fixture
completions are an explicit external serving substitute, confined to an existing
immutable fixture state root; they confer no production approval or training claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from admin.artifacts import StageError, canonical, content_digest, read_record, same_domain
from admin.candidates import bound_record, file_identity


def base_profile(profile: str = "bf16") -> dict[str, str]:
    from hermes.base_model import load

    pin = load()
    if profile == "bf16":
        return {"repository": pin.repository, "revision": pin.revision, "dialect": pin.hermes_dialect}
    if profile != "rtx5090-poc":
        raise StageError("unknown competition model profile")
    import yaml

    from admin.training import RECIPES

    record = yaml.safe_load((RECIPES / "rtx5090-poc.yaml").read_text())
    return {"repository": record["base_model"], "revision": record["base_model_revision"], "dialect": "qwen35"}


def base_agent(profile: str = "bf16") -> dict[str, Any]:
    from hermes.pin import load_tool_schemas
    from hermesbench.runner import HARNESS_DIR

    return {
        "schema": "spark-agent-v1",
        "system": (HARNESS_DIR / "system_prompt.txt").read_bytes().decode("utf-8"),
        "dialect": base_profile(profile)["dialect"],
        "tool_schemas": load_tool_schemas(HARNESS_DIR / "tools.json"),
        "native_tool_messages": False,
    }


def base_agent_id(profile: str = "bf16") -> str:
    return content_digest(base_agent(profile))


def epoch_binding(pair: dict[str, Any]) -> dict[str, Any]:
    return {k: pair[k] for k in ("origin", "candidate", "approval", "generation", "epoch_id", "model_id", "agent_id")}


def active_pair(release_root: Path) -> dict[str, Any]:
    from admin.release import ReleaseAuthority
    from admin.runtime_protocol import writable

    authority = ReleaseAuthority(release_root)
    writable(authority)
    pair = authority.active_pair()
    # Capture exact committed bytes, not a second unchecked read of the agent path.
    if canonical(bound_record(pair["agent"])) != canonical(pair["agent_record"]):
        raise StageError("active agent bytes differ from the accepted pair")
    return pair


def competition_harness(
    tasks: list[Any],
    *,
    agent_id: str,
    suite_name: str = "all",
    salt: str = "",
    tool_timeout_s: int = 120,
    sampling: dict[str, Any] | None = None,
) -> str:
    from hermes.harness import crossed_runtime_identity, digest_suite, fingerprint_task
    from hermesbench.verify import OBSERVATION_LIMIT

    # Installed file bytes replace a fictitious clean Git commit. The crossed runtime
    # covers policy, tools, verifiers and protocol; include competition adapters too.
    from miner import evaluate
    from validator import judge

    runtime = crossed_runtime_identity()
    adapters = {}
    for module in (evaluate, judge):
        if module.__file__ is None:
            raise StageError("competition adapter has no installed source identity")
        adapters[module.__name__] = file_identity(Path(module.__file__))["sha256"]
    return content_digest(
        {
            "schema": "spark-competition-harness-v1",
            "runtime": runtime,
            "adapters": adapters,
            "agent_id": agent_id,
            "suite": digest_suite(suite_name, [fingerprint_task(t, salt=salt) for t in tasks]).to_record(),
            "executor": "local",
            "observation_limit": OBSERVATION_LIMIT,
            "tool_timeout_s": tool_timeout_s,
            "sampling": sampling or {},
            "completion_budget": {"max_tokens": 32768},
        }
    )


def build_epoch(
    tasks: list[Any],
    *,
    attempt_ids: list[str],
    epoch_id: str | None = None,
    task_root: Path | None = None,
    release_root: Path | None = None,
    profile: str = "bf16",
    salt: str = "",
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from validator.score import policy_record

    if (
        len(attempt_ids) < 10
        or any(not isinstance(a, str) or not a for a in attempt_ids)
        or len(set(attempt_ids)) != len(attempt_ids)
    ):
        raise StageError("competition requires at least ten unique string attempt IDs")
    pair = active_pair(release_root) if release_root is not None else None
    if pair is not None and epoch_id is not None and epoch_id != pair["epoch_id"]:
        raise StageError("explicit epoch differs from active release")
    epoch_id = pair["epoch_id"] if pair is not None else epoch_id
    if not isinstance(epoch_id, str) or not epoch_id:
        raise StageError("bootstrap competition requires an explicit epoch ID")
    agent_id = pair["agent_id"] if pair is not None else base_agent_id(profile)
    result = {
        "schema": "spark-competition-epoch-v1",
        "epoch_id": epoch_id,
        "model_revision": pair["model_id"] if pair is not None else base_profile(profile)["revision"],
        "agent_id": agent_id,
        "harness_digest": competition_harness(tasks, agent_id=agent_id, salt=salt, sampling=sampling),
        "attempt_ids": list(attempt_ids),
        "score_policy": policy_record(),
        "profile": pair["model"]["profile"] if pair is not None else profile,
        "sampling": dict(sampling or {}),
    }
    if pair is not None:
        result["incumbent"] = epoch_binding(pair)
    if task_root is not None:
        result["task_root"] = str(task_root.resolve())
    return result


def check_pair(pair: dict[str, Any], *, epoch: dict[str, Any], origin: dict[str, str], model: str) -> None:
    same_domain(pair["origin"], origin)
    if canonical(epoch.get("incumbent")) != canonical(epoch_binding(pair)):
        raise StageError("competition epoch has a stale or different incumbent pair")
    if epoch.get("epoch_id") != pair["epoch_id"] or epoch.get("model_revision") != pair["model_id"]:
        raise StageError("competition epoch does not name the active model/epoch")
    if epoch.get("agent_id") != pair["agent_id"] or model != pair["model_id"]:
        raise StageError("competition execution model/agent differs from active pair")


class FixtureServing:
    """Scripted external completions indexed by exact model, agent, task and attempt."""

    def __init__(
        self, path: Path, *, root: Path, origin: dict[str, str], model_id: str, agent_id: str, epoch: dict[str, Any]
    ):
        from validator.persistence import state_identity

        if not (root / ".identity").is_file():
            raise StageError("fixture serving requires an existing explicit fixture root")
        identity = state_identity(root)
        same_domain(identity, origin)
        if identity["mode"] != "fixture" or origin["mode"] != "fixture":
            raise StageError("fixture serving is forbidden in production")
        script = bound_record(file_identity(path))
        if (
            script.get("schema") != "spark-serving-fixture-v1"
            or canonical(script.get("origin")) != canonical(identity)
            or script.get("model_id") != model_id
        ):
            raise StageError("fixture serving issuer/model mismatch")
        self.script, self.agent_id, self.epoch = script, agent_id, epoch
        self.observations: list[dict[str, Any]] = []

    def completion(self, task_id: str, attempt: int):
        attempt_id = self.epoch["attempt_ids"][attempt]
        try:
            row = self.script["agents"][self.agent_id][task_id][attempt_id]
        except (KeyError, TypeError) as exc:
            raise StageError("fixture has no exact agent/task/attempt response") from exc
        responses = row.get("responses")
        if not isinstance(responses, list) or not responses or any(not isinstance(v, str) for v in responses):
            raise StageError("fixture requires explicit completion responses")
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
            raise StageError("fixture token counts must be nonnegative integers")
        scripted = iter(responses)

        def complete(messages, *, tools=None):
            self.observations.append(
                {
                    "fixture_only": True,
                    "model_id": self.script["model_id"],
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                }
            )
            try:
                return next(scripted), {k: row[k] for k in ("prompt_tokens", "completion_tokens")}
            except StopIteration as exc:
                raise StageError("fixture exhausted exact attempt responses") from exc

        return complete


def serving_record(path: Path) -> dict[str, Any]:
    """Read a deployment declaration; TrustedServing verifies it on each completion."""
    return read_record(path)

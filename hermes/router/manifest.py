"""Agent modules and expert manifests.

    Do not route to a model. Route to a verified worker configuration.

A model id cannot express what a route actually needs to be valid: which tools the worker
holds, whether the sandbox has a GPU, how much context it can take, which verifier will
score it, or whether we are permitted to train on its output. Routing to a bare model
name means those constraints get discovered at execution time, as failures.

An `AgentModule` is the routable unit: model + Hermes profile + tools + environment +
verifier + budget. `ExpertManifest` is the machine-readable declaration a router reads
*before* any live evidence exists, which is what lets a brand-new expert be routed to on
its first day instead of waiting for a thousand samples.

Registry lookups go through **aliases** (`expert.cuda.optimization`), never model ids, so
upgrading `spark-hermes-cuda-3.6-27b` to `-3.8-` is a registry edit rather than a change
to routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Whether we are permitted to train on a teacher's output. Unknown is not approved:
# the gate fails closed, because "we were not sure" is not a defence.
RIGHTS_APPROVED = "approved"
RIGHTS_DENIED = "denied"
RIGHTS_UNKNOWN = "unknown"


class ManifestError(ValueError):
    """A manifest is malformed or contradicts itself."""


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int = 100
    max_wall_time_s: int = 14_400
    max_tokens: int = 0  # 0 = unbounded

    def to_record(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_wall_time_s": self.max_wall_time_s,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class AgentModule:
    """A model paired with everything it needs to actually do work.

    The model supplies domain intelligence; the module supplies the operating procedure.
    Routing selects the pairing.
    """

    module_id: str
    model_id: str
    domains: tuple[str, ...]
    actions: tuple[str, ...]
    tools: tuple[str, ...] = ()
    verifiers: tuple[str, ...] = ()
    backend: str = "sparkinfer"
    quantization: str = ""
    sandbox: str = ""
    requires_gpu: bool = False
    supported_architectures: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("text",)
    context_limit: int = 131_072
    # Whether this module may be used to generate training data. Local student workers
    # are `approved` trivially; frontier teachers depend on their provider's terms.
    training_rights: str = RIGHTS_APPROVED
    budget: Budget = field(default_factory=Budget)
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.module_id or not self.model_id:
            raise ManifestError("agent module needs a module_id and a model_id")
        if self.training_rights not in (RIGHTS_APPROVED, RIGHTS_DENIED, RIGHTS_UNKNOWN):
            raise ManifestError(f"unknown training_rights {self.training_rights!r}")
        if self.requires_gpu and not self.supported_architectures:
            # A GPU worker that does not say which architectures it supports cannot be
            # eligibility-checked against a real machine, so it would pass filters it
            # should fail.
            raise ManifestError(f"{self.module_id}: requires_gpu needs supported_architectures")

    @property
    def may_generate_training_data(self) -> bool:
        return self.training_rights == RIGHTS_APPROVED

    def to_record(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "model_id": self.model_id,
            "domains": list(self.domains),
            "actions": list(self.actions),
            "tools": list(self.tools),
            "verifiers": list(self.verifiers),
            "backend": self.backend,
            "requires_gpu": self.requires_gpu,
            "supported_architectures": list(self.supported_architectures),
            "modalities": list(self.modalities),
            "context_limit": self.context_limit,
            "training_rights": self.training_rights,
            "budget": self.budget.to_record(),
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> AgentModule:
        raw_budget = record.get("budget") or {}
        return cls(
            module_id=str(record.get("module_id") or ""),
            model_id=str(record.get("model_id") or ""),
            domains=tuple(record.get("domains") or ()),
            actions=tuple(record.get("actions") or ()),
            tools=tuple(record.get("tools") or ()),
            verifiers=tuple(record.get("verifiers") or ()),
            backend=str(record.get("backend", "sparkinfer")),
            quantization=str(record.get("quantization", "")),
            sandbox=str(record.get("sandbox", "")),
            requires_gpu=bool(record.get("requires_gpu", False)),
            supported_architectures=tuple(record.get("supported_architectures") or ()),
            modalities=tuple(record.get("modalities") or ("text",)),
            context_limit=int(record.get("context_limit", 131_072)),
            training_rights=str(record.get("training_rights", RIGHTS_APPROVED)),
            budget=Budget(
                max_tool_calls=int(raw_budget.get("max_tool_calls", 100)),
                max_wall_time_s=int(raw_budget.get("max_wall_time_s", 14_400)),
                max_tokens=int(raw_budget.get("max_tokens", 0)),
            ),
            aliases=tuple(record.get("aliases") or ()),
            metadata=record.get("metadata") or {},
        )


class ModuleRegistry:
    """Alias -> agent module. Aliases exist so model upgrades are a registry edit."""

    def __init__(self, modules: list[AgentModule] | None = None) -> None:
        self._modules: dict[str, AgentModule] = {}
        self._aliases: dict[str, str] = {}
        for module in modules or []:
            self.register(module)

    def register(self, module: AgentModule) -> None:
        if module.module_id in self._modules:
            raise ManifestError(f"duplicate module_id {module.module_id!r}")
        self._modules[module.module_id] = module
        for alias in module.aliases:
            if alias in self._aliases:
                # Silently reassigning an alias would move production traffic to a
                # different worker with no diff to review.
                raise ManifestError(
                    f"alias {alias!r} already points at {self._aliases[alias]!r}; "
                    f"cannot repoint it to {module.module_id!r}"
                )
            self._aliases[alias] = module.module_id

    def resolve(self, name: str) -> AgentModule:
        """Look up by alias or module_id."""
        module_id = self._aliases.get(name, name)
        if module_id not in self._modules:
            raise ManifestError(f"unknown agent module or alias {name!r}")
        return self._modules[module_id]

    def all(self) -> tuple[AgentModule, ...]:
        return tuple(self._modules[k] for k in sorted(self._modules))

    def __len__(self) -> int:
        return len(self._modules)


@dataclass(frozen=True)
class HarnessPin:
    """What a trajectory was generated against.

    A release number alone is not a pin. The same tag with different tool schemas, a
    different system prompt or a different container produces different behaviour, so
    every digest that can change the agent's observations is recorded and bound into the
    SparkProof manifest alongside the trajectory.
    """

    release: str = ""
    tag: str = ""
    commit: str = ""
    trace_schema: str = "spark.hermes.v1"
    system_prompt_digest: str = ""
    tool_schema_digest: str = ""
    container_image_digest: str = ""
    dependency_lock_digest: str = ""
    # Set only once the conformance suite has passed against this exact pin. Trajectories
    # generated against an unconfirmed harness must not enter training: a harness that
    # mangles tool results teaches the student to expect mangled observations.
    conformance_verified: bool = False

    @property
    def is_pinned(self) -> bool:
        return bool(self.commit and self.tool_schema_digest and self.system_prompt_digest)

    def to_record(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "tag": self.tag,
            "commit": self.commit,
            "trace_schema": self.trace_schema,
            "system_prompt_digest": self.system_prompt_digest,
            "tool_schema_digest": self.tool_schema_digest,
            "container_image_digest": self.container_image_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "conformance_verified": self.conformance_verified,
        }

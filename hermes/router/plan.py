"""Eligibility filtering and the Phase-A deterministic route planner.

Two stages, in this order, and the order is the point:

1. **Hard filters.** Eliminate candidates that *cannot* do the job — missing tool,
   missing hardware, too little context, wrong modality, no training rights. A brilliant
   CUDA expert on a machine with no GPU is not a slightly worse route, it is not a route.
   Scoring first and filtering later would let a high capability score paper over an
   impossibility, and the failure would surface mid-episode instead of at selection.

2. **Ranked scoring.** Only survivors get ranked, by shrunk capability estimate.

Phase A is deliberately deterministic and legible. A learned router trained before the
schemas and logs are pinned down is one whose mistakes cannot be explained, and the
router is infrastructure sitting in front of every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hermes.router.capability import LOW_COVERAGE_ATTEMPTS, CapabilityDB
from hermes.router.manifest import AgentModule, ModuleRegistry
from hermes.router.spec import TaskSpec

# Why a candidate was excluded. Structured so a route can be audited without parsing
# prose, and so "nothing was eligible" is always explainable.
MISSING_TOOL = "missing_tool"
MISSING_GPU = "missing_gpu"
UNSUPPORTED_ARCH = "unsupported_arch"
CONTEXT_TOO_SMALL = "context_too_small"
UNSUPPORTED_MODALITY = "unsupported_modality"
DOMAIN_MISMATCH = "domain_mismatch"
ACTION_MISMATCH = "action_mismatch"
NO_TRAINING_RIGHTS = "no_training_rights"

# Routing modes, chosen by confidence rather than fixed per task.
MODE_SINGLE = "single"
MODE_DUAL = "dual"
MODE_ALL = "all_eligible"


@dataclass(frozen=True)
class Exclusion:
    module_id: str
    reason_code: str
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"module_id": self.module_id, "reason_code": self.reason_code, "detail": self.detail}


@dataclass(frozen=True)
class RouteDecision:
    """The full contract a route hands to the harness.

    Deliberately more than a model id: this is written verbatim into the trajectory, so
    a run can be reproduced and audited from the decision alone. `reason_codes` and
    `policy_version` exist so a route that later looks wrong can be explained rather than
    guessed at.
    """

    agent_module: str
    model: str
    toolsets: tuple[str, ...] = ()
    verifier: str = ""
    effort: str = "default"
    budget: dict[str, Any] = field(default_factory=dict)
    fallback_modules: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    excluded: tuple[Exclusion, ...] = ()
    mode: str = MODE_SINGLE
    confidence: float = 0.0
    policy_version: str = "spark-router-v0.1"

    def to_record(self) -> dict[str, Any]:
        return {
            "agent_module": self.agent_module,
            "model": self.model,
            "toolsets": list(self.toolsets),
            "verifier": self.verifier,
            "effort": self.effort,
            "budget": self.budget,
            "fallback_modules": list(self.fallback_modules),
            "reason_codes": list(self.reason_codes),
            "excluded": [e.to_record() for e in self.excluded],
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "policy_version": self.policy_version,
        }


class NoEligibleModule(RuntimeError):
    """Nothing could do the job. Carries why, per candidate."""

    def __init__(self, task_id: str, exclusions: list[Exclusion]) -> None:
        self.exclusions = exclusions
        detail = "; ".join(f"{e.module_id}:{e.reason_code}" for e in exclusions) or "no candidates registered"
        super().__init__(f"no eligible agent module for task {task_id!r} ({detail})")


def check_eligibility(
    task: TaskSpec,
    module: AgentModule,
    *,
    available_gpu_architectures: tuple[str, ...] = (),
    for_training_data: bool = False,
) -> Exclusion | None:
    """Return why `module` cannot serve `task`, or None if it can.

    `available_gpu_architectures` describes the machine the work would actually run on,
    not what the module wishes for. Passing it empty means "no GPU here", which is what
    makes a GPU-requiring route correctly ineligible on a CPU box.
    """
    if not (set(task.domain) & set(module.domains)):
        return Exclusion(module.module_id, DOMAIN_MISMATCH, f"needs {list(task.domain)}, has {list(module.domains)}")

    # Action, not just domain: the best model for writing a kernel need not be the best
    # for reviewing its numerical correctness.
    if module.actions and not (set(task.action) & set(module.actions)):
        return Exclusion(module.module_id, ACTION_MISMATCH, f"needs {list(task.action)}, has {list(module.actions)}")

    missing = set(task.required_tools) - set(module.tools)
    if missing:
        return Exclusion(module.module_id, MISSING_TOOL, f"missing {sorted(missing)}")

    if task.requires_gpu:
        if not module.requires_gpu:
            return Exclusion(module.module_id, MISSING_GPU, "task needs a GPU worker")
        if available_gpu_architectures and not (set(module.supported_architectures) & set(available_gpu_architectures)):
            return Exclusion(
                module.module_id,
                UNSUPPORTED_ARCH,
                f"supports {list(module.supported_architectures)}, host has {list(available_gpu_architectures)}",
            )
        if not available_gpu_architectures:
            return Exclusion(module.module_id, MISSING_GPU, "no GPU available on this host")

    if task.context_tokens_required > module.context_limit:
        return Exclusion(
            module.module_id,
            CONTEXT_TOO_SMALL,
            f"needs {task.context_tokens_required}, limit {module.context_limit}",
        )

    unsupported = set(task.modality) - set(module.modalities)
    if unsupported:
        return Exclusion(module.module_id, UNSUPPORTED_MODALITY, f"cannot handle {sorted(unsupported)}")

    # Fails closed: unknown rights are not approved rights.
    if for_training_data and not module.may_generate_training_data:
        return Exclusion(module.module_id, NO_TRAINING_RIGHTS, f"rights={module.training_rights}")

    return None


def eligible_modules(
    task: TaskSpec,
    registry: ModuleRegistry,
    *,
    available_gpu_architectures: tuple[str, ...] = (),
    for_training_data: bool = False,
) -> tuple[list[AgentModule], list[Exclusion]]:
    """Split the registry into (can do this task, cannot — with reasons)."""
    keep: list[AgentModule] = []
    dropped: list[Exclusion] = []
    for module in registry.all():
        exclusion = check_eligibility(
            task,
            module,
            available_gpu_architectures=available_gpu_architectures,
            for_training_data=for_training_data,
        )
        (dropped.append(exclusion) if exclusion else keep.append(module))
    return keep, dropped


def plan_route(
    task: TaskSpec,
    registry: ModuleRegistry,
    capabilities: CapabilityDB,
    *,
    available_gpu_architectures: tuple[str, ...] = (),
    for_training_data: bool = False,
    harness: str = "",
    effort: str = "default",
    min_margin: float = 0.12,
    policy_version: str = "spark-router-v0.1",
) -> RouteDecision:
    """Filter, then rank, then choose a routing mode.

    The mode is a consequence of the evidence rather than a per-task setting: a clear
    winner in a well-covered bucket is routed alone, a near-tie or a thinly-sampled
    bucket widens to two, and a bucket with no coverage at all runs everything eligible
    because that is the only way it acquires coverage.
    """
    survivors, dropped = eligible_modules(
        task,
        registry,
        available_gpu_architectures=available_gpu_architectures,
        for_training_data=for_training_data,
    )
    if not survivors:
        raise NoEligibleModule(task.task_id, dropped)

    ranked = capabilities.rank([m.module_id for m in survivors], task.bucket, harness=harness, effort=effort)
    by_id = {m.module_id: m for m in survivors}
    best_id, best_score = ranked[0]
    best = by_id[best_id]

    coverage = capabilities.coverage(best_id, task.bucket, harness=harness, effort=effort)
    margin = best_score - ranked[1][1] if len(ranked) > 1 else 1.0

    reasons: list[str] = []
    if coverage == 0:
        mode = MODE_ALL
        reasons.append("no_coverage_in_bucket")
    elif margin < min_margin or coverage < LOW_COVERAGE_ATTEMPTS:
        mode = MODE_DUAL
        reasons.append("close_margin" if margin < min_margin else "low_coverage")
    else:
        mode = MODE_SINGLE
        reasons.append("clear_winner")

    if not task.deterministically_verifiable:
        # Nothing can be proven here, so a second opinion is worth more than usual.
        reasons.append("no_deterministic_verifier")
        if mode == MODE_SINGLE:
            mode = MODE_DUAL

    fallbacks = tuple(module_id for module_id, _ in ranked[1:]) if mode != MODE_SINGLE else ()

    return RouteDecision(
        agent_module=best.module_id,
        model=best.model_id,
        toolsets=best.tools,
        verifier=best.verifiers[0] if best.verifiers else "",
        effort=effort,
        budget=best.budget.to_record(),
        fallback_modules=fallbacks,
        reason_codes=tuple(reasons),
        excluded=tuple(dropped),
        mode=mode,
        confidence=best_score,
        policy_version=policy_version,
    )

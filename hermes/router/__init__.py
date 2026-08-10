"""Phase 4 -- the Hermes router.

Dispatches a task to the specialist best suited to it, or to the generalist when it is
not sure. Routing is a much easier problem than solving, which is why a 3B-7B model can
do it; `KeywordRouter` is the free baseline that keeps "easier" from becoming "assumed".

See docs/roadmap-hermes.md Phase 4 and hermes/router/README.md.
"""

from hermes.router.base import (
    ESCALATABLE,
    TIER_KEYWORD,
    TIER_MODEL,
    Router,
    RoutingDecision,
    RoutingError,
    abstain,
)
from hermes.router.capability import CapabilityDB, CapabilityRecord
from hermes.router.cascade import CascadeRouter
from hermes.router.domains import ALL_TARGETS, DOMAINS, GENERAL, ROUTABLE, model_for
from hermes.router.evaluate import RouterMetrics, RoutingExample, compare, evaluate, load_suite
from hermes.router.keyword import KeywordRouter
from hermes.router.manifest import AgentModule, Budget, HarnessPin, ManifestError, ModuleRegistry
from hermes.router.model import ModelRouter, build_prompt, parse_decision
from hermes.router.plan import (
    NoEligibleModule,
    RouteDecision,
    check_eligibility,
    eligible_modules,
    plan_route,
)
from hermes.router.spec import TaskSpec, TaskSpecError

__all__ = [
    "ALL_TARGETS",
    "DOMAINS",
    "ESCALATABLE",
    "GENERAL",
    "ROUTABLE",
    "AgentModule",
    "Budget",
    "CapabilityDB",
    "CapabilityRecord",
    "CascadeRouter",
    "HarnessPin",
    "ManifestError",
    "ModuleRegistry",
    "NoEligibleModule",
    "RouteDecision",
    "TaskSpec",
    "TaskSpecError",
    "check_eligibility",
    "eligible_modules",
    "plan_route",
    "KeywordRouter",
    "ModelRouter",
    "Router",
    "RouterMetrics",
    "RoutingDecision",
    "RoutingError",
    "RoutingExample",
    "TIER_KEYWORD",
    "TIER_MODEL",
    "abstain",
    "build_prompt",
    "compare",
    "evaluate",
    "load_suite",
    "model_for",
    "parse_decision",
]

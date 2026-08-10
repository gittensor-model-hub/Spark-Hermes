"""SparkRouter Phase A: schemas, eligibility, capability shrinkage, route planning."""

import json

import pytest

from hermes.router import (
    AgentModule,
    CapabilityDB,
    CapabilityRecord,
    HarnessPin,
    ManifestError,
    ModuleRegistry,
    NoEligibleModule,
    TaskSpec,
    TaskSpecError,
    check_eligibility,
    plan_route,
)
from hermes.router.manifest import RIGHTS_UNKNOWN, Budget
from hermes.router.plan import (
    ACTION_MISMATCH,
    CONTEXT_TOO_SMALL,
    MISSING_GPU,
    MISSING_TOOL,
    MODE_ALL,
    MODE_DUAL,
    MODE_SINGLE,
    NO_TRAINING_RIGHTS,
    UNSUPPORTED_ARCH,
    UNSUPPORTED_MODALITY,
)

BLACKWELL = ("sm_120",)


def _cuda_module(**overrides) -> AgentModule:
    record = {
        "module_id": "spark-cuda-worker-v1",
        "model_id": "spark-hermes-cuda-3.8-27b",
        "domains": ["cuda"],
        "actions": ["profile", "optimize", "debug", "review"],
        "tools": ["terminal", "python", "nvcc", "ncu"],
        "verifiers": ["numerical_equivalence", "benchmark"],
        "requires_gpu": True,
        "supported_architectures": ["sm_120", "sm_121"],
        "context_limit": 262_144,
        "aliases": ["expert.cuda.optimization"],
    }
    record.update(overrides)
    return AgentModule.from_record(record)


def _swe_module(**overrides) -> AgentModule:
    record = {
        "module_id": "spark-swe-worker-v1",
        "model_id": "spark-hermes-swe-3.8-27b",
        "domains": ["swe"],
        "actions": ["implement", "debug", "review"],
        "tools": ["terminal", "python", "git"],
        "verifiers": ["unit_tests"],
        "aliases": ["expert.swe.repository"],
    }
    record.update(overrides)
    return AgentModule.from_record(record)


def _cuda_task(**overrides) -> TaskSpec:
    record = {
        "task_id": "cuda_000184",
        "prompt": "Profile and optimize the attention kernel",
        "domain": ["cuda"],
        "action": ["optimize"],
        "horizon": "long",
        "required_tools": ["terminal", "ncu"],
        "required_environments": ["gpu"],
        "verification": "benchmark",
    }
    record.update(overrides)
    return TaskSpec.from_record(record)


# --- TaskSpec ----------------------------------------------------------------------


def test_unknown_domain_is_rejected():
    with pytest.raises(TaskSpecError, match="unknown domain"):
        TaskSpec(task_id="t", prompt="p", domain=("astrology",), action=("implement",))


def test_unknown_action_is_rejected():
    with pytest.raises(TaskSpecError, match="unknown action"):
        TaskSpec(task_id="t", prompt="p", domain=("swe",), action=("vibe",))


def test_bucket_is_domain_action_horizon():
    assert _cuda_task().bucket == "cuda.optimize.long"


def test_deterministic_verification_is_distinguished_from_judging():
    assert _cuda_task(verification="benchmark").deterministically_verifiable
    assert not _cuda_task(verification="judge_only").deterministically_verifiable


def test_task_record_round_trips():
    task = _cuda_task()
    assert TaskSpec.from_record(task.to_record() | {"prompt": task.prompt}).bucket == task.bucket


# --- manifests ---------------------------------------------------------------------


def test_gpu_module_must_declare_architectures():
    """Otherwise it cannot be eligibility-checked and would pass filters it should fail."""
    with pytest.raises(ManifestError, match="supported_architectures"):
        AgentModule(module_id="m", model_id="x", domains=("cuda",), actions=("optimize",), requires_gpu=True)


def test_registry_resolves_by_alias_and_by_id():
    registry = ModuleRegistry([_cuda_module()])
    assert registry.resolve("expert.cuda.optimization").module_id == "spark-cuda-worker-v1"
    assert registry.resolve("spark-cuda-worker-v1").model_id == "spark-hermes-cuda-3.8-27b"


def test_alias_cannot_be_silently_repointed():
    """Repointing an alias would move production traffic with no diff to review."""
    registry = ModuleRegistry([_cuda_module()])
    with pytest.raises(ManifestError, match="already points at"):
        registry.register(_cuda_module(module_id="other", aliases=["expert.cuda.optimization"]))


def test_duplicate_module_id_is_rejected():
    registry = ModuleRegistry([_cuda_module()])
    with pytest.raises(ManifestError, match="duplicate module_id"):
        registry.register(_cuda_module(aliases=[]))


def test_unknown_alias_raises():
    with pytest.raises(ManifestError, match="unknown agent module"):
        ModuleRegistry().resolve("expert.nope")


def test_harness_pin_needs_more_than_a_release_number():
    assert not HarnessPin(release="v0.20.0", tag="v2026.8.3").is_pinned
    assert HarnessPin(commit="abc", tool_schema_digest="d1", system_prompt_digest="d2").is_pinned


def test_harness_conformance_defaults_to_unverified():
    """Trajectories from an unconfirmed harness must not silently become training data."""
    assert HarnessPin(commit="abc").conformance_verified is False


# --- eligibility (hard filters) ----------------------------------------------------


def test_cuda_expert_without_a_gpu_host_is_not_eligible():
    """A brilliant CUDA expert on a CPU box is not a worse route -- it is not a route."""
    exclusion = check_eligibility(_cuda_task(), _cuda_module(), available_gpu_architectures=())
    assert exclusion is not None
    assert exclusion.reason_code == MISSING_GPU


def test_cuda_expert_with_a_matching_gpu_is_eligible():
    assert check_eligibility(_cuda_task(), _cuda_module(), available_gpu_architectures=BLACKWELL) is None


def test_wrong_gpu_architecture_is_not_eligible():
    exclusion = check_eligibility(_cuda_task(), _cuda_module(), available_gpu_architectures=("sm_90",))
    assert exclusion.reason_code == UNSUPPORTED_ARCH


def test_missing_tool_blocks_the_route():
    exclusion = check_eligibility(_cuda_task(), _cuda_module(tools=["terminal"]), available_gpu_architectures=BLACKWELL)
    assert exclusion.reason_code == MISSING_TOOL
    assert "ncu" in exclusion.detail


def test_action_mismatch_blocks_the_route():
    """The best model for writing a kernel need not be the best for reviewing it."""
    exclusion = check_eligibility(
        _cuda_task(action=["review"]),
        _cuda_module(actions=["optimize"]),
        available_gpu_architectures=BLACKWELL,
    )
    assert exclusion.reason_code == ACTION_MISMATCH


def test_context_limit_blocks_the_route():
    exclusion = check_eligibility(
        _cuda_task(context_tokens_required=999_999), _cuda_module(), available_gpu_architectures=BLACKWELL
    )
    assert exclusion.reason_code == CONTEXT_TOO_SMALL


def test_modality_blocks_the_route():
    exclusion = check_eligibility(
        _cuda_task(modality=["text", "image"]), _cuda_module(), available_gpu_architectures=BLACKWELL
    )
    assert exclusion.reason_code == UNSUPPORTED_MODALITY


def test_unknown_training_rights_fail_closed():
    """'We were not sure' is not a defence; unknown rights are not approved rights."""
    exclusion = check_eligibility(
        _cuda_task(),
        _cuda_module(training_rights=RIGHTS_UNKNOWN),
        available_gpu_architectures=BLACKWELL,
        for_training_data=True,
    )
    assert exclusion.reason_code == NO_TRAINING_RIGHTS


def test_rights_are_only_checked_for_training_data():
    assert (
        check_eligibility(
            _cuda_task(),
            _cuda_module(training_rights=RIGHTS_UNKNOWN),
            available_gpu_architectures=BLACKWELL,
            for_training_data=False,
        )
        is None
    )


# --- capability shrinkage ----------------------------------------------------------


def test_sparse_perfect_record_does_not_outrank_a_well_sampled_one():
    """2/2 must not beat 800/1000, or the router chases noise forever."""
    sparse = CapabilityRecord(model="new", bucket="b", attempts=2, verified_successes=2)
    proven = CapabilityRecord(model="old", bucket="b", attempts=1000, verified_successes=800)
    assert sparse.raw_success > proven.raw_success
    assert sparse.posterior_success() < proven.posterior_success()


def test_uncertainty_shrinks_as_evidence_accumulates():
    thin = CapabilityRecord(model="m", bucket="b", attempts=3, verified_successes=2)
    thick = CapabilityRecord(model="m", bucket="b", attempts=900, verified_successes=600)
    assert thin.posterior_stddev() > thick.posterior_stddev()


def test_successes_cannot_exceed_attempts():
    with pytest.raises(ValueError, match="more successes than attempts"):
        CapabilityRecord(model="m", bucket="b", attempts=1, verified_successes=5)


def test_unmeasured_pairing_scores_the_prior_not_zero():
    """Never having been tried is not evidence of failure."""
    assert CapabilityDB().estimate("brand-new", "cuda.optimize.long") == 0.5


def test_model_versions_keep_separate_histories():
    """Merging them lets a regression hide behind the old build's record."""
    db = CapabilityDB(
        [
            CapabilityRecord(model="qwen-2026-08", bucket="b", attempts=100, verified_successes=90),
            CapabilityRecord(model="qwen-2026-09", bucket="b", attempts=100, verified_successes=40),
        ]
    )
    assert db.estimate("qwen-2026-08", "b") > db.estimate("qwen-2026-09", "b")
    assert len(db) == 2


def test_ranking_is_deterministic_on_ties():
    db = CapabilityDB()
    assert [m for m, _ in db.rank(["b", "a", "c"], "bucket")] == ["a", "b", "c"]


def test_capability_db_round_trips_through_disk(tmp_path):
    db = CapabilityDB([CapabilityRecord(model="m", bucket="b", attempts=10, verified_successes=7, harness="h")])
    path = tmp_path / "caps.jsonl"
    db.save(path)
    assert CapabilityDB.load(path).estimate("m", "b", harness="h") == db.estimate("m", "b", harness="h")


def test_malformed_capability_row_reports_its_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"model":"m","bucket":"b"}\n{ not json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        CapabilityDB.load(path)


# --- route planning ----------------------------------------------------------------


def _registry() -> ModuleRegistry:
    return ModuleRegistry([_cuda_module(), _swe_module()])


def test_route_selects_the_eligible_expert_and_reports_the_rest():
    db = CapabilityDB(
        [
            CapabilityRecord(
                model="spark-cuda-worker-v1", bucket="cuda.optimize.long", attempts=200, verified_successes=170
            )
        ]
    )
    decision = plan_route(_cuda_task(), _registry(), db, available_gpu_architectures=BLACKWELL)

    assert decision.agent_module == "spark-cuda-worker-v1"
    assert decision.model == "spark-hermes-cuda-3.8-27b"
    assert decision.mode == MODE_SINGLE
    assert "ncu" in decision.toolsets
    assert decision.verifier == "numerical_equivalence"
    # The SWE worker was considered and rejected, and the record says why.
    assert any(e.module_id == "spark-swe-worker-v1" for e in decision.excluded)


def test_route_carries_the_budget_and_policy_version():
    db = CapabilityDB(
        [
            CapabilityRecord(
                model="spark-cuda-worker-v1", bucket="cuda.optimize.long", attempts=200, verified_successes=170
            )
        ]
    )
    decision = plan_route(
        _cuda_task(),
        ModuleRegistry([_cuda_module(budget={"max_tool_calls": 100, "max_wall_time_s": 14400})]),
        db,
        available_gpu_architectures=BLACKWELL,
    )
    assert decision.budget["max_tool_calls"] == 100
    assert decision.policy_version == "spark-router-v0.1"


def test_no_coverage_runs_every_eligible_candidate():
    """A bucket with no evidence only acquires evidence by being explored."""
    registry = ModuleRegistry([_cuda_module(), _cuda_module(module_id="alt", aliases=[])])
    decision = plan_route(_cuda_task(), registry, CapabilityDB(), available_gpu_architectures=BLACKWELL)
    assert decision.mode == MODE_ALL
    assert "no_coverage_in_bucket" in decision.reason_codes


def test_low_coverage_widens_to_two():
    db = CapabilityDB(
        [CapabilityRecord(model="spark-cuda-worker-v1", bucket="cuda.optimize.long", attempts=5, verified_successes=5)]
    )
    registry = ModuleRegistry([_cuda_module(), _cuda_module(module_id="alt", aliases=[])])
    decision = plan_route(_cuda_task(), registry, db, available_gpu_architectures=BLACKWELL)
    assert decision.mode == MODE_DUAL
    assert decision.fallback_modules


def test_a_task_with_no_deterministic_verifier_widens_to_two():
    """Nothing can be proven, so a second opinion is worth more than usual."""
    db = CapabilityDB(
        [
            CapabilityRecord(
                model="spark-swe-worker-v1", bucket="swe.review.medium", attempts=300, verified_successes=250
            )
        ]
    )
    task = TaskSpec.from_record(
        {
            "task_id": "t",
            "prompt": "review this design",
            "domain": ["swe"],
            "action": ["review"],
            "horizon": "medium",
            "verification": "judge_only",
        }
    )
    decision = plan_route(task, _registry(), db, available_gpu_architectures=())
    assert decision.mode == MODE_DUAL
    assert "no_deterministic_verifier" in decision.reason_codes


def test_no_eligible_module_raises_with_the_reasons():
    with pytest.raises(NoEligibleModule) as excinfo:
        plan_route(_cuda_task(), _registry(), CapabilityDB(), available_gpu_architectures=())
    assert any(e.reason_code == MISSING_GPU for e in excinfo.value.exclusions)


def test_decision_record_is_json_safe():
    db = CapabilityDB(
        [
            CapabilityRecord(
                model="spark-cuda-worker-v1", bucket="cuda.optimize.long", attempts=200, verified_successes=170
            )
        ]
    )
    decision = plan_route(_cuda_task(), _registry(), db, available_gpu_architectures=BLACKWELL)
    assert json.loads(json.dumps(decision.to_record()))["agent_module"] == "spark-cuda-worker-v1"


def test_budget_defaults_are_sane():
    assert Budget().max_tool_calls > 0 and Budget().max_wall_time_s > 0

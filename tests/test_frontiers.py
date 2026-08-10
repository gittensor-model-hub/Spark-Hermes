import json
from pathlib import Path

from eval import frontiers as frontiers_mod
from eval.frontiers import (
    FRONTIERS_PATH,
    apply_verified_report_to_frontiers,
    candidate_scores_from_report,
    load_frontier_record,
    load_frontier_scores,
    load_frontiers,
    merge_frontier_record,
)


def test_load_frontiers_from_repo_file():
    frontiers = load_frontiers(FRONTIERS_PATH)
    assert set(frontiers) == {"blackwell", "hopper"}
    assert frontiers["blackwell"]["scores"]["triton"] > 0
    assert frontiers["hopper"]["run_id"] == "2026-07-15-magicrails-hopper-v2"
    assert frontiers["hopper"]["scores"]["gsm8k"] == 0.74


def test_load_frontier_scores_hopper_seeded():
    scores = load_frontier_scores("hopper", path=FRONTIERS_PATH)
    assert scores is not None
    assert scores["triton"] == 0.3719444444444444


def test_load_frontier_scores_blackwell_has_triton():
    scores = load_frontier_scores("blackwell", path=FRONTIERS_PATH)
    assert scores is not None
    assert scores["gsm8k"] == 0.6


def test_legacy_frontier_json_seeds_blackwell_only(tmp_path: Path):
    legacy = {
        "run_id": "legacy-run",
        "proof_bundle": "https://example.com/bundle",
        "scores": {"gsm8k": 0.55, "triton": 0.40},
    }
    frontiers_path = tmp_path / "frontiers.json"
    legacy_path = tmp_path / "frontier.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    original_legacy = frontiers_mod.LEGACY_FRONTIER_PATH
    original_frontiers = frontiers_mod.FRONTIERS_PATH
    try:
        frontiers_mod.LEGACY_FRONTIER_PATH = legacy_path
        frontiers_mod.FRONTIERS_PATH = frontiers_path
        loaded = load_frontiers(frontiers_path)
    finally:
        frontiers_mod.LEGACY_FRONTIER_PATH = original_legacy
        frontiers_mod.FRONTIERS_PATH = original_frontiers

    assert loaded["blackwell"]["run_id"] == "legacy-run"
    assert loaded["blackwell"]["scores"]["triton"] == 0.40
    assert loaded["hopper"]["scores"] == {}


def test_merge_frontier_record_updates_arch_bucket_only():
    frontiers = load_frontiers(FRONTIERS_PATH)
    updated, updates = merge_frontier_record(
        frontiers,
        "hopper",
        {"gsm8k": 0.8, "triton": 0.5},
        run_id="hopper-improve-001",
        proof_bundle="https://example.com/hopper",
    )
    assert "gsm8k" in updates and "triton" in updates
    assert updated["hopper"]["scores"]["triton"] == 0.5
    assert updated["blackwell"]["scores"]["triton"] == frontiers["blackwell"]["scores"]["triton"]


def test_load_frontier_record_preserves_metadata():
    record = load_frontier_record("blackwell", path=FRONTIERS_PATH)
    assert record["gpu_architecture"] == "blackwell"
    # Canary on the live blackwell frontier holder — update when it is (re)crowned.
    assert record["run_id"] == "2026-07-29-philluiz-blackwell-xl-v1"


def test_apply_verified_report_seeds_empty_bucket(tmp_path: Path):
    frontiers_path = tmp_path / "frontiers.json"
    frontiers_path.write_text(
        json.dumps(
            {
                "blackwell": {"gpu_architecture": "blackwell", "run_id": None, "proof_bundle": None, "scores": {}},
                "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "run_id": "hopper-baseline",
        "gpu_architecture": "hopper",
        "per_benchmark": {
            "gsm8k": {"candidate": 0.74, "frontier": None},
            "triton": {"candidate": 0.37, "frontier": None},
            "triton_syntax_pass_rate": {"candidate": 0.66, "frontier": None},
        },
    }
    updates = apply_verified_report_to_frontiers(
        report,
        proof_bundle="https://huggingface.co/org/hopper-proof",
        path=frontiers_path,
    )
    assert "gsm8k" in updates and "triton" in updates
    loaded = json.loads(frontiers_path.read_text(encoding="utf-8"))
    assert loaded["hopper"]["run_id"] == "hopper-baseline"
    assert loaded["hopper"]["scores"]["gsm8k"] == 0.74
    assert loaded["hopper"]["scores"]["triton_syntax_pass_rate"] == 0.66
    legacy = json.loads((tmp_path / "frontier.json").read_text(encoding="utf-8"))
    assert legacy["scores"] == {}


def test_apply_verified_report_skips_reject(tmp_path: Path):
    frontiers_path = tmp_path / "frontiers.json"
    frontiers_path.write_text(
        json.dumps(
            {
                "blackwell": {
                    "gpu_architecture": "blackwell",
                    "run_id": "b1",
                    "proof_bundle": None,
                    "scores": {"triton": 0.4},
                },
                "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
            }
        ),
        encoding="utf-8",
    )
    before = frontiers_path.read_text(encoding="utf-8")
    updates = apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:REJECT",
            "run_id": "bad",
            "gpu_architecture": "blackwell",
            "per_benchmark": {"triton": {"candidate": 0.9, "frontier": 0.4}},
        },
        proof_bundle="https://example.com/x",
        path=frontiers_path,
    )
    assert updates == []
    assert frontiers_path.read_text(encoding="utf-8") == before


def test_candidate_scores_from_report():
    assert candidate_scores_from_report(
        {"per_benchmark": {"gsm8k": {"candidate": 0.5}, "triton": {"candidate": 0.2}}}
    ) == {"gsm8k": 0.5, "triton": 0.2}


def test_candidate_scores_prefers_full_claim_over_per_benchmark():
    """`per_benchmark` is the candidate-vs-frontier projection, not the full claim."""
    report = {
        "scores": {"gsm8k": 0.5, "triton": 0.2, "humaneval": 0.6},
        "per_benchmark": {"gsm8k": {"candidate": 0.5}, "triton": {"candidate": 0.2}},
    }
    assert candidate_scores_from_report(report) == {"gsm8k": 0.5, "triton": 0.2, "humaneval": 0.6}


def test_apply_verified_report_adds_benchmark_missing_from_frontier(tmp_path: Path):
    """A benchmark the frontier does not carry yet must still raise it (issue #233).

    `eval.score` omits such a key from `per_benchmark`, and a benchmark absent
    from the frontier is never regression-guarded — so dropping it here leaves it
    unguarded permanently.
    """
    frontiers_path = tmp_path / "frontiers.json"
    frontiers_path.write_text(
        json.dumps(
            {
                "blackwell": {"gpu_architecture": "blackwell", "run_id": None, "proof_bundle": None, "scores": {}},
                "hopper": {
                    "gpu_architecture": "hopper",
                    "run_id": "hopper-1",
                    "proof_bundle": "https://example.com/1",
                    "scores": {"triton": 0.30, "gsm8k": 0.60},
                },
            }
        ),
        encoding="utf-8",
    )
    report = {
        "verified": True,
        "label": "eval:XL",
        "run_id": "hopper-2",
        "gpu_architecture": "hopper",
        # what eval.score emits: only keys in both candidate and frontier
        "per_benchmark": {
            "triton": {"candidate": 0.40, "frontier": 0.30},
            "gsm8k": {"candidate": 0.62, "frontier": 0.60},
        },
        # the full claim, including a benchmark the frontier has never carried
        "scores": {"triton": 0.40, "gsm8k": 0.62, "humaneval": 0.55, "triton_syntax_pass_rate": 0.9},
    }

    updates = apply_verified_report_to_frontiers(
        report,
        proof_bundle="https://huggingface.co/org/hopper-2",
        path=frontiers_path,
    )
    assert "humaneval" in updates
    scores = json.loads(frontiers_path.read_text(encoding="utf-8"))["hopper"]["scores"]
    assert scores["humaneval"] == 0.55
    # diagnostic breakdown keys keep seeding alongside the basket, not just at BASELINE
    assert scores["triton_syntax_pass_rate"] == 0.9


def test_frontier_bucket_and_track_helpers():
    from eval.frontiers import frontier_bucket, training_track_of

    assert frontier_bucket("blackwell") == "blackwell"
    assert frontier_bucket("blackwell", "sft") == "blackwell"
    assert frontier_bucket("blackwell", "dpo") == "blackwell::dpo"
    assert training_track_of({}) == "sft"
    assert training_track_of({"train_objective": "dpo"}) == "dpo"
    assert training_track_of({"train_objective": "DPO"}) == "dpo"
    assert training_track_of({"train_objective": "sft"}) == "sft"


def test_first_dpo_run_hits_empty_bucket_baseline_signal(tmp_path: Path):
    path = tmp_path / "frontiers.json"
    frontiers_mod.write_frontiers(
        {
            "blackwell": {
                "gpu_architecture": "blackwell",
                "run_id": "sft-king",
                "proof_bundle": "b",
                "scores": {"triton": 0.43, "gsm8k": 0.6},
            }
        },
        path=path,
    )
    # SFT frontier is seeded, but the DPO bucket has never been touched -> None (=> BASELINE).
    assert load_frontier_scores("blackwell", path=path) is not None
    assert load_frontier_scores("blackwell", track="dpo", path=path) is None


def test_dpo_report_seeds_independent_bucket_without_touching_sft(tmp_path: Path):
    path = tmp_path / "frontiers.json"
    frontiers_mod.write_frontiers(
        {
            "blackwell": {
                "gpu_architecture": "blackwell",
                "run_id": "sft-king",
                "proof_bundle": "b",
                "scores": {"triton": 0.43, "gsm8k": 0.6},
            }
        },
        path=path,
    )
    apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:BASELINE",
            "gpu_architecture": "blackwell",
            "train_objective": "dpo",
            "run_id": "dpo-seed",
            "scores": {"triton": 0.30, "gsm8k": 0.55},
        },
        proof_bundle="dpo-bundle",
        path=path,
    )
    # DPO frontier now exists, in its own bucket.
    assert load_frontier_scores("blackwell", track="dpo", path=path)["triton"] == 0.30
    # SFT frontier is byte-identical / untouched.
    assert load_frontier_scores("blackwell", path=path)["triton"] == 0.43
    data = json.loads(path.read_text())
    assert data["blackwell"]["run_id"] == "sft-king"
    assert data["blackwell::dpo"]["run_id"] == "dpo-seed"
    assert data["blackwell::dpo"]["gpu_architecture"] == "blackwell"


def test_second_dpo_run_tiers_over_dpo_frontier(tmp_path: Path):
    path = tmp_path / "frontiers.json"
    # Pre-seed so `path` exists before the first apply — otherwise load_frontiers's
    # legacy-file fallback resolves the repo-relative LEGACY_FRONTIER_PATH against the
    # real runs/frontier.json (this test's cwd is the repo root), leaking live frontier
    # state into what should be an isolated tmp fixture.
    frontiers_mod.write_frontiers(
        {
            "blackwell": {"gpu_architecture": "blackwell", "run_id": None, "proof_bundle": None, "scores": {}},
            "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
        },
        path=path,
    )
    apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:BASELINE",
            "gpu_architecture": "blackwell",
            "train_objective": "dpo",
            "run_id": "dpo-1",
            "scores": {"triton": 0.30},
        },
        proof_bundle="b1",
        path=path,
    )
    updates = apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:M",
            "gpu_architecture": "blackwell",
            "train_objective": "dpo",
            "run_id": "dpo-2",
            "scores": {"triton": 0.40},
        },
        proof_bundle="b2",
        path=path,
    )
    assert "triton" in updates
    assert load_frontier_scores("blackwell", track="dpo", path=path)["triton"] == 0.40


def test_sft_and_dpo_frontiers_do_not_cross(tmp_path: Path):
    path = tmp_path / "frontiers.json"
    # Pre-seed so `path` exists before the first apply — see comment in
    # test_second_dpo_run_tiers_over_dpo_frontier for why this matters: without it,
    # the SFT bucket silently inherits the real repo's live blackwell frontier via
    # LEGACY_FRONTIER_PATH, and the merge-only-raises rule then keeps whichever is
    # higher — masking this test's own "sft-1" value once the real frontier passes it.
    frontiers_mod.write_frontiers(
        {
            "blackwell": {"gpu_architecture": "blackwell", "run_id": None, "proof_bundle": None, "scores": {}},
            "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
        },
        path=path,
    )
    apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:BASELINE",
            "gpu_architecture": "blackwell",
            "train_objective": "dpo",
            "run_id": "dpo-1",
            "scores": {"triton": 0.30},
        },
        proof_bundle="b1",
        path=path,
    )
    apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:BASELINE",
            "gpu_architecture": "blackwell",
            "run_id": "sft-1",
            "scores": {"triton": 0.45},
        },  # no train_objective -> sft
        proof_bundle="b2",
        path=path,
    )
    assert load_frontier_scores("blackwell", path=path)["triton"] == 0.45  # sft
    assert load_frontier_scores("blackwell", track="dpo", path=path)["triton"] == 0.30  # dpo unchanged

"""Normalized GPU architecture families for eval frontiers and dataset provenance."""

from __future__ import annotations

import json
from typing import Literal

GpuArchitecture = Literal["blackwell", "hopper"]
GPU_ARCHITECTURES: tuple[GpuArchitecture, ...] = ("blackwell", "hopper")
DEFAULT_GPU_ARCHITECTURE: GpuArchitecture = "blackwell"

_BLACKWELL_TOKENS = ("rtx pro 6000", "gb20", "b200", "b300", "gb200", "gb102", "blackwell")
_HOPPER_TOKENS = ("h100", "h200", "gh100", "gh200", "hopper")


def normalize_gpu_architecture(value: str | None) -> GpuArchitecture | None:
    """Map free-text GPU claims to a supported architecture family."""
    if value is None:
        return None
    blob = str(value).strip().lower()
    if not blob:
        return None
    if blob in GPU_ARCHITECTURES:
        return blob  # type: ignore[return-value]
    if any(token in blob for token in _HOPPER_TOKENS):
        return "hopper"
    if any(token in blob for token in _BLACKWELL_TOKENS):
        return "blackwell"
    return None


def infer_gpu_architecture_from_attestation(attestation: dict | None) -> GpuArchitecture | None:
    if not attestation:
        return None
    return normalize_gpu_architecture(json.dumps(attestation.get("claims") or {}))


def tier_benchmark_for_arch(arch: GpuArchitecture) -> str:
    """Both Blackwell and Hopper tier the reward label on TritonBench.

    Kept as a per-arch function (rather than a bare constant) because each
    architecture keeps its own frontier bucket (see `eval.frontiers`) — absolute
    TritonBench composites are hardware-sensitive and are never compared across
    architectures, only within one.
    """
    return "triton"


def dataset_architecture_allowed(arch: GpuArchitecture) -> bool:
    """SparkProof accepts dataset generation on both Blackwell and Hopper H100/H200 CC nodes."""
    return arch in GPU_ARCHITECTURES

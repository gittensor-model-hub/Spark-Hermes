"""The specialist taxonomy the router dispatches across.

One entry per Phase 3 worker, plus `general` as the deliberate fallback. `general` is
not a specialist that failed to be identified -- it is the correct answer whenever the
router is not confident, because the cost of the two mistakes is not symmetric:

- routing a CUDA task to `general` costs some quality; the generalist still tries, and
  its answer is checked by the same verification loop as anyone's.
- routing a firmware task to `cyber` puts a model that is *confident in the wrong
  domain* on the job. Specialists are tuned to act, and a specialist acting on a task
  outside its training is the expensive failure.

So the router is built to abstain rather than guess. See docs/roadmap-hermes.md Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GENERAL = "general"


@dataclass(frozen=True)
class Domain:
    """A routable specialist."""

    key: str
    model: str
    description: str
    # Terms that indicate this domain. Weighted: a term that *only* appears in one
    # domain ("cutlass") discriminates far better than one that appears everywhere
    # ("optimize"), and a flat bag of words lets the common terms drown the rare ones.
    strong_terms: tuple[str, ...] = ()
    weak_terms: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


DOMAINS: dict[str, Domain] = {
    "cuda": Domain(
        key="cuda",
        model="Spark-Hermes-CUDA",
        description="GPU kernels, profiling, CUDA/Triton performance work",
        strong_terms=(
            "cuda",
            "triton",
            "cutlass",
            "tensorrt",
            "nsight",
            "kernel",
            "warp",
            "cudnn",
            "cublas",
            "ptx",
            "shared memory",
            "coalesc",
            "tensor core",
            "occupancy",
            "gpu",
        ),
        weak_terms=("throughput", "latency", "optimize", "profile", "benchmark", "speedup", "memory bandwidth"),
        tags=("gpu", "performance"),
    ),
    "firmware": Domain(
        key="firmware",
        model="Spark-Hermes-Firmware",
        description="Embedded systems, drivers, RTOS, bare-metal work",
        strong_terms=(
            "esp32",
            "stm32",
            "zephyr",
            "freertos",
            "u-boot",
            "uart",
            "i2c",
            "spi",
            "gpio",
            "firmware",
            "bootloader",
            "device tree",
            "interrupt handler",
            "baremetal",
            "bare-metal",
            "ota update",
            "kernel module",
            "device driver",
        ),
        weak_terms=("driver", "embedded", "flash", "register", "peripheral", "watchdog"),
        tags=("embedded",),
    ),
    "cyber": Domain(
        key="cyber",
        model="Spark-Hermes-Cyber",
        description="Vulnerability research, exploitation, patch verification, fuzzing",
        strong_terms=(
            "cve",
            "vulnerability",
            "exploit",
            # `fuzz`, `fuzzing` and `fuzzer` are separate entries on purpose: term
            # matching is word-bounded, so `fuzz` does not match inside `fuzzing`, and
            # listing only the longer forms misses "fuzz the parser" entirely.
            "fuzz",
            "fuzzing",
            "fuzzer",
            "asan",
            "sanitizer",
            "use-after-free",
            "buffer overflow",
            "heap overflow",
            "privilege escalation",
            "proof of concept",
            "poc",
            "attack surface",
            "memory corruption",
            "sql injection",
        ),
        weak_terms=("security", "patch", "crash", "audit", "hardening", "untrusted"),
        tags=("security",),
    ),
    "swe": Domain(
        key="swe",
        model="Spark-Hermes-SWE",
        description="Repository work: issues, tests, refactors, pull requests",
        strong_terms=(
            "pull request",
            "failing test",
            "unit test",
            "regression",
            "refactor",
            "merge conflict",
            "git bisect",
            "changelog",
            "pytest",
            "code review",
            "issue tracker",
            "stack trace",
            "traceback",
            "dependency upgrade",
            "type error",
        ),
        weak_terms=("repository", "commit", "branch", "test", "bug", "function", "module", "import"),
        tags=("software",),
    ),
}

ROUTABLE = tuple(DOMAINS)
ALL_TARGETS = (*ROUTABLE, GENERAL)


def is_valid_target(target: str) -> bool:
    return target in ALL_TARGETS


def model_for(target: str) -> str:
    """The worker that serves a routing target."""
    if target == GENERAL:
        return "Spark-Hermes-Glimmer-30B"
    return DOMAINS[target].model

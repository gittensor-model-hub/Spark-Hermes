"""Measured capability per (worker, bucket) -- with shrinkage, because raw rates lie.

A model with 2 verified successes from 2 attempts has a raw success rate of 1.00 and one
with 800 from 1000 has 0.80. Ranking on the raw rate routes everything to the model we
know almost nothing about. Every estimate here is therefore a Beta posterior mean, which
pulls sparse evidence toward the prior and only lets a candidate climb once it has earned
the samples:

    posterior = (successes + alpha) / (attempts + alpha + beta)

The uncertainty is kept alongside the estimate rather than discarded, because the router
needs it for two separate decisions: whether to consult a second teacher, and whether a
bucket is explored enough to trust at all.

Public benchmark numbers are priors, not entries. Models are published under different
agent harnesses, so a score earned under someone else's scaffold says little about
behaviour under ours -- which is the whole reason `HarnessPin` exists. A capability
record is only comparable to another record measured under the same pin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Beta(1, 1) -- uniform. Deliberately weak: a strong prior would take real evidence to
# overcome, and the point of calibration is to let evidence win quickly.
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0

# Below this many attempts a bucket is "low coverage": usable for ranking, but the router
# should widen to a second candidate rather than commit.
LOW_COVERAGE_ATTEMPTS = 20


@dataclass(frozen=True)
class CapabilityRecord:
    """Measured outcomes for one worker in one bucket, under one harness pin."""

    model: str
    bucket: str
    harness: str = ""
    model_version: str = ""
    effort: str = "default"
    attempts: int = 0
    verified_successes: int = 0
    tool_call_validity: float = 0.0
    recovery_rate: float = 0.0
    median_tokens: int = 0
    median_runtime_s: float = 0.0
    estimated_cost: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verified_successes > self.attempts:
            raise ValueError(f"{self.model}/{self.bucket}: more successes than attempts")

    def posterior_success(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> float:
        """Shrunk success estimate. 2/2 must not outrank 800/1000."""
        return (self.verified_successes + alpha) / (self.attempts + alpha + beta)

    def posterior_stddev(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> float:
        """Spread of the posterior -- how much the estimate should be trusted."""
        a = self.verified_successes + alpha
        b = (self.attempts - self.verified_successes) + beta
        total = a + b
        return math.sqrt((a * b) / (total * total * (total + 1)))

    @property
    def raw_success(self) -> float:
        """Unshrunk rate. Reporting only; never rank on this."""
        return self.verified_successes / self.attempts if self.attempts else 0.0

    @property
    def low_coverage(self) -> bool:
        return self.attempts < LOW_COVERAGE_ATTEMPTS

    def key(self) -> tuple[str, str, str, str]:
        return (self.model, self.bucket, self.harness, self.effort)

    def to_record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_version": self.model_version,
            "bucket": self.bucket,
            "harness": self.harness,
            "effort": self.effort,
            "attempts": self.attempts,
            "verified_successes": self.verified_successes,
            "raw_success": round(self.raw_success, 4),
            "posterior_success": round(self.posterior_success(), 4),
            "posterior_stddev": round(self.posterior_stddev(), 4),
            "low_coverage": self.low_coverage,
            "tool_call_validity": round(self.tool_call_validity, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "median_tokens": self.median_tokens,
            "median_runtime_s": round(self.median_runtime_s, 2),
            "estimated_cost": round(self.estimated_cost, 4),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CapabilityRecord:
        return cls(
            model=str(record["model"]),
            bucket=str(record["bucket"]),
            harness=str(record.get("harness", "")),
            model_version=str(record.get("model_version", "")),
            effort=str(record.get("effort", "default")),
            attempts=int(record.get("attempts", 0)),
            verified_successes=int(record.get("verified_successes", 0)),
            tool_call_validity=float(record.get("tool_call_validity", 0.0)),
            recovery_rate=float(record.get("recovery_rate", 0.0)),
            median_tokens=int(record.get("median_tokens", 0)),
            median_runtime_s=float(record.get("median_runtime_s", 0.0)),
            estimated_cost=float(record.get("estimated_cost", 0.0)),
            extra=record.get("extra") or {},
        )


class CapabilityDB:
    """Capability records keyed by (model, bucket, harness, effort).

    A model version is never overwritten by its successor: `qwen3.8-max-2026-08-02` and
    `qwen3.8-max-2026-09-15` are separate candidates with separate histories. Merging
    them would let a regression in a new build hide behind the old build's record.
    """

    def __init__(self, records: list[CapabilityRecord] | None = None) -> None:
        self._records: dict[tuple[str, str, str, str], CapabilityRecord] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: CapabilityRecord) -> None:
        self._records[record.key()] = record

    def get(self, model: str, bucket: str, *, harness: str = "", effort: str = "default") -> CapabilityRecord | None:
        return self._records.get((model, bucket, harness, effort))

    def estimate(self, model: str, bucket: str, *, harness: str = "", effort: str = "default") -> float:
        """Shrunk success estimate, falling back to the prior for an unseen bucket.

        An unmeasured pairing scores the prior mean rather than 0.0: never having been
        tried is not evidence of failure, and scoring it as such would freeze a new
        expert out of the traffic it needs to prove itself.
        """
        record = self.get(model, bucket, harness=harness, effort=effort)
        if record is None:
            return DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
        return record.posterior_success()

    def coverage(self, model: str, bucket: str, *, harness: str = "", effort: str = "default") -> int:
        record = self.get(model, bucket, harness=harness, effort=effort)
        return record.attempts if record else 0

    def rank(
        self,
        models: list[str],
        bucket: str,
        *,
        harness: str = "",
        effort: str = "default",
    ) -> list[tuple[str, float]]:
        """Candidates ordered by shrunk estimate, ties broken by name for determinism."""
        scored = [(m, self.estimate(m, bucket, harness=harness, effort=effort)) for m in models]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))

    def buckets(self) -> tuple[str, ...]:
        return tuple(sorted({key[1] for key in self._records}))

    def models(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._records}))

    def __len__(self) -> int:
        return len(self._records)

    # --- persistence ---------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        return [self._records[k].to_record() for k in sorted(self._records)]

    def save(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.to_records():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(self._records)

    @classmethod
    def load(cls, path: Path) -> CapabilityDB:
        db = cls()
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    db.add(CapabilityRecord.from_record(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return db

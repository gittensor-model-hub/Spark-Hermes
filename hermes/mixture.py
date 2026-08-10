"""Dataset mixture: what a corpus is supposed to be made of, and whether it is.

OpenHermes' contribution was not architecture -- it was selection. Which examples got in,
which were dropped, how the mix was balanced. The same discipline applied to trajectories
is what separates "100k verified Hermes trajectories" from "whatever the generators
happened to produce most of".

A `DatasetMix` is a declared target: named buckets with weights summing to 1. Given an
actual corpus it reports, per bucket, what share it holds against what it should.

The load-bearing decision is what happens when a bucket is **short**. Silently drawing the
shortfall from elsewhere produces a corpus that does not match its own specification and
where nobody notices -- you set out to build 40% coding and shipped 12%, and the model's
weakness at coding looks like a training mystery rather than a supply problem. So
`plan_sample` reports the shortfall and refuses to backfill, and `validate` reports drift
per bucket rather than one aggregate distance.

Everything here is counting. It does not read trajectories or judge quality; it answers
"is this corpus the thing we said we were building", which is a question nobody asks until
after the training run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Buckets for the first agent corpus. These are what a Hermes worker has to be good at,
# not what happens to be easy to collect -- the difference is the whole point of
# declaring a target before assembling a corpus.
AGENT_BUCKETS = (
    "coding",
    "terminal",
    "research",
    "browser",
    "recovery",
    "tool_use",
    "memory",
    "long_horizon",
    "verification",
)

# Tolerance before a bucket counts as off-target. Wide enough that ordinary sampling
# noise does not raise an alarm, narrow enough that a systematically starved bucket does.
DEFAULT_TOLERANCE = 0.05


class MixtureError(ValueError):
    """A mixture specification is malformed."""


@dataclass(frozen=True)
class DatasetMix:
    """Named buckets with target proportions."""

    name: str
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise MixtureError(f"{self.name}: a mixture needs at least one bucket")
        negative = [b for b, w in self.weights.items() if w < 0]
        if negative:
            raise MixtureError(f"{self.name}: negative weight for {negative}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise MixtureError(f"{self.name}: weights sum to {total:.4f}, expected 1.0")

    @property
    def buckets(self) -> tuple[str, ...]:
        return tuple(sorted(self.weights))

    def target_count(self, bucket: str, total: int) -> int:
        return round(self.weights.get(bucket, 0.0) * total)

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "weights": {k: round(v, 4) for k, v in sorted(self.weights.items())}}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> DatasetMix:
        return cls(
            name=str(record.get("name") or "mix"), weights={str(k): float(v) for k, v in record["weights"].items()}
        )


# The first agent corpus, per the OpenHermes-style "one excellent dataset" strategy:
# start narrow and deep rather than spreading across every domain at once.
AGENT_DATASET_V1 = DatasetMix(
    name="spark-hermes-agent-v1",
    weights={
        "coding": 0.40,
        "terminal": 0.20,
        "research": 0.20,
        "browser": 0.10,
        "recovery": 0.10,
    },
)


@dataclass(frozen=True)
class BucketReport:
    bucket: str
    have: int
    target: int
    share: float
    target_share: float

    @property
    def drift(self) -> float:
        """Actual share minus target share. Negative means starved."""
        return self.share - self.target_share

    @property
    def shortfall(self) -> int:
        return max(0, self.target - self.have)

    def to_record(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "have": self.have,
            "target": self.target,
            "share": round(self.share, 4),
            "target_share": round(self.target_share, 4),
            "drift": round(self.drift, 4),
            "shortfall": self.shortfall,
        }


@dataclass(frozen=True)
class MixReport:
    mix: str
    total: int
    buckets: tuple[BucketReport, ...]
    unknown_buckets: dict[str, int] = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def off_target(self) -> tuple[BucketReport, ...]:
        return tuple(b for b in self.buckets if abs(b.drift) > self.tolerance)

    @property
    def starved(self) -> tuple[BucketReport, ...]:
        """Buckets below target beyond tolerance -- where collection effort is needed."""
        return tuple(b for b in self.buckets if b.drift < -self.tolerance)

    @property
    def on_spec(self) -> bool:
        return not self.off_target and not self.unknown_buckets

    def to_record(self) -> dict[str, Any]:
        return {
            "mix": self.mix,
            "total": self.total,
            "on_spec": self.on_spec,
            "tolerance": self.tolerance,
            "buckets": [b.to_record() for b in self.buckets],
            "starved": [b.bucket for b in self.starved],
            "unknown_buckets": dict(sorted(self.unknown_buckets.items())),
        }


def validate(counts: dict[str, int], mix: DatasetMix, *, tolerance: float = DEFAULT_TOLERANCE) -> MixReport:
    """Compare an actual corpus against its declared mixture.

    Drift is reported per bucket rather than as a single distance. One number would say
    the corpus is 12% off without saying which half of it is missing, and the two are not
    interchangeable when deciding what to generate next.

    Buckets present in the corpus but absent from the spec are reported separately rather
    than folded in: unplanned data is not automatically bad, but it should be a decision.
    """
    total = sum(counts.values())
    reports = []
    for bucket in mix.buckets:
        have = counts.get(bucket, 0)
        reports.append(
            BucketReport(
                bucket=bucket,
                have=have,
                target=mix.target_count(bucket, total),
                share=have / total if total else 0.0,
                target_share=mix.weights[bucket],
            )
        )
    unknown = {b: n for b, n in counts.items() if b not in mix.weights and n}
    return MixReport(
        mix=mix.name,
        total=total,
        buckets=tuple(reports),
        unknown_buckets=unknown,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class SamplePlan:
    """How many rows to draw per bucket, and where supply runs out."""

    mix: str
    requested: int
    draw: dict[str, int]
    shortfall: dict[str, int]

    @property
    def achievable(self) -> int:
        return sum(self.draw.values())

    @property
    def complete(self) -> bool:
        return not self.shortfall

    def to_record(self) -> dict[str, Any]:
        return {
            "mix": self.mix,
            "requested": self.requested,
            "achievable": self.achievable,
            "complete": self.complete,
            "draw": dict(sorted(self.draw.items())),
            "shortfall": dict(sorted(self.shortfall.items())),
        }


def plan_sample(available: dict[str, int], mix: DatasetMix, target_size: int) -> SamplePlan:
    """Plan a corpus of `target_size` rows, without backfilling a starved bucket.

    A shortfall is reported, never quietly covered from a bucket with spare rows. Filling
    a 40%-coding target from whatever else is lying around yields a corpus that claims a
    balance it does not have, and the resulting model's weakness reads as a training
    mystery rather than the supply problem it is.
    """
    if target_size < 0:
        raise MixtureError("target_size cannot be negative")

    draw: dict[str, int] = {}
    shortfall: dict[str, int] = {}
    for bucket in mix.buckets:
        want = mix.target_count(bucket, target_size)
        have = available.get(bucket, 0)
        draw[bucket] = min(want, have)
        if have < want:
            shortfall[bucket] = want - have
    return SamplePlan(mix=mix.name, requested=target_size, draw=draw, shortfall=shortfall)


def count_by(items: Iterable[Any], key: str = "bucket") -> dict[str, int]:
    """Tally items by a bucket attribute or dict key. Unlabelled items count as ``""``."""
    counts: dict[str, int] = {}
    for item in items:
        bucket = item.get(key, "") if isinstance(item, dict) else getattr(item, key, "")
        counts[str(bucket or "")] = counts.get(str(bucket or ""), 0) + 1
    return counts

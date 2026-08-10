"""What work costs: token accounting, prices, and a budget that can actually stop.

Nothing in this repo knew what anything cost. `CandidateRun.cost` exists and is a
tie-break in `select_winner`; `Budget` exists on every `AgentModule` and is copied into
every `RouteDecision`; `EpisodeMetrics.tokens_used` is aggregated into `mean_tokens`. Not
one of them is written by any code path, so the cost tie-break has been comparing 0.0
against 0.0 and `mean_tokens` is a column of zeros.

The usage data was already there. `teacher/providers.py` captures the provider's own
`usage` object verbatim into `Trajectory.metadata` on every generated row and nothing has
ever read it. This module reads it.

**An unpriced model must not cost zero.** That is the central refusal, and it is not
fastidiousness: `select_winner` breaks ties on lowest cost, so a model nobody priced would
win every cost tie-break in the tournament, become the SFT winner, and have its rival
demoted to a DPO rejection -- on the strength of having no price. Silence is not cheapness.
`PriceBook.price_of` raises.

**Cached input is not input, and the two provider shapes disagree about which.** OpenAI's
`prompt_tokens` *includes* `prompt_tokens_details.cached_tokens`; Anthropic's
`input_tokens` *excludes* `cache_read_input_tokens`. Summing them the same way overcharges
one and undercharges the other, and since cached input is an order of magnitude cheaper the
error is largest exactly where the saving is largest -- on the long, stable system prompts
an agent harness sends every single turn. Normalising this correctly is most of the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

ANTHROPIC = "anthropic"
OPENAI = "openai"

PROVIDER_SHAPES = (ANTHROPIC, OPENAI)

# Keys that identify each shape. Disjoint between the two, so a usage object declared as
# the wrong shape can be caught rather than parsed into zeros. Membership of *any* key is
# enough on purpose: Anthropic's streaming `message_delta` carries only `output_tokens` and
# must still parse. `total_tokens` appears in both dialects and so discriminates nothing.
SHAPE_KEYS = {
    ANTHROPIC: frozenset({"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}),
    OPENAI: frozenset({"prompt_tokens", "completion_tokens", "prompt_tokens_details"}),
}

# Rates are per million tokens, in the price book's currency.
PER_MILLION = 1_000_000


class CostError(ValueError):
    """A cost cannot be computed, or would be misleading if it were."""


class UnpricedModel(CostError):
    """No price is on file for this model.

    Its own class because it is the one a caller may reasonably want to catch: a new model
    appearing in a tournament is an operational fact, not a bug, and the response is to add
    a price rather than to treat the run as free.
    """


@dataclass(frozen=True)
class Usage:
    """Normalised token counts: how many were fresh, cached, written to cache, produced.

    `input_tokens` is always the *uncached* remainder, whatever the provider called it.
    Keeping the fields in provider terms would push the reconciliation onto every caller,
    and the whole failure this module exists to prevent is each caller reconciling it
    slightly differently.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"):
            value = getattr(self, name)
            # `nan < 0` is False, so a sign test alone lets NaN through -- and one NaN
            # makes `spent` NaN, which makes `spent >= limit` False forever. The ledger
            # then reports itself as never exhausted and stops refusing anything, which is
            # the exact opposite of what a budget is for. It also makes `to_record()` emit
            # bare NaN, which is invalid JSON that Python's own decoder happens to accept.
            if not math.isfinite(value):
                raise CostError(f"{name} is {value!r}; usage counts must be finite")
            if value < 0:
                raise CostError(f"{name} is negative; usage counts cannot be negative")

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.cache_write_tokens

    @property
    def total(self) -> int:
        return self.total_input + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input that was served from cache. 0.0 when there was no input."""
        return self.cached_input_tokens / self.total_input if self.total_input else 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "total": self.total,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


def usage_from_provider(raw: dict[str, Any], *, shape: str) -> Usage:
    """Normalise a provider's `usage` object.

    The two shapes disagree about whether the headline input number already contains the
    cached tokens, and getting it backwards is silent -- the total still looks plausible,
    it is just wrong by the size of the discount. So the shape is required rather than
    sniffed from the keys present: a provider that omits its cache fields on an uncached
    request looks exactly like the other shape, and guessing would be right most of the
    time and wrong on the requests that matter.
    """
    if shape not in PROVIDER_SHAPES:
        raise CostError(f"unknown provider shape {shape!r}; expected one of {list(PROVIDER_SHAPES)}")
    if raw is not None and not isinstance(raw, dict):
        raise CostError(f"usage object must be a mapping, got {type(raw).__name__}")
    if not raw:
        return Usage()
    if not raw.keys() & SHAPE_KEYS[shape]:
        # Without this, a usage object read under the wrong shape parses to all zeros and a
        # real model costs 0.0 -- which then wins `select_winner`'s cost tie-break against
        # every model that was measured honestly. A mis-declared shape must be as loud as
        # an unpriced model, and for the same reason.
        raise CostError(
            f"usage object {sorted(raw)} has none of the {shape} keys {sorted(SHAPE_KEYS[shape])}; "
            "it is not that shape, and pricing it would silently cost zero"
        )

    if shape == ANTHROPIC:
        # `input_tokens` already excludes cache reads and cache writes.
        return Usage(
            input_tokens=int(raw.get("input_tokens", 0)),
            cached_input_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0)),
        )

    # OpenAI-compatible: `prompt_tokens` is the whole prompt, cached tokens included.
    prompt = int(raw.get("prompt_tokens", 0))
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    # Some OpenAI-compatible gateways report cache writes here too. Like cached reads they
    # are part of the prompt, so they are subtracted rather than added; absent, this is a
    # no-op and the behaviour is unchanged.
    written = int(details.get("cache_write_tokens", 0) or 0)
    if cached + written > prompt:
        raise CostError(
            f"cached ({cached}) + written ({written}) exceeds prompt_tokens ({prompt}); in the "
            "OpenAI shape both are subsets of the prompt, so this usage object is not that shape"
        )
    return Usage(
        input_tokens=prompt - cached - written,
        cached_input_tokens=cached,
        cache_write_tokens=written,
        output_tokens=int(raw.get("completion_tokens", 0)),
    )


@dataclass(frozen=True)
class Price:
    """Per-million-token rates for one model.

    `cached_input` defaults to the full input rate rather than to zero. A model whose
    provider offers no cache discount is the normal case, and defaulting to free would
    quietly credit a discount that was never given.
    """

    model: str
    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None
    currency: str = "USD"
    # When these rates were taken. Costs computed under different revisions are not
    # comparable, and this is what lets `Cost` say so.
    effective: str = ""

    def __post_init__(self) -> None:
        for name in ("input", "output", "cached_input", "cache_write"):
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(value):
                raise CostError(f"{self.model}: {name} rate is {value!r}; rates must be finite")
            if value < 0:
                raise CostError(f"{self.model}: {name} rate is negative")

    @property
    def cached_rate(self) -> float:
        return self.input if self.cached_input is None else self.cached_input

    @property
    def cache_write_rate(self) -> float:
        return self.input if self.cache_write is None else self.cache_write


@dataclass(frozen=True)
class Cost:
    """What one unit of work cost, and under which price revision."""

    total: float
    currency: str
    revision: str
    components: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "currency": self.currency,
            "revision": self.revision,
            "components": {k: round(v, 6) for k, v in sorted(self.components.items())},
        }


def cost_of(usage: Usage, price: Price, *, revision: str = "") -> Cost:
    """Price one usage record, keeping the components apart.

    Reported per component because the aggregate hides the decision a reader wants to make.
    A run whose cost is nine-tenths cached input is telling you the harness is re-sending a
    stable prefix and the cache is working; the same total made of fresh input says the
    opposite, and both print the same number.
    """
    if usage.cache_write_tokens and price.cache_write is None:
        # Cache writes are billed at a *premium*, not at the input rate: Anthropic charges
        # 1.25x for the five-minute TTL and 2x for the hour. Defaulting to the base rate
        # would undercharge by 20-50% on exactly the turn that primes a long agent prefix,
        # and since the TTL is not visible in the usage object no single default is right.
        raise CostError(
            f"{price.model}: {usage.cache_write_tokens} cache-write tokens but no cache_write rate "
            "on file; cache writes are billed above the input rate, so assuming one undercharges"
        )
    components = {
        "input": usage.input_tokens * price.input / PER_MILLION,
        "cached_input": usage.cached_input_tokens * price.cached_rate / PER_MILLION,
        "cache_write": usage.cache_write_tokens * price.cache_write_rate / PER_MILLION,
        "output": usage.output_tokens * price.output / PER_MILLION,
    }
    return Cost(
        total=sum(components.values()),
        currency=price.currency,
        revision=revision or price.effective,
        components=components,
    )


@dataclass
class PriceBook:
    """Prices on file, and a refusal for everything else."""

    revision: str
    prices: dict[str, Price] = field(default_factory=dict)

    def add(self, price: Price) -> PriceBook:
        """Record a price, refusing to re-price a model already on file.

        `cheaper()` refuses to compare costs from different revisions, which only means
        anything if a revision names one rate card. Overwriting in place lets charges made
        before and after the swap carry the same revision label and different rates, so the
        comparison it protects becomes exactly the thing it was meant to prevent.
        """
        if price.model in self.prices:
            raise CostError(
                f"{price.model!r} already has a price in revision {self.revision!r}; re-pricing it "
                "in place would leave two rate cards sharing one revision label"
            )
        self.prices[price.model] = price
        return self

    def price_of(self, model: str) -> Price:
        """The model's price, or a refusal.

        Never a zero. `select_winner` breaks ties on lowest cost, so an unpriced model
        returning 0.0 would win every cost tie-break against every priced rival, take the
        SFT slot, and push the model it beat into the DPO rejections -- on the strength of
        being unknown rather than cheap.
        """
        price = self.prices.get(model)
        if price is None:
            raise UnpricedModel(
                f"no price on file for {model!r} in revision {self.revision!r}; treating it as "
                "free would make the model nobody priced the cheapest one in every comparison"
            )
        return price

    def cost(self, model: str, usage: Usage) -> Cost:
        return cost_of(usage, self.price_of(model), revision=self.revision)

    def knows(self, model: str) -> bool:
        return model in self.prices


def cheaper(a: Cost, b: Cost) -> Cost:
    """The lower of two costs, refusing to compare across revisions or currencies.

    Prices move. Two totals computed months apart are two different questions, and picking
    the smaller one silently answers "which model ran when rates were lower".
    """
    if a.currency != b.currency:
        raise CostError(f"cannot compare {a.currency} against {b.currency} without a conversion rate")
    if a.revision != b.revision:
        raise CostError(
            f"costs come from price revisions {a.revision!r} and {b.revision!r}; comparing them "
            "measures when the work ran rather than what it cost"
        )
    return a if a.total <= b.total else b


@dataclass
class Ledger:
    """Accumulated spend against a ceiling.

    The ceiling is a refusal rather than a warning. A budget that only reports overrun is a
    log line someone reads afterwards, and the runs it would have stopped have already been
    paid for.
    """

    price_book: PriceBook
    limit: float | None = None
    currency: str = "USD"
    spent: float = 0.0
    usage: Usage = field(default_factory=Usage)
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return float("inf") if self.limit is None else max(0.0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def would_exceed(self, cost: Cost) -> bool:
        return self.limit is not None and self.spent + cost.total > self.limit

    def charge(self, model: str, usage: Usage, *, label: str = "") -> Cost:
        """Record work and its cost, refusing to book past the ceiling.

        The check is *before* recording, so a ledger never holds a total it was not allowed
        to reach. Callers that want to overrun by one job should ask `would_exceed` and
        decide; they do not get to find out by having already spent it.
        """
        if self.exhausted:
            raise CostError(f"budget of {self.limit} {self.currency} is exhausted; {model!r} was not charged")
        cost = self.price_book.cost(model, usage)
        if cost.currency != self.currency:
            raise CostError(f"{model!r} is priced in {cost.currency}, ledger is in {self.currency}")
        if self.would_exceed(cost):
            raise CostError(
                f"charging {cost.total:.6f} for {model!r} would exceed the remaining "
                f"{self.remaining:.6f} {self.currency}"
            )
        self.spent += cost.total
        self.usage = self.usage + usage
        self.entries.append({"model": model, "label": label, **cost.to_record(), "usage": usage.to_record()})
        return cost

    def to_record(self) -> dict[str, Any]:
        return {
            "revision": self.price_book.revision,
            "currency": self.currency,
            "limit": self.limit,
            "spent": round(self.spent, 6),
            "remaining": None if self.limit is None else round(self.remaining, 6),
            "exhausted": self.exhausted,
            "entries": len(self.entries),
            "usage": self.usage.to_record(),
        }

"""The base model, pinned to a revision rather than to a name.

Phase 0 is pipeline validation and it needs a model that exists. `Qwen/Qwen3.8-27B` does
not: as of 2026-08-10 the only repositories under that name on the Hub are third-party
derivatives with no official base, which cannot be pinned or verified and should not be
trained against. `Qwen/Qwen3.6-27B` does exist, and the recipes now point at it.

**A name is not a pin.** `base_model: Qwen/Qwen3.6-27B` resolves to whatever that
repository holds when someone runs it. `eval.hf_pin` already refuses movable refs on the
mining side for exactly that reason; the base model was the one place still naming a
repository without saying which commit of it. Two runs that agree on every other digest
this project computes could still have trained on different weights.

## What the pin records, and why each field is here

`revision` is the whole point -- a 40-character commit, checked by `eval.hf_pin`.

`hermes_dialect` is recorded with its evidence rather than asserted. The repository's own
chat template emits `<tool_call>`, `<tools>`, `<tool_response>` and `<think>`, and does not
emit `<scratch_pad>`. That is the Hermes 4 shape, established by reading the model rather
than by choosing for it -- which matters, because the project's rule is that Hermes is
upstream and the model is what adapts.

`multimodal: true` is recorded because it is easy to miss and changes things. This is a
`Qwen3_5ForConditionalGeneration` with a vision tower, so "27B" is not 27B of text
parameters, the text hyperparameters live under `text_config`, and any memory estimate
that reads the top level of `config.json` silently gets `None` for every field.

`kv_bytes_per_token` is derived from those hyperparameters and stored so a budget can be
checked without a network call: 2 x 64 layers x 4 KV heads x 256 head_dim. A budget that
says `input_tokens: 200000` without saying whether that is cumulative or peak context
cannot be checked against a card at all -- the two readings differ by more than an order
of magnitude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIN_PATH = Path(__file__).resolve().parent / "base_model.json"


class BaseModelError(ValueError):
    """The base-model pin is missing, malformed, or names a ref that cannot be pinned."""


@dataclass(frozen=True)
class BaseModel:
    repository: str
    revision: str
    hermes_dialect: str
    chat_template: str
    multimodal: bool
    kv_bytes_per_token: dict[str, int]
    raw: dict[str, Any]

    @property
    def at_revision(self) -> str:
        """How to name this model anywhere a human will read it."""
        return f"{self.repository}@{self.revision[:12]}"

    def kv_bytes(self, context_tokens: int, *, dtype: str = "fp8") -> int:
        """KV cache bytes for a peak context, so a budget can be checked against a card."""
        per_token = self.kv_bytes_per_token.get(dtype)
        if per_token is None:
            raise BaseModelError(f"no KV size recorded for dtype {dtype!r}; have {sorted(self.kv_bytes_per_token)}")
        return per_token * context_tokens


def load(path: Path | None = None) -> BaseModel:
    """Read the pin and refuse anything that is not actually pinned."""
    from eval.hf_pin import check_revision

    source = path or PIN_PATH
    if not source.is_file():
        raise BaseModelError(f"no base-model pin at {source}")
    record = json.loads(source.read_text(encoding="utf-8"))

    repository = str(record.get("repository") or "")
    if "/" not in repository:
        raise BaseModelError(f"repository {repository!r} is not an org/name Hub id")

    # The same refusal the mining side already makes. A movable ref here would mean two
    # runs agreeing on every other digest and still training on different weights.
    issues = check_revision(record.get("revision"), field="base model revision")
    if issues:
        raise BaseModelError("; ".join(issues))

    return BaseModel(
        repository=repository,
        revision=str(record["revision"]),
        hermes_dialect=str(record.get("hermes_dialect") or ""),
        chat_template=str(record.get("chat_template") or ""),
        multimodal=bool(record.get("multimodal")),
        kv_bytes_per_token={k: int(v) for k, v in (record.get("kv_bytes_per_token") or {}).items()},
        raw=record,
    )


__all__ = ["PIN_PATH", "BaseModel", "BaseModelError", "load"]

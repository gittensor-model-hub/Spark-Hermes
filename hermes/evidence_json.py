"""Strict JSON at competition evidence boundaries, before assertions can be lost."""

from __future__ import annotations

import json
import math
from typing import Any


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def evidence_value(raw: bytes | str) -> Any:
    """Decode one complete JSON value without losing contradictory assertions.

    Callers read a snapshot once and use these same bytes for checks and provenance.
    JSON payload strings remain strings: transcripts are never recursively decoded.
    Arrays are valid provider responses; endpoint callers still check their shape.
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds decoder limit") from exc


def evidence_object(raw: bytes | str) -> dict[str, Any]:
    """Decode one complete object; reject duplicate keys recursively and nonfinite values."""
    value = evidence_value(raw)
    if not isinstance(value, dict):
        raise ValueError("evidence record must be a JSON object")
    return value


def evidence_records(text: str) -> list[dict[str, Any]]:
    """Decode every nonblank JSONL record before returning any metadata.

    Only newline separates records; Unicode line separators inside JSON strings
    are ordinary payload, not record boundaries.
    """
    records = []
    for number, line in enumerate(text.split("\n"), start=1):
        if line.strip(" \t\r"):
            try:
                records.append(evidence_object(line))
            except ValueError as exc:
                raise ValueError(f"line {number}: invalid JSON object metadata: {exc}") from exc
    return records

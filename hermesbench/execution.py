"""Expected execution identity within a trusted operator process.

Generated context files and argv are transport. This scope preserves the resolved
expectation independently until the runner checks its final captured inputs. It
is not an admission seal or isolation from arbitrary code in the same process.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from hermes.evidence_json import evidence_object


def context_snapshot(context: dict[str, Any]) -> str:
    """An immutable, type-sensitive snapshot of all resolved context fields."""
    return json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ExpectedExecution:
    bundle_path: Path | None
    transport_path: Path
    context_json: str

    def checked_context(self, bundle_path: Path | None, transport_path: Path | None) -> dict[str, Any]:
        """Check transport once, then return the independently preserved snapshot."""
        if (bundle_path.resolve() if bundle_path is not None else None) != self.bundle_path:
            raise ValueError("runner miner directory differs from expected execution")
        if transport_path is None or transport_path.is_symlink() or transport_path.resolve() != self.transport_path:
            raise ValueError("runner context transport missing or replaced")
        observed = evidence_object(transport_path.read_bytes())
        if context_snapshot(observed) != self.context_json:
            raise ValueError("runner context transport differs from expected execution")
        return evidence_object(self.context_json)


_expected_execution: ContextVar[ExpectedExecution | None] = ContextVar("runner_expected_execution", default=None)


def expected_execution() -> ExpectedExecution | None:
    return _expected_execution.get()


@contextmanager
def execution_scope(bundle_path: Path | None, transport_path: Path, context: dict[str, Any]) -> Iterator[None]:
    """Bind a resolved expectation for synchronous calls, including CLI wrappers.

    A thread adapter must explicitly forward its context to runner_for, which
    establishes this scope in that thread. No process propagation is implied.
    """
    expected = ExpectedExecution(
        bundle_path.resolve() if bundle_path is not None else None, transport_path.resolve(), context_snapshot(context)
    )
    token = _expected_execution.set(expected)
    try:
        yield
    finally:
        _expected_execution.reset(token)

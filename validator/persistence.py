"""Local durable writes. Locks cover read/modify/write, including across processes."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hermes.evidence_json import evidence_object


def state_identity(root: Path, *, mode: str | None = None, namespace: str | None = None) -> dict[str, str]:
    """An immutable trust domain, created explicitly as fixture or defaulting to production."""
    with locked(root / ".identity.lock"):
        path = root / ".identity"
        if path.exists():
            value = evidence_object(path.read_bytes())
            if (
                not isinstance(value, dict)
                or set(value) != {"mode", "namespace", "issuer"}
                or value["mode"] not in ("fixture", "production")
                or any(not isinstance(v, str) or not v for v in value.values())
            ):
                raise ValueError("invalid state root identity")
            if mode is not None and mode != value["mode"] or namespace is not None and namespace != value["namespace"]:
                raise ValueError("state root mode/namespace is immutable")
            return value
        if mode not in (None, "production", "fixture") or namespace == "":
            raise ValueError("invalid state root mode/namespace")
        value = {"mode": mode or "production", "namespace": namespace or "default", "issuer": uuid.uuid4().hex}
        atomic_write(path, json.dumps(value, sort_keys=True).encode())
        return value


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)

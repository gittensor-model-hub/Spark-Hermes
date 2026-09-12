"""Small, atomic records binding pipeline stages to the files they actually used."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class StageError(RuntimeError):
    """A stage cannot run with the available inputs."""


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    from hermes.evidence_json import evidence_object

    try:
        data = evidence_object(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise StageError(f"{path}: missing or invalid JSON record") from exc
    if not isinstance(data, dict):
        raise StageError(f"{path}: expected a JSON object")
    return data


def write_record(path: Path, data: dict[str, Any]) -> None:
    from validator.persistence import atomic_write

    atomic_write(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def same_domain(left: dict[str, Any], right: dict[str, Any]) -> None:
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or any(
            left.get(k) != right.get(k) or not isinstance(left.get(k), str) or not left[k]
            for k in ("mode", "namespace")
        )
    ):
        raise StageError("artifact trust mode/namespace differs from configured authority")


class AuthorityStore:
    """Private operator authority, with immutable role and issuer-bound records.

    Only trusted producers may call put. JSON exports are never imported as authority.
    Like RoundStore, this protects data ingress, not arbitrary local code/DB writers.
    """

    def __init__(self, root: Path, *, role: str, mode: str | None = None, namespace: str | None = None):
        from admin.runtime_protocol import historical_reading
        from validator.persistence import state_identity

        self.root = root.resolve()
        if historical_reading():
            self.identity = read_record(self.root / ".identity")
            if (
                set(self.identity) != {"mode", "namespace", "issuer"}
                or self.identity["mode"] not in {"production", "fixture"}
                or any(not isinstance(v, str) or not v for v in self.identity.values())
                or (mode is not None and self.identity["mode"] != mode)
                or (namespace is not None and self.identity["namespace"] != namespace)
            ):
                raise StageError("invalid original authority identity")
        else:
            self.identity = state_identity(self.root, mode=mode, namespace=namespace)
        self.role = role
        self.path = self.root / "authority.sqlite3"
        with self.connect() as db:
            binding = canonical({"identity": self.identity, "role": role})
            if not historical_reading():
                db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, value BLOB NOT NULL)")
                db.execute("INSERT OR IGNORE INTO metadata VALUES ('binding', ?)", (binding,))
            row = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if row is None or row[0] != binding:
                raise StageError("authority database role/issuer mismatch")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        from admin.runtime_protocol import historical_reading

        db = (
            sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
            if historical_reading()
            else sqlite3.connect(self.path, timeout=60)
        )
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def envelope(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Construct an issuer-bound record; it is authority only after commit."""
        record = {
            "schema": "spark-authority-v1",
            "kind": kind,
            "issuer": self.identity,
            "role": self.role,
            "payload": payload,
        }
        return {"id": content_digest(record), **record}

    def put(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        from admin.runtime_protocol import authority_write

        record = self.envelope(kind, payload)
        with authority_write(self), self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?)",
                (record["id"], canonical({k: v for k, v in record.items() if k != "id"})),
            )
        return record

    def kind(self, identifier: str) -> str:
        from hermes.evidence_json import evidence_object

        with self.connect() as db:
            row = db.execute("SELECT value FROM records WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise StageError("record has no committed authority in configured store")
        kind = evidence_object(row[0])["kind"]
        self.get(identifier, kind=kind)
        return kind

    def get(self, identifier: str, *, kind: str) -> dict[str, Any]:
        from hermes.evidence_json import evidence_object

        with self.connect() as db:
            row = db.execute("SELECT value FROM records WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise StageError("record has no committed authority in configured store")
        record = evidence_object(row[0])
        if (
            content_digest(record) != identifier
            or record.get("issuer") != self.identity
            or record.get("role") != self.role
            or record.get("kind") != kind
        ):
            raise StageError("authority record identity/kind changed")
        return {"id": identifier, **record}

    def records(self, *, kind: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            identifiers = [row[0] for row in db.execute("SELECT id FROM records ORDER BY id")]
        # Check every stored record, including records of other kinds.
        result = []
        from hermes.evidence_json import evidence_object

        for identifier in identifiers:
            with self.connect() as db:
                row = db.execute("SELECT value FROM records WHERE id=?", (identifier,)).fetchone()
            record = self.get(identifier, kind=evidence_object(row[0])["kind"])
            if record["kind"] == kind:
                result.append(record)
        return result


def checkpoint_files(path: Path) -> dict[str, str]:
    """Hash one complete supported checkpoint representation and its tokenizer.

    A single Hugging Face weight file or one index is supported. The index is the
    inventory, including nonstandard shard basenames; a glob of present shards is
    never evidence that a checkpoint is complete. Hub-cache blob symlinks remain
    usable here; derived-artifact consumers separately forbid symlinks.
    """
    from hermes.evidence_json import evidence_object

    snapshots = {}

    def checkpoint_record(file: Path) -> dict[str, Any]:
        try:
            raw = file.read_bytes()
            record = evidence_object(raw)
        except (OSError, ValueError) as exc:
            raise StageError(f"{file}: missing or invalid JSON record") from exc
        snapshots[file.name] = hashlib.sha256(raw).hexdigest()
        return record

    config = checkpoint_record(path / "config.json")
    if not config.get("model_type"):
        raise StageError(f"{path}: model config has no model_type")
    indexes = set(path.glob("*.index.json"))
    weights = {p for pattern in ("*.safetensors", "pytorch_model*.bin") for p in path.glob(pattern)}
    single = {path / name for name in ("model.safetensors", "pytorch_model.bin")}
    if not indexes:
        if not weights:
            raise StageError(f"{path}: checkpoint contains no weight files")
        if len(weights) != 1 or not weights <= single:
            raise StageError(f"{path}: unindexed shards or ambiguous/unsupported checkpoint layout")
    else:
        supported = {"model.safetensors.index.json": ".safetensors", "pytorch_model.bin.index.json": ".bin"}
        if len(indexes) != 1 or next(iter(indexes)).name not in supported or weights & single:
            raise StageError(f"{path}: ambiguous or unsupported checkpoint indexes/representations")
        index = next(iter(indexes))
        weight_map = checkpoint_record(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or any(not k for k in weight_map):
            raise StageError(f"{index}: missing weight_map")
        referenced = set()
        for name in weight_map.values():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or "\\" in name
                or not name.endswith(supported[index.name])
                or not (path / name).is_file()
            ):
                raise StageError(f"{index}: missing or unsafe weight shard {name!r}")
            referenced.add(path / name)
        if weights - referenced:
            raise StageError(f"{index}: checkpoint has weight files omitted from its index")
        # Numbered shards additionally declare a complete series, independently of
        # the tensor map. Refuse missing numbers and conflicting series/totals.
        numbered = [re.fullmatch(r"(.+)-(\d+)-of-(\d+)(\.[^.]+)", p.name) for p in referenced]
        if any(numbered):
            if not all(numbered):
                raise StageError(f"{index}: inconsistent numbered shard layout")
            parts = [m.groups() for m in numbered if m is not None]
            prefix, _, total, suffix = parts[0]
            if (
                int(total) != len(parts)
                or any((p, t, s) != (prefix, total, suffix) for p, _, t, s in parts)
                or {int(n) for _, n, _, _ in parts} != set(range(1, len(parts) + 1))
            ):
                raise StageError(f"{index}: incomplete or inconsistent numbered shards")
        weights = referenced
    files = weights | set(indexes) | {path / "config.json"}
    files.update(
        p
        for pattern in ("*token*.json", "*.jinja", "*.model", "vocab.*", "merges.txt", "generation_config.json")
        for p in path.glob(pattern)
    )
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise StageError(f"{path}: checkpoint contains missing, non-regular or empty files")
    # Bind the exact config/index bytes whose semantics selected the inventory.
    return {p.name: snapshots[p.name] if p.name in snapshots else file_digest(p) for p in sorted(files)}

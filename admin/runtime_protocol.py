"""Internal campaign serialization and bounded historical verification scope.

Historical scope is read-only. It is entered only by the transition verifier, never
by a candidate/parent caller flag, and never changes the executing runtime identity.
"""

from __future__ import annotations

import fcntl
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from admin.artifacts import StageError

PROTOCOL = "spark-runtime-transition-v1"
_held = threading.local()
_history: ContextVar[Any] = ContextVar("spark_historical_verification", default=None)
_graph: ContextVar[Any] = ContextVar("spark_verification_graph", default=None)


@contextmanager
def campaign_lock(root: Path):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    owner = os.getpid()
    held = getattr(_held, "roots", set()) if getattr(_held, "pid", None) == owner else set()
    if held:
        yield
        return
    with Path("/tmp/spark-hermes-campaign-writes.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        _held.pid = owner
        _held.roots = held | {root}
        try:
            yield
        finally:
            # A forked child never owns its parent's reentrancy or lock release.
            if os.getpid() == owner:
                _held.roots = held
                fcntl.flock(stream, fcntl.LOCK_UN)


def historical_reading():
    return _history.get() is not None


def writable(authority) -> None:
    if _history.get() is not None:
        raise StageError("historical verification is read-only")
    with authority.store.connect() as db:
        row = db.execute("SELECT successor FROM runtime_cutover WHERE singleton=1").fetchone()
    if row:
        raise StageError("source campaign retired by supported cutover; resume/use successor " + row[0])


def campaign_writer(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        authority = getattr(self, "release", self)
        with campaign_lock(authority.store.root):
            writable(authority)
            return method(self, *args, **kwargs)

    return run


def campaign_owners(store):
    from admin.artifacts import content_digest
    from hermes.evidence_json import evidence_object

    with store.connect() as db:
        rows = list(db.execute("SELECT key,value FROM metadata WHERE key LIKE 'campaign:%' ORDER BY key"))
    result = []
    for key, raw in rows:
        binding = evidence_object(raw)
        if set(binding) != {"root", "identity"} or key != "campaign:" + content_digest(binding):
            raise StageError("campaign ownership metadata changed")
        result.append(binding)
    return result


def bind_owner(store, binding):
    from admin.artifacts import canonical, content_digest

    with store.connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO metadata VALUES (?,?)", ("campaign:" + content_digest(binding), canonical(binding))
        )


def bind_campaign(store, authority):
    bind_owner(store, {"root": str(authority.store.root), "identity": authority.identity})


def check_owner(binding):
    import sqlite3

    from admin.artifacts import read_record

    root = Path(binding["root"])
    if read_record(root / ".identity") != binding["identity"]:
        raise StageError("original campaign issuer changed")
    with sqlite3.connect((root / "authority.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_cutover'").fetchone():
            if db.execute("SELECT 1 FROM runtime_cutover WHERE singleton=1").fetchone():
                raise StageError("source campaign retired; preparation/workspace cannot start a divergent campaign")


def preparation_store(workspace):
    from admin.artifacts import AuthorityStore

    return AuthorityStore(
        workspace.root / "preparation-authority",
        role="preparation",
        mode=workspace.identity["mode"],
        namespace=workspace.identity["namespace"],
    )


def check_workspace(workspace):
    from admin.artifacts import read_record

    if historical_reading():
        raise StageError("historical verification is read-only")
    if (workspace.root / "preparation-authority/authority.sqlite3").exists():
        for binding in campaign_owners(preparation_store(workspace)):
            check_owner(binding)
    for stage in ("sft", "dpo"):
        path = workspace.models / stage / "prepared.json"
        if path.exists():
            parent = read_record(path).get("parent")
            if parent and "root" in parent and "identity" in parent:
                check_owner({"root": parent["root"], "identity": parent["identity"]})


def preparation_writer(method):
    @wraps(method)
    def run(workspace, *args, **kwargs):
        from admin.artifacts import AuthorityStore

        with campaign_lock(workspace.root):
            check_workspace(workspace)
            parent_root = kwargs.get("release_root")
            binding = None
            if parent_root is not None:
                authority = AuthorityStore(Path(parent_root), role="release")
                binding = {"root": str(authority.root), "identity": authority.identity}
                check_owner(binding)
            result = method(workspace, *args, **kwargs)
            if binding:
                bind_owner(preparation_store(workspace), binding)
            return result

    return run


def candidate_writer(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with campaign_lock(self.store.root):
            owners = campaign_owners(self.store)
            for binding in owners:
                check_owner(binding)
            workspace = kwargs.get("workspace")
            if workspace is not None:
                check_workspace(workspace)
            result = method(self, *args, **kwargs)
            if workspace is not None:
                for binding in owners:
                    bind_owner(preparation_store(workspace), binding)
            return result

    return run


@contextmanager
def authority_write(store):
    if _history.get() is not None:
        raise StageError("historical verification is read-only")
    with campaign_lock(store.root):
        roots = [store.root]
        if store.role == "crossed-evaluation":
            roots.append(store.root.parent)
        if store.role == "candidate":
            roots.extend(Path(b["root"]) for b in campaign_owners(store))
        import sqlite3

        for root in roots:
            path = root / "authority.sqlite3"
            if not path.exists():
                continue
            with sqlite3.connect(path) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_cutover'").fetchone():
                    row = db.execute("SELECT successor FROM runtime_cutover WHERE singleton=1").fetchone()
                    if row:
                        raise StageError("source campaign retired; use its committed successor")
        yield


def resolution_runtime():
    from hermes.harness import crossed_runtime_identity

    history = _history.get()
    return history["runtime"] if history is not None else crossed_runtime_identity()


@contextmanager
def historical_scope(runtime, identity):
    token = _history.set({"runtime": runtime, "identity": identity, "cache": {}, "visiting": set()})
    try:
        yield _history.get()
    finally:
        _history.reset(token)


def candidate_runtime(candidate, executing):
    history = _history.get()
    expected = executing
    if history is not None:
        from admin.artifacts import same_domain

        same_domain(history["identity"], candidate["origin"])
        expected = history["runtime"]
    if candidate["runtime"] != expected:
        raise StageError("candidate evaluator/environment changed; retain the original runtime and use runtime propose")


def verification_call(method):
    """Memoize only inside one synchronous verification, never across execution.

    Every top-level read checks originals again. This avoids exponential ancestry
    traversal without turning a prior process/result into authority.
    """

    @wraps(method)
    def run(*args, **kwargs):
        if _history.get() is not None or _graph.get() is not None:
            return method(*args, **kwargs)
        token = _graph.set({"cache": {}, "visiting": set()})
        try:
            return method(*args, **kwargs)
        finally:
            _graph.reset(token)

    return run


def historical_checked(method):
    @wraps(method)
    @verification_call
    def run(self, identifier, *args, **kwargs):
        graph = _history.get() or _graph.get()
        key = (str(self.store.root), method.__qualname__, identifier)
        if _history.get() is not None and method.__qualname__ == "ReleaseAuthority.resolve_decision":
            graph.setdefault("strict_roots", set()).add(str(self.store.root))
        if key in graph["visiting"]:
            raise StageError("cyclic original approval/candidate lineage")
        if key in graph["cache"]:
            return graph["cache"][key]
        graph["visiting"].add(key)
        try:
            result = method(self, identifier, *args, **kwargs)
            graph["cache"][key] = result
            return result
        finally:
            graph["visiting"].remove(key)

    return run
